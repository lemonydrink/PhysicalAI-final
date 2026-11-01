"""Pinned LeRobot v2.1 -> per-episode HDF5 with explicit temporal codec."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .io import ROOT, read_json, write_json, digest, config
from .codec import reconstruct


def normalize(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def resolve_tasks():
    c = config()
    raw = ROOT / 'data/raw'
    tasks = read_json(ROOT / 'configs/tasks.json')
    labels = {t['task_index']: t['task'] for t in (json.loads(l) for l in (raw/'meta/tasks.jsonl').read_text().splitlines())}
    # Source is already available locally. No edits to PARC are made.
    libero = ROOT/'third_party/libero/libero/libero'
    for task in tasks['tasks']:
        label = labels[task['dataset_task_index']]
        if normalize(label) != normalize(task['instruction']):
            raise ValueError(f'dataset instruction mismatch: {task["key"]}: {label}')
        task['dataset_instruction'] = label
        slug = label.replace(' ', '_')
        matches = [p for p in (libero/'bddl_files').glob('*/'+slug+'.bddl') if p.parent.name in ('libero_object','libero_goal','libero_spatial')]
        if len(matches) != 1:
            raise ValueError(f'expected one unperturbed BDDL for {label}, found {matches}')
        bddl = matches[0]
        init = libero/'init_files'/bddl.parent.name/(slug+'.pruned_init')
        if not init.exists():
            raise FileNotFoundError(init)
        task.update(suite=bddl.parent.name, environment=slug, bddl_sha256=digest(bddl), initial_states_sha256=digest(init))
    tasks.update(dataset_revision=c['dataset_revision'], environment_scope='unperturbed_base_tasks',
                 source_metadata_sha256=digest(raw/'meta/tasks.jsonl'))
    write_json(ROOT/'data/tasks.resolved.json', tasks)
    return tasks


def prepare(max_per_task=0, validation_fraction=0.1, output_dir=None):
    import av
    import h5py
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    destination = Path(output_dir) if output_dir else ROOT/'data'
    if (destination/'manifest.json').exists():raise FileExistsError('choose a new --output-dir; dataset manifests are immutable')
    if max_per_task<0 or not 0<validation_fraction<1:raise ValueError('invalid dataset sampling/split')
    c = config(); tasks = resolve_tasks(); codec = read_json(ROOT/'configs/codec.json')
    info = read_json(ROOT/'data/raw/meta/info.json')
    selected = {t['dataset_instruction']: t for t in tasks['tasks']}
    episodes = [json.loads(l) for l in (ROOT/'data/raw/meta/episodes.jsonl').read_text().splitlines()]
    rows = []; counts = {k: 0 for k in selected}
    def get(path):
        return Path(hf_hub_download(c['dataset_repo'], filename=path, repo_type='dataset', revision=c['dataset_revision'], local_dir=ROOT/'data/raw'))
    def video(path):
        with av.open(str(path)) as container:
            return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    for e in episodes:
        matching = [x for x in e['tasks'] if x in selected]
        if not matching:
            continue
        if len(matching) != 1 or len(e['tasks']) != 1:
            raise ValueError('mixed-task episode requires explicit handling')
        label = matching[0]
        if max_per_task and counts[label] >= max_per_task:
            continue
        task = selected[label]; idx = int(e['episode_index'])
        values = dict(episode_chunk=idx//info['chunks_size'], episode_index=idx)
        parquet = get(info['data_path'].format(**values))
        table = pq.read_table(parquet).to_pydict()
        if set(table['task_index']) != {task['dataset_task_index']}:
            raise ValueError('parquet task mismatch')
        state, action = np.asarray(table['observation.state']), np.asarray(table['action'])
        timestamps = np.asarray(table['timestamp'])
        if len(timestamps) != len(state) or not np.allclose(np.diff(timestamps), 1/info['fps'], atol=1e-4):
            raise ValueError('invalid timestamps/fps')
        targets, indices = reconstruct(state, action, codec['state_action_alignment'])
        front_path = get(info['video_path'].format(**values, video_key='observation.images.front'))
        wrist_path = get(info['video_path'].format(**values, video_key='observation.images.wrist'))
        front, wrist = video(front_path), video(wrist_path)
        if front.shape != (len(state),256,256,3) or wrist.shape != front.shape:
            raise ValueError('video/frame alignment mismatch')
        # Stable episode-group split; all windows from an episode stay together.
        bucket = int(hashlib.sha256(f'{c["seed"]}:{idx}'.encode()).hexdigest()[:8],16)/2**32
        split = 'validation' if bucket < validation_fraction else 'train'
        out = destination/f'processed/episode_{idx:06d}.h5'
        if out.exists():
            raise FileExistsError(f'do not overwrite dataset artifact: {out}')
        out.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(out,'x') as f:
            f.create_dataset('front',data=front[indices],compression='lzf')
            f.create_dataset('wrist',data=wrist[indices],compression='lzf')
            f.create_dataset('targets',data=targets)
            f.create_dataset('timestamps',data=timestamps[indices])
            f.create_dataset('source_state',data=state)
            f.create_dataset('source_action',data=action)
            f.attrs['instruction'] = label
            f.attrs['codec_hash'] = digest(ROOT/'configs/codec.json')
        rows.append({'episode_index':idx,'task':task['key'],'split':split,'file':str(out),
                     'sha256':digest(out),'frames':len(targets),'source_sha256':digest(parquet),
                     'source_front_sha256':digest(front_path),'source_wrist_sha256':digest(wrist_path)})
        counts[label]+=1
    if not all(counts.values()):
        raise ValueError(f'missing task demonstrations: {counts}')
    # For small smoke datasets, guarantee a validation episode per task.
    for key in {row['task'] for row in rows}:
        group=sorted([row for row in rows if row['task']==key],key=lambda row:hashlib.sha256(f'{c["seed"]}:{row["episode_index"]}'.encode()).hexdigest())
        if len(group)>=2:
            count=max(1,round(len(group)*validation_fraction))
            for i,row in enumerate(group):row['split']='validation' if i<count else 'train'
    result = {'schema_version':1,'dataset_revision':c['dataset_revision'],'codec_hash':digest(ROOT/'configs/codec.json'),
              'action_dt':1/info['fps'],'partial':bool(max_per_task),'counts':counts,'episodes':rows}
    write_json(destination/'manifest.json',result,exclusive=True)
    return result


def split_manifest(source, output):
    """Derive a new immutable, task-stratified episode split without copying videos."""
    import copy
    m=copy.deepcopy(read_json(source))
    groups={}
    for e in m['episodes']:groups.setdefault(e['task'],[]).append(e)
    if len(groups)!=8 or any(len(v)<2 for v in groups.values()):
        raise ValueError('need >=2 episodes per task to create both splits')
    for task,rows in groups.items():
        ordered=sorted(rows,key=lambda e:hashlib.sha256(f'{config()["seed"]}:{e["episode_index"]}'.encode()).hexdigest())
        n=max(1,round(len(rows)*.1))
        for index,e in enumerate(ordered):e['split']='validation' if index<n else 'train'
    m['split_parent_sha256']=digest(source)
    m['split_method']='task-stratified episode hash, >=1 validation episode per task; variant lineage unknown'
    write_json(output,m,exclusive=True)
    return m

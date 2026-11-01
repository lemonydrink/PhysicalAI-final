"""Build resumable same-time SmolVLA data directly from verified raw episodes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path


def prepare_raw(source, output, resume=False):
    import av
    import h5py
    import numpy as np
    import pyarrow.parquet as pq
    from run import read_json, write_json, digest, settings
    source = Path(source); output = Path(output)
    m = read_json(source); source_hash = digest(source)
    if m.get('partial') is not False or len(m['counts']) != 8:
        raise ValueError('requires verified full eight-task acquisition manifest')
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists():
        result = read_json(output/'manifest.json')
        if result['source_manifest_hash'] != source_hash:
            raise ValueError('prepared source changed')
        for row in result['episodes']:
            if digest(row['file']) != row['sha256']:
                raise ValueError('prepared data changed')
        return output/'manifest.json'
    groups = {}
    for ep in m['episodes']:
        groups.setdefault(ep['task'], []).append(ep)
    splits = {}
    for task, episodes in groups.items():
        ordered = sorted(episodes, key=lambda e: hashlib.sha256(f'{settings()["seed"]}:{e["episode_index"]}'.encode()).hexdigest())
        n = max(1, round(len(ordered)*.1))
        if len(ordered) <= n:
            raise ValueError('missing train split')
        splits.update({e['episode_index']: 'validation' if i < n else 'train' for i, e in enumerate(ordered)})
    raw = Path(m['raw_root'])
    def convert(ep):
        idx = ep['episode_index']; target = output/f'episode_{idx:06d}.h5'; sidecar = target.with_suffix('.json')
        if sidecar.exists():
            row = read_json(sidecar)
            if row['source_manifest_hash'] != source_hash or digest(target) != row['sha256']:
                raise ValueError('inconsistent prepared episode')
            return row
        for path in ep['paths']:
            if digest(raw/path) != ep['files'][path]['sha256']:
                raise ValueError('raw data changed')
        table = pq.read_table(raw/ep['paths'][0]).to_pydict()
        state = np.asarray(table['observation.state'], dtype=np.float32)
        action = np.asarray(table['action'], dtype=np.float32)
        n = ep['length']
        if state.shape != (n,8) or action.shape != (n,7) or set(table['task_index']) != {ep['task_index']}:
            raise ValueError('raw schema mismatch')
        temporary = target.with_suffix('.h5.tmp')
        with h5py.File(temporary,'w') as f:
            f.create_dataset('state',data=state);f.create_dataset('action',data=action)
            f.create_dataset('timestamps',data=table['timestamp'])
            for camera,path in zip(('front','wrist'),ep['paths'][1:]):
                dataset=f.create_dataset(camera,shape=(n,256,256,3),dtype='uint8',chunks=(1,256,256,3),compression='lzf')
                count=0
                with av.open(str(raw/path)) as container:
                    container.streams.video[0].codec_context.thread_count=1
                    for frame in container.decode(video=0):
                        dataset[count]=frame.to_ndarray(format='rgb24');count+=1
                if count!=n:raise ValueError('video length mismatch')
            f.attrs['instruction']=ep['tasks'][0]
        temporary.replace(target)
        row=dict(episode_index=idx,task=ep['task'],split=splits[idx],file=str(target.resolve()),frames=n,
                 sha256=digest(target),source_manifest_hash=source_hash)
        write_json(sidecar,row)
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(convert,e) for e in m['episodes']]):
            rows.append(future.result())
            if len(rows)%100==0 or len(rows)==len(m['episodes']):
                print(f'prepared {len(rows)}/{len(m["episodes"])}',flush=True)
    arrays={'observation.state':[],'action':[]}
    for row in rows:
        if row['split']=='train':
            with h5py.File(row['file'],'r') as f:
                arrays['observation.state'].append(f['state'][()]);arrays['action'].append(f['action'][()])
    stats={}
    for key,values in arrays.items():
        a=np.concatenate(values).astype(np.float64)
        stats[key]={'mean':a.mean(0).tolist(),'std':np.maximum(a.std(0),1e-6).tolist()}
    result=dict(model_family='smolvla',source_manifest_hash=source_hash,episodes=sorted(rows,key=lambda e:e['episode_index']),stats=stats,
                action_representation='relative_osc',alignment='state[t],action[t],images[t]',fps=20,
                chunk_size=settings()['chunk_size'],partial=False,
                split_method='seed42 task-stratified episode hash; 10% validation; augmentation lineage unavailable')
    write_json(output/'manifest.json',result,exclusive=True)
    return output/'manifest.json'

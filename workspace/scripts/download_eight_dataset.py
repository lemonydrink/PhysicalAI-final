"""Acquire all eight selected tasks, preserving pinned original LeRobot files.

Run with the project .venv. Restarting resumes downloads and revalidates files.
This only acquires data; it does not train or upload experiment information.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import os
import threading
from urllib.parse import quote
os.environ["HF_HUB_DISABLE_XET"] = "1"

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download, HfApi
    import av
    import numpy as np
    import pyarrow.parquet as pq
    config = read(ROOT / 'configs/project.json')
    raw = ROOT / 'data/raw'
    repo, revision = config['dataset_repo'], config['dataset_revision']
    # Refresh the metadata from the same immutable revision before selecting data.
    for name in ('info.json', 'tasks.jsonl', 'episodes.jsonl'):
        hf_hub_download(repo, f'meta/{name}', repo_type='dataset', revision=revision, local_dir=raw)
    info = read(raw / 'meta/info.json')
    tasks = read(ROOT / 'data/tasks.resolved.json')['tasks']
    labels = {json.loads(l)['task_index']: json.loads(l)['task'] for l in (raw / 'meta/tasks.jsonl').read_text().splitlines()}
    selected = {}
    for task in tasks:
        if labels[task['dataset_task_index']] != task['dataset_instruction']:
            raise ValueError('task metadata differs from resolved task selection')
        selected[task['dataset_instruction']] = task
    episodes = []
    counts = {t['key']: 0 for t in tasks}
    for line in (raw / 'meta/episodes.jsonl').read_text().splitlines():
        ep = json.loads(line)
        matches = set(ep['tasks']) & selected.keys()
        if not matches:
            continue
        if len(ep['tasks']) != 1:
            raise ValueError('mixed task episode')
        task = selected[ep['tasks'][0]]
        counts[task['key']] += 1
        values = dict(episode_index=ep['episode_index'], episode_chunk=ep['episode_index'] // info['chunks_size'])
        paths = [info['data_path'].format(**values)]
        paths += [info['video_path'].format(**values, video_key=f'observation.images.{camera}') for camera in ('front', 'wrist')]
        episodes.append(dict(**ep, task=task['key'], task_index=task['dataset_task_index'], paths=paths))
    if not all(counts.values()) or len(counts) != 8:
        raise ValueError('missing selected task')
    print(json.dumps(dict(status='selected', episodes=len(episodes), frames=sum(e['length'] for e in episodes), counts=counts)), flush=True)
    # Server blob IDs provide independent integrity verification, including reused local files.
    remote_path = ROOT / 'data/eight-tasks-remote-blobs.json'
    if remote_path.exists():
        cached = read(remote_path)
        if cached['revision'] != revision or cached['repo'] != repo:
            raise ValueError('remote metadata revision mismatch')
        remote = cached['files']
    else:
        for attempt in range(30):
            try:
                metadata = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
                break
            except Exception:
                if attempt == 29:
                    raise
                print(json.dumps(dict(status='metadata_retry', attempt=attempt + 1)), flush=True)
                time.sleep(30)
        remote = {s.rfilename: dict(size=s.size, blob_id=s.blob_id, lfs_sha256=s.lfs.sha256 if s.lfs else None) for s in metadata.siblings}
        remote_path.write_text(json.dumps(dict(repo=repo, revision=revision, files=remote)))
    all_paths = [p for e in episodes for p in e['paths']]
    sessions = threading.local()
    def acquire(path):
        source = remote[path]
        error = None
        for attempt in range(30):
            try:
                local = raw / path
                # Immutable content is checked below; valid local files need no network request.
                if not local.exists() or attempt > 0:
                    import requests
                    if not hasattr(sessions, 'http'):
                        sessions.http = requests.Session()
                    url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{quote(path)}'
                    with sessions.http.get(url, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        local.parent.mkdir(parents=True, exist_ok=True)
                        temporary = local.with_suffix(local.suffix + '.acquiring')
                        with temporary.open('wb') as stream:
                            for block in response.iter_content(1024 * 1024):
                                stream.write(block)
                        temporary.replace(local)
                size = local.stat().st_size
                if size != source['size']:
                    raise ValueError(f'size mismatch {path}')
                digest = sha(local)
                if source['lfs_sha256']:
                    if digest != source['lfs_sha256']:
                        raise ValueError(f'SHA256 mismatch {path}')
                else:
                    git = hashlib.sha1(f'blob {size}\0'.encode() + local.read_bytes()).hexdigest()
                    if git != source['blob_id']:
                        raise ValueError(f'git blob mismatch {path}')
                return path, dict(bytes=size, sha256=digest)
            except Exception as exc:
                error = exc
                if attempt == 0 or attempt % 5 == 0:
                    print(json.dumps(dict(status='file_retry', path=path, attempt=attempt + 1, error=type(exc).__name__)), flush=True)
                time.sleep(30)
        raise RuntimeError(f'failed {path}') from error
    verified = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(acquire, p) for p in all_paths]
        for future in as_completed(futures):
            path, record = future.result()
            verified[path] = record
            if len(verified) % 100 == 0 or len(verified) == len(all_paths):
                print(json.dumps(dict(status='download_verified', files=len(verified), total=len(all_paths))), flush=True)
    # Validate schema, timestamps and both complete decoded camera streams locally.
    def validate(ep):
        table = pq.read_table(raw / ep['paths'][0]).to_pydict()
        n = ep['length']
        if len(table['action']) != n or set(table['task_index']) != {ep['task_index']}:
            raise ValueError(f'episode/task mismatch {ep["episode_index"]}')
        state, action = np.asarray(table['observation.state']), np.asarray(table['action'])
        if state.shape != (n, 8) or action.shape != (n, 7) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('invalid state/action')
        if not np.allclose(np.diff(table['timestamp']), 1 / info['fps'], atol=1e-4):
            raise ValueError('invalid timestamps')
        for path in ep['paths'][1:]:
            with av.open(str(raw / path)) as container:
                container.streams.video[0].codec_context.thread_count = 1
                count = 0
                for frame in container.decode(video=0):
                    if (frame.width, frame.height) != (256, 256):
                        raise ValueError('invalid video shape')
                    count += 1
            if count != n:
                raise ValueError(f'video frame mismatch {path}: {count} != {n}')
        return dict(**ep, files={p: verified[p] for p in ep['paths']})
    rows = []
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as pool:
        futures = [pool.submit(validate, e) for e in episodes]
        for future in as_completed(futures):
            rows.append(future.result())
            if len(rows) % 100 == 0 or len(rows) == len(episodes):
                print(json.dumps(dict(status='decoded_verified', episodes=len(rows), total=len(episodes))), flush=True)
    manifest = dict(schema_version=1, dataset_repo=repo, dataset_revision=revision,
                    raw_root=str(raw), task_set='draft_eight', partial=False,
                    counts=counts, total_episodes=len(rows), total_frames=sum(e['length'] for e in rows),
                    total_files=len(verified), total_bytes=sum(f['bytes'] for f in verified.values()),
                    fps=info['fps'], state_dim=8, action_dim=7,
                    validation='remote blob hash/size, parquet task/schema/finiteness/timestamps, full two-camera decode/frame count',
                    metadata_sha256={p: sha(raw / 'meta' / p) for p in ('info.json', 'tasks.jsonl', 'episodes.jsonl')},
                    episodes=sorted(rows, key=lambda e: e['episode_index']))
    target = ROOT / 'data/eight-tasks-local.manifest.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(dict(status='complete', manifest=str(target), bytes=manifest['total_bytes'], episodes=len(rows))), flush=True)


if __name__ == '__main__':
    main()

"""Count all preregistered simulation trials and match eight success videos."""
from __future__ import annotations
import math
from pathlib import Path


def wilson(success, total):
    if not total:
        return None
    z = 1.959963984540054
    p, den = success / total, 1 + z*z / total
    mid = (p + z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/den
    return [mid-half, mid+half]


def summarize(records, task_keys, per_task=10, target=0.7, require_videos=True):
    if type(per_task) is not int or per_task < 1 or len(task_keys)!=8 or len(set(task_keys)) != 8 or not math.isfinite(target) or not 0<=target<=1:
        raise ValueError('requires eight unique tasks and positive trial count')
    groups = {key: [] for key in task_keys}
    seen, bundles, protocols = set(), set(), set()
    for r in records:
        identity = (r['task'], r['episode_id'])
        if r['task'] not in groups or identity in seen or type(r['episode_id']) is not int or not 0 <= r['episode_id'] < per_task:
            raise ValueError('unexpected or duplicate trial')
        if type(r['success']) is not bool or type(r.get('infra_error')) is not bool:
            raise ValueError('success/infra_error must be booleans')
        if r['success'] and (r['infra_error'] or r.get('intervention', False)):
            raise ValueError('intervened or infrastructure trial cannot count as success')
        if r.get('backend') != 'simulation':
            raise ValueError('evaluation backend must be simulation')
        if not r.get('bundle_hash') or not r.get('protocol_hash'):
            raise ValueError('missing bundle/protocol identity')
        bundles.add(r['bundle_hash']); protocols.add(r['protocol_hash'])
        groups[r['task']].append(r); seen.add(identity)
    if len(bundles) > 1 or len(protocols) > 1:
        raise ValueError('mixed models or protocols')
    rows = []
    for key, trials in groups.items():
        s = sum(r['success'] for r in trials)
        video = any(r['success'] and r.get('video') and Path(r['video']).is_file() and Path(r['video']).stat().st_size > 0 for r in trials)
        # All planned trials are the denominator, including missing/infra trials.
        rows.append({'task': key, 'completed_trials': len(trials), 'planned_trials': per_task,
                     'successes': s, 'success_rate': s/per_task, 'wilson95': wilson(s, per_task), 'success_video': video,
                     'infra_errors': sum(r['infra_error'] for r in trials),
                     'collision_free_success_rate': sum(r.get('collision_free_success',False) for r in trials)/per_task if all(r.get('collision_available',False) for r in trials) and len(trials)==per_task else None})
    macro = sum(row['success_rate'] for row in rows)/8
    complete = all(row['completed_trials'] == per_task for row in rows)
    videos = all(row['success_video'] for row in rows)
    return {'backend': 'simulation', 'macro_success_rate': macro, 'target': target, 'tasks': rows,
            'all_trials_present': complete, 'all_success_videos_present': videos,
            'accepted': complete and macro >= target and (videos or not require_videos)}


def log_evaluation(directory, mode='offline'):
    import os
    import wandb
    from .io import read_json,digest,config
    directory=Path(directory);summary=read_json(directory/'summary.json');protocol=read_json(directory/'protocol.json')
    c=config()
    with wandb.init(project=os.environ.get('WANDB_PROJECT',c['wandb_project']),entity=os.environ.get('WANDB_ENTITY'),
                    mode=mode,job_type='eval',group=directory.name,dir=str(directory),
                    config={'protocol':protocol,'protocol_hash':digest(directory/'protocol.json'),'summary_hash':digest(directory/'summary.json')}) as run:
        metrics={'eval/macro_success_rate':summary['macro_success_rate'],'eval/accepted':int(summary['accepted'])}
        for row in summary['tasks']:
            metrics[f'eval/{row["task"]}/success_rate']=row['success_rate']
            if row['collision_free_success_rate'] is not None:metrics[f'eval/{row["task"]}/collision_free_success_rate']=row['collision_free_success_rate']
        run.log(metrics,step=0)
        return run.id

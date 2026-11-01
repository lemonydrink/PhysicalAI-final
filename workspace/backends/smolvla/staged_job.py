#!/usr/bin/env python3
"""Audited public baseline -> one epoch SFT -> rollout-based checkpoint selection."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
from run import HERE,ROOT,read_json,write_json,digest


def status(job,stage,**extra):
    record=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    temporary=job/'status.json.tmp';write_json(temporary,record);temporary.replace(job/'status.json')
    print(json.dumps(record),flush=True)


def environment(job):
    plan=read_json(job/'plan.json');config=Path(plan['config_path'])
    if digest(config)!=plan['config_sha256']:raise ValueError('queued configuration changed')
    return dict(os.environ,SMOLVLA_CONFIG=str(config),SMOLVLA_STATE_ENCODING='libero_quat2axisangle',WANDB_MODE='offline',WANDB_PROJECT='physicalai-smolvla-eight',
                HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')


def command(args,env):subprocess.run([str(x) for x in args],cwd=ROOT,env=env,check=True)


def score_key(summary):
    rates=[t['success_rate'] for t in summary['tasks']]
    return (sum(t['successes'] for t in summary['tasks']),min(rates),sum(x>0 for x in rates))


def select(job):
    candidates=[]
    states=list(job.glob('train-*/ckpt-*/state.json'))
    initial=read_json(job/'plan.json').get('initial_resume')
    if initial:states.append(Path(initial)/'state.json')
    for state in states:
        record=read_json(state);evaluation=job/f'eval-{record["step"]:06d}'
        if (evaluation/'summary.json').exists():
            summary=read_json(evaluation/'summary.json')
            if summary['all_trials_present']:candidates.append((score_key(summary),-record['step'],state.parent,evaluation,summary))
    if not candidates:return
    best=max(candidates,key=lambda x:(x[0],x[1]))
    baseline=read_json(job/'baseline80/summary.json')
    write_json(job/'best_checkpoint.json',dict(checkpoint=str(best[2]),step=-best[1],evaluation=str(best[3]),macro_success_rate=best[4]['macro_success_rate'],
               task_success_rates={t['task']:t['success_rate'] for t in best[4]['tasks']},baseline_success_rate=baseline['macro_success_rate'],
               better_than_baseline=best[0]>score_key(baseline),selection_rule='max total successes; ties: max minimum task rate, max tasks with success, earlier step',
               evaluation_scope='checkpoint selection on paired 80 trials; not an independent held-out confirmation'))


def evaluate_checkpoint(job,checkpoint=None):
    env=environment(job);plan=read_json(job/'plan.json');manifest=Path(plan.get('evaluation_manifest_path',HERE/'data/full-v1/manifest.json'))
    step=read_json(checkpoint/'state.json')['step'] if checkpoint else 0
    output=job/f'eval-{step:06d}' if checkpoint else job/'baseline80'
    status(job,'checkpoint_evaluation' if checkpoint else 'baseline',checkpoint=str(checkpoint) if checkpoint else None,planned_trials=80,optimizer_step=step)
    args=[HERE/'.venv/bin/python',HERE/'run.py','evaluate','--manifest',manifest,'--output',output,'--url',f'http://127.0.0.1:{plan["sim_port"]}',
          '--episodes','10','--max-steps','300','--resume']
    if checkpoint:args+=['--checkpoint',checkpoint]
    command(args,env);command([ROOT/'.venv/bin/vla','score',output,'--per-task','10'],env)
    summary=read_json(output/'summary.json')
    if not summary['all_trials_present']:raise ValueError('paired rollout incomplete')
    command([ROOT/'.venv/bin/vla','log-eval',output,'--wandb-mode','offline'],env)
    if checkpoint:select(job)
    status(job,'training' if checkpoint else 'baseline_complete',optimizer_step=step,macro_success_rate=summary['macro_success_rate'])
    return summary


def main(job):
    job=job.resolve();lock=(job/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=environment(job);plan=read_json(job/'plan.json');manifest=Path(plan.get('training_manifest_path',HERE/'data/full-v1/manifest.json'));server=None
    if digest(manifest)!=plan['manifest_sha256']:raise ValueError('dataset split changed')
    audit=read_json(job/'audit/contract.json')
    if not audit.get('passed'):raise ValueError('input/loss audit must pass before baseline and training')
    try:
        status(job,'starting_simulator')
        server=subprocess.Popen([str(ROOT/'.venv/bin/vla'),'serve','--kind','sim','--port',str(plan['sim_port'])],cwd=ROOT,env=env)
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('simulator exited')
            try:
                with urlopen(f'http://127.0.0.1:{plan["sim_port"]}/health',timeout=2) as r:
                    if r.status==200:break
            except OSError:time.sleep(1)
        else:raise TimeoutError('simulator startup')
        if not (job/'baseline80/summary.json').exists():evaluate_checkpoint(job)
        candidates=[(read_json(p)['step'],p.parent) for p in job.glob('train-*/ckpt-*/state.json')]
        latest=max(candidates,key=lambda x:x[0]) if candidates else None
        if latest is None and plan.get('initial_resume'):
            initial=Path(plan['initial_resume']);latest=(read_json(initial/'state.json')['step'],initial)
        final_ready=latest and read_json(latest[1]/'state.json')['consumed_frames']==plan['train_frames']
        if not final_ready:
            attempt=1
            while (job/f'train-{attempt:03d}').exists():attempt+=1
            output=job/f'train-{attempt:03d}'
            status(job,'training',optimizer_step=latest[0] if latest else 0,total_optimizer_steps=plan['optimizer_steps'],batch_size=read_json(Path(plan['config_path']))['batch_size'],gradient_accumulation_steps=read_json(Path(plan['config_path']))['gradient_accumulation_steps'])
            args=[HERE/'.venv/bin/python',HERE/'staged_training.py','--manifest',manifest,'--output',output,'--job',job,'--wandb-mode','offline']
            if latest:args+=['--resume',latest[1]]
            command(args,env)
        else:evaluate_checkpoint(job,latest[1])
        select(job);best=read_json(job/'best_checkpoint.json')
        status(job,'complete',best_checkpoint=best['checkpoint'],best_success_rate=best['macro_success_rate'],baseline_success_rate=best['baseline_success_rate'],
               better_than_baseline=best['better_than_baseline'],optimizer_steps=plan['optimizer_steps'],epochs=1)
    except BaseException as exc:
        status(job,'failed',error=repr(exc));raise
    finally:
        if server:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    if args.checkpoint:evaluate_checkpoint(args.job.resolve(),args.checkpoint.resolve())
    else:main(args.job)

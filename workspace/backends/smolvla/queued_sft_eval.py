#!/usr/bin/env python3
"""One local durable job: prepare all data, SFT, then exactly 80 trials."""
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from physicalai_vla.io import read_json,write_json,digest


def main(job):
    job=Path(job).resolve();job.mkdir(parents=True,exist_ok=True)
    lock=(job/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read_json(job/'plan.json')
    config_path=Path(plan.get('config_path',HERE/'configs/train.json'))
    config=read_json(config_path)
    if digest(config_path)!=plan['config_sha256'] or digest(ROOT/'data/eight-tasks-local.manifest.json')!=plan['raw_manifest_sha256']:
        raise ValueError('queued settings or dataset changed')
    learner=HERE/'.venv/bin/python';core=ROOT/'.venv/bin/vla'
    manifest=HERE/'data/full-v1/manifest.json'
    env=dict(os.environ,SMOLVLA_CONFIG=str(config_path),WANDB_MODE='offline',WANDB_PROJECT=config['wandb_project'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4')
    def status(stage,**extra):
        record=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),**extra)
        temporary=job/'status.json.tmp';write_json(temporary,record);temporary.replace(job/'status.json')
        print(json.dumps(record),flush=True)
    def call(args):
        subprocess.run([str(x) for x in args],cwd=ROOT,env=env,check=True)
    server=None
    try:
        status('preparing',episodes=3071)
        call([learner,HERE/'run.py','prepare','--source',ROOT/'data/eight-tasks-local.manifest.json','--output',manifest.parent,'--resume'])
        checkpoints=[]
        for state in job.glob('train-*/ckpt-*/state.json'):
            record=read_json(state)
            if record['manifest_sha256']!=digest(manifest):raise ValueError('checkpoint dataset mismatch')
            checkpoints.append((record['step'],state.parent))
        latest=max(checkpoints,key=lambda x:x[0]) if checkpoints else None
        if latest is None and plan.get('initial_resume'):
            initial=Path(plan['initial_resume'])
            latest=(read_json(initial/'state.json')['step'],initial)
        if latest is None or latest[0]<plan['sft_steps']:
            attempt=1
            while (job/f'train-{attempt:03d}').exists():attempt+=1
            training=job/f'train-{attempt:03d}'
            status('training',batch_size=4,target_steps=plan['sft_steps'],resume_step=latest[0] if latest else 0,output=str(training))
            args=[learner,HERE/'run.py',plan.get('train_command','train'),'--manifest',manifest,'--output',training,'--wandb-mode','offline']
            if plan.get('train_command','train')=='train':args+=['--steps',str(plan['sft_steps'])]
            if latest:args+=['--resume',latest[1]]
            call(args)
            checkpoint=training/f'ckpt-{plan["sft_steps"]:06d}'
        else:checkpoint=latest[1]
        if read_json(checkpoint/'state.json')['step']!=plan['sft_steps']:raise ValueError('full SFT checkpoint missing')
        if plan.get('train_command')=='train-epoch' and not read_json(checkpoint/'state.json').get('validation_full_complete'):raise ValueError('full held-out validation missing')
        status('evaluating',checkpoint=str(checkpoint),planned_trials=80)
        server=subprocess.Popen([str(core),'serve','--kind','sim','--port',str(plan['sim_port'])],cwd=ROOT,env=env)
        url=f'http://127.0.0.1:{plan["sim_port"]}'
        for _ in range(120):
            if server.poll() is not None:raise RuntimeError('simulation worker exited')
            try:
                with urlopen(url+'/health',timeout=2) as response:
                    if response.status==200:break
            except OSError:time.sleep(1)
        else:raise TimeoutError('simulation worker did not become ready')
        evaluation=job/'evaluation80'
        call([learner,HERE/'run.py','evaluate','--manifest',manifest,'--output',evaluation,'--checkpoint',checkpoint,'--url',url,'--episodes','10','--max-steps','300','--resume'])
        call([core,'score',evaluation,'--per-task','10'])
        summary=read_json(evaluation/'summary.json')
        if not summary['all_trials_present']:raise RuntimeError('80 trial evaluation incomplete')
        # The same offline project records rollout metrics as well as train/validation metrics.
        call([core,'log-eval',evaluation,'--wandb-mode','offline'])
        status('complete',checkpoint=str(checkpoint),summary=str(evaluation/'summary.json'),
               completed_trials=80,macro_success_rate=summary['macro_success_rate'],target_met=summary['accepted'])
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:
        if server is not None:
            server.terminate()
            try:server.wait(timeout=20)
            except subprocess.TimeoutExpired:server.kill();server.wait()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True)
    main(parser.parse_args().job)

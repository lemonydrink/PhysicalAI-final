#!/usr/bin/env python3
"""Wait for a specified local experiment to finish before starting the next one."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run import HERE,ROOT,read_json,write_json,digest


def predecessor_ready(record,active_state,allow_user_stop=False):
    stage=record.get('stage')
    return active_state=='inactive' and (stage=='complete' or (allow_user_stop and stage=='stopped_by_user'))


def status(job,stage,**extra):
    record=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    temporary=job/'status.json.tmp';write_json(temporary,record);temporary.replace(job/'status.json')
    print(json.dumps(record),flush=True)


def preflight(job):
    import torch
    from run import load,normalization_stats,initial_directory,settings
    from epoch_training import FrameWindows,batch_for
    from staged_training import loss_seven
    plan=read_json(job/'plan.json');c=settings();manifest=Path(plan.get('training_manifest_path',HERE/'data/full-v1/manifest.json'))
    if digest(Path(plan['config_path']))!=plan['config_sha256'] or digest(manifest)!=plan['manifest_sha256']:raise ValueError('queued settings/data changed')
    policy,pre,post=load(read_json(manifest))
    from safetensors import safe_open
    state=policy.state_dict();matched=0
    with safe_open(initial_directory()/'model.safetensors',framework='pt',device='cpu') as source:
        for key in source.keys():
            if not torch.equal(state[key].detach().cpu(),source.get_tensor(key)):raise ValueError(f'initial tensor mismatch: {key}')
            matched+=1
    for p in policy.parameters():
        if p.requires_grad:p.data=p.data.float()
    ds=FrameWindows(read_json(manifest),'train',c['chunk_size']);raw=batch_for(ds,[0,len(ds)-1]);batch=pre(raw)
    stats=normalization_stats({},c)
    for key in ('observation.state','action'):
        restored=batch[key].cpu()*stats[key]['std']+stats[key]['mean']
        torch.testing.assert_close(restored,raw[key],atol=1e-6,rtol=1e-5)
    with torch.autocast('cuda',dtype=torch.bfloat16):total,n=loss_seven(policy,batch)
    if n!=2 or not torch.isfinite(total):raise ValueError('invalid seven-dimensional loss')
    (total/n).backward()
    grad=policy.model.action_out_proj.weight.grad
    if torch.count_nonzero(grad[7:]) or not torch.count_nonzero(grad[:7]):raise ValueError('loss padding leaked')
    record=dict(passed=True,optimizer_updates_during_audit=0,initial_model=c['initial_model'],initial_revision=c['initial_revision'],
                all_initial_tensors_match=True,tensor_count=matched,loss_dim=7,normalizer_roundtrip=True,trainable_weights_fp32=True,
                weight_sha256=digest(initial_directory()/'model.safetensors'),camera_contract='same audited canonical camera transforms as predecessor',
                predecessor_contract_sha256=digest(job/'audit/input-contract-inherited.json'),state_encoding=c['state_encoding'])
    write_json(job/'audit/contract.json',record);print(json.dumps(record),flush=True)


def main(job):
    job=job.resolve();lock=(job/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read_json(job/'plan.json');predecessor=Path(plan['predecessor_job']);unit=plan['predecessor_unit']
    while True:
        record=read_json(predecessor/'status.json') if (predecessor/'status.json').exists() else {}
        query=subprocess.run(['systemctl','--user','show',unit,'--property=ActiveState','--value'],capture_output=True,text=True)
        active=query.stdout.strip() if query.returncode==0 else 'unknown'
        if predecessor_ready(record,active,plan.get('start_immediately_after_user_stop',False)):break
        status(job,'waiting_for_predecessor',predecessor=str(predecessor),predecessor_stage=record.get('stage','unknown'),predecessor_service=active)
        time.sleep(30)
    config=Path(plan['config_path'])
    env=dict(os.environ,SMOLVLA_CONFIG=str(config),SMOLVLA_STATE_ENCODING='libero_quat2axisangle',WANDB_MODE='offline',
             WANDB_PROJECT=read_json(config)['wandb_project'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    status(job,'preflight')
    try:
        subprocess.run([str(HERE/'.venv/bin/python'),str(HERE/'deferred_experiment.py'),'--job',str(job),'--preflight'],cwd=ROOT,env=env,check=True)
        # The active backend changes only after the previous job has fully exited.
        path=ROOT/'configs/active_backend.json';active=read_json(path)
        active.update(config=str(config.relative_to(ROOT)),entrypoint='backends/smolvla/staged_job.py',current_job=str(job),initial_model=read_json(config)['initial_model'],
                      initial_revision=read_json(config)['initial_revision'],batch_size=32,gradient_accumulation_steps=1,effective_batch_size=32,performance_verified=False)
        write_json(path,active)
        subprocess.run([str(HERE/'.venv/bin/python'),str(HERE/'staged_job.py'),'--job',str(job)],cwd=ROOT,env=env,check=True)
    except BaseException as exc:
        status(job,'failed',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    if args.preflight:preflight(args.job.resolve())
    else:main(args.job)

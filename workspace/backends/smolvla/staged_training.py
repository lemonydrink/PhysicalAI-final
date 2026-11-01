"""One-epoch FP32/BF16 action-expert SFT with accumulation and paired rollouts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from run import HERE,ROOT,settings,read_json,write_json,digest,load,initial_directory
from epoch_training import FrameWindows,batch_for,order_for


def scheduled_lr(step,total,warmup,peak=3e-5,minimum=3e-6):
    if step<=warmup:return peak*step/warmup
    progress=min(1.,(step-warmup)/(total-warmup))
    return minimum+(peak-minimum)*.5*(1+math.cos(math.pi*progress))


def reduce_valid_seven(losses,padding):
    losses=losses[...,:7].float()
    valid=(~padding).to(losses.dtype)
    lengths=valid.sum(1)
    sample=(losses*valid.unsqueeze(-1)).sum((1,2))/(7*lengths.clamp_min(1))
    real=lengths>0
    return (sample*real).sum(),int(real.sum())


def loss_seven(policy,batch):
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS,OBS_LANGUAGE_ATTENTION_MASK
    if policy.config.adapt_to_pi_aloha:raise ValueError('unexpected ALOHA representation')
    images,masks=policy.prepare_images(batch)
    losses=policy.model.forward(images,masks,batch[OBS_LANGUAGE_TOKENS],batch[OBS_LANGUAGE_ATTENTION_MASK],
                                policy.prepare_state(batch),policy.prepare_action(batch))
    return reduce_valid_seven(losses,batch['action_is_pad'])


def train(manifest_path,output,job,resume=None,mode=None):
    import wandb
    import time
    job=Path(job)
    queued=read_json(job/'plan.json')
    if queued.get('require_data_audit'):
        from staged_job import status
        report=Path(queued['data_audit_report'])
        while not report.exists():
            status(job,'waiting_for_data_audit',report=str(report));time.sleep(10)
        audit=read_json(report)
        if not audit.get('passed'):raise ValueError('dataset alignment audit has unresolved failures')
        if audit['config_sha256']!=digest(Path(os.environ['SMOLVLA_CONFIG'])) or audit['manifest_sha256']!=digest(manifest_path):raise ValueError('dataset audit no longer matches training inputs')
    c=settings();job=Path(job);output=Path(output);m=read_json(manifest_path)
    if m.get('partial') or c['batch_size']*c['gradient_accumulation_steps']!=32 or not torch.cuda.is_available():raise ValueError('full data, CUDA and effective batch32 required')
    for e in m['episodes']:
        if digest(e['file'])!=e['sha256']:raise ValueError('dataset changed')
    ds=FrameWindows(m,'train',c['chunk_size']);val=FrameWindows(m,'validation',c['chunk_size'])
    total=math.ceil(len(ds)/32);warmup=math.ceil(total*c['warmup_fraction'])
    if total>c['max_optimizer_steps'] or total!=c['steps']:raise ValueError('one epoch step plan changed')
    order=order_for(len(ds),c['seed']);order_hash=hashlib.sha256(order.numpy().tobytes()).hexdigest()
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    policy,pre,post=load(m,resume)
    params=[p for p in policy.parameters() if p.requires_grad]
    for p in params:p.data=p.data.float()
    if any(p.dtype!=torch.float32 for p in params):raise ValueError('trainable parameters must be FP32')
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=c['weight_decay'])
    cursor=0;step=0;batch_migration=None
    if resume:
        state=read_json(Path(resume)/'state.json')
        if state['manifest_sha256']!=digest(manifest_path) or state['order_sha256']!=order_hash:raise ValueError('resume data/order changed')
        if state['config_sha256']!=digest(Path(os.environ['SMOLVLA_CONFIG'])):
            previous=read_json(Path(resume).parent/'run.json')['config']
            drop={'batch_size','gradient_accumulation_steps'}
            if {k:v for k,v in previous.items() if k not in drop}!={k:v for k,v in c.items() if k not in drop} or previous['batch_size']*previous['gradient_accumulation_steps']!=32:raise ValueError('resume permits only equivalent batch/accumulation migration')
            batch_migration={'from_batch':previous['batch_size'],'from_accumulation':previous['gradient_accumulation_steps'],'to_batch':c['batch_size'],'to_accumulation':c['gradient_accumulation_steps']}
        cursor=state['consumed_frames'];step=state['step']
        restored=torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(restored['optimizer']);torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng'])
    output.mkdir(parents=True,exist_ok=False);torch.save(order,output/'epoch_order.pt')
    provenance=dict(config=c,manifest_sha256=digest(manifest_path),config_sha256=digest(Path(os.environ['SMOLVLA_CONFIG'])),
                    initial_weight_sha256=digest(initial_directory()/'model.safetensors'),resume=str(resume) if resume else None,
                    batch_migration=batch_migration,train_frames=len(ds),validation_frames=len(val),optimizer_steps=total,warmup_steps=warmup,trainable_parameters=sum(p.numel() for p in params),
                    trainable_dtypes=sorted({str(p.dtype) for p in params}),order_sha256=order_hash,code_sha256=digest(__file__),
                    entrypoint_sha256=digest(HERE/'run.py'),lock_sha256=digest(HERE/'uv.lock'),smoke=False)
    write_json(output/'run.json',provenance)
    selected=[];offset=0
    for key in sorted({e['task'] for e in val.episodes}):
        points=[];offset=0
        for e in val.episodes:
            if e['task']==key:points.extend([offset,offset+e['frames']//2,offset+e['frames']-1])
            offset+=e['frames']
        selected.extend(points[i] for i in np.linspace(0,len(points)-1,8,dtype=int))
    def validate(full):
        policy.eval();value=0.;count=0;indices=range(len(val)) if full else selected
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(c['seed'])
            for i in range(0,len(indices),4):
                batch=pre(batch_for(val,indices[i:i+4]))
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=loss_seven(policy,batch)
                value+=float(loss);count+=n
                if full and i%4000==0:print(f'full validation {count}/{len(val)}',flush=True)
        return {'validation/loss':value/count,'validation/samples':count,'validation/full':int(full)}
    def rollout(checkpoint):
        subprocess.run([str(HERE/'.venv/bin/python'),str(HERE/'staged_job.py'),'--job',str(job),'--checkpoint',str(checkpoint)],cwd=ROOT,check=True)
        return read_json(job/f'eval-{read_json(checkpoint/"state.json")["step"]:06d}'/'summary.json')
    # A saved checkpoint whose paired rollout was interrupted must be evaluated before proceeding.
    if resume and not (job/f'eval-{step:06d}'/'summary.json').exists():rollout(Path(resume))
    mode=mode or os.environ.get('WANDB_MODE',c['wandb_mode'])
    if mode not in ('online','offline','disabled'):raise ValueError('invalid W&B mode')
    with wandb.init(project=os.environ.get('WANDB_PROJECT',c['wandb_project']),entity=os.environ.get('WANDB_ENTITY'),mode=mode,dir=str(output),config=provenance,group=job.name) as tracking:
        while cursor<len(ds):
            stop=min(cursor+32,len(ds));group=order[cursor:stop].tolist();real_count=len(group);step+=1
            lr=scheduled_lr(step,total,warmup,c['learning_rate'],c['min_learning_rate'])
            for g in optimizer.param_groups:g['lr']=lr
            optimizer.zero_grad(set_to_none=True);policy.train();loss_sum=0.;micro_count=0
            for i in range(0,len(group),c['batch_size']):
                batch=pre(batch_for(ds,group[i:i+c['batch_size']],batch_size=c['batch_size']))
                with torch.autocast('cuda',dtype=torch.bfloat16):micro_loss,n=loss_seven(policy,batch);scaled=micro_loss/real_count
                if not torch.isfinite(scaled):raise RuntimeError('nonfinite loss')
                scaled.backward();loss_sum+=float(micro_loss.detach());micro_count+=1
            grad=torch.nn.utils.clip_grad_norm_(params,c['gradient_clip_norm'])
            if not torch.isfinite(grad):raise RuntimeError('nonfinite gradient')
            optimizer.step();optimizer.zero_grad(set_to_none=True);cursor=stop;final=cursor==len(ds)
            metrics={'train/loss':loss_sum/real_count,'learning_rate':lr,'global_step':step,'epoch':cursor/len(ds),
                     'coverage/consumed_frames':cursor,'coverage/total_frames':len(ds),'batch/real_samples':real_count,'batch/microbatches':micro_count,
                     'grad_norm':float(grad),'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            if step%c['validation_interval']==0 or final:metrics.update(validate(final))
            is_checkpoint=step%c['checkpoint_interval']==0 or final
            tracking.log(metrics,step=step,commit=not is_checkpoint)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            if is_checkpoint:
                checkpoint=output/f'ckpt-{step:06d}';policy.save_pretrained(checkpoint);pre.save_pretrained(checkpoint);post.save_pretrained(checkpoint)
                torch.save({'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},checkpoint/'training_state.pt')
                write_json(checkpoint/'state.json',dict(step=step,manifest_sha256=digest(manifest_path),config_sha256=provenance['config_sha256'],
                           order_sha256=order_hash,consumed_frames=cursor,total_frames=len(ds),validation_full_complete=final,selected_by_rollout=False))
                torch.cuda.empty_cache()
                summary=rollout(checkpoint)
                rollout_metrics={'rollout/macro_success_rate':summary['macro_success_rate']}
                rollout_metrics.update({f'rollout/{t["task"]}/success_rate':t['success_rate'] for t in summary['tasks']})
                tracking.log(rollout_metrics,step=step)
        write_json(output/'coverage.json',dict(train_frames=len(ds),consumed_frames=cursor,unique_observations=len(ds),epoch=1.,optimizer_steps=step,
                   order_sha256=order_hash,validation_samples=metrics['validation/samples'],validation_loss=metrics['validation/loss']))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--resume',type=Path);parser.add_argument('--wandb-mode',choices=['online','offline','disabled'])
    a=parser.parse_args();train(a.manifest,a.output,a.job,a.resume,a.wandb_mode)

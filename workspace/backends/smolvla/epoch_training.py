"""Exactly one permutation of every train observation, including episode tails."""
import json
from pathlib import Path
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,default_collate
from run import HERE,settings,read_json,write_json,digest,load,initial_directory
from input_contract import rgb_tensor


class FrameWindows(Dataset):
    def __init__(self,manifest,split,chunk):
        self.chunk=chunk
        self.config=settings()
        self.episodes=[e for e in manifest['episodes'] if e['split']==split]
        self.ends=np.cumsum([e['frames'] for e in self.episodes])
    def __len__(self):return int(self.ends[-1]) if len(self.ends) else 0
    def locate(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        episode=int(np.searchsorted(self.ends,index,side='right'))
        return self.episodes[episode],int(index-(self.ends[episode-1] if episode else 0))
    def __getitem__(self,index):
        e,t=self.locate(index)
        with h5py.File(e['file'],'r') as f:
            raw=f['action'][t:min(t+self.chunk,e['frames'])]
            action=np.repeat(raw[-1:],self.chunk,axis=0);action[:len(raw)]=raw
            return {'observation.state':torch.from_numpy(f['state'][t]).float(),
                    'observation.images.image':rgb_tensor(f['front'][t],'front',self.config),
                    'observation.images.image2':rgb_tensor(f['wrist'][t],'wrist',self.config),
                    'action':torch.from_numpy(action).float(),
                    'action_is_pad':torch.arange(self.chunk)>=len(raw),'task':str(f.attrs['instruction'])}


def batch_for(dataset,indices,batch_size=4):
    items=[dataset[int(i)] for i in indices]
    if not items:raise ValueError('empty batch')
    while len(items)<batch_size:
        dummy={**items[-1],'action_is_pad':torch.ones_like(items[-1]['action_is_pad'])}
        items.append(dummy)
    return default_collate(items)


def order_for(size,seed):
    return torch.randperm(size,generator=torch.Generator().manual_seed(seed))


def masked_loss(policy,batch):
    # This pinned LeRobot implementation reads the legacy actions_id_pad key.
    batch={**batch,'actions_id_pad':batch['action_is_pad']}
    values,_=policy.forward(batch,reduction='none')
    valid=(~batch['action_is_pad']).sum(dim=1)
    # Policy averages over chunk length; normalize only real timesteps and samples.
    weights=torch.where(valid>0,batch['action_is_pad'].shape[1]/valid.clamp_min(1),0)
    return (values*weights).sum(),int((valid>0).sum())


def train_epoch(manifest_path,output,resume,mode=None):
    import os
    import wandb
    c=settings();m=read_json(manifest_path);output=Path(output);resume=Path(resume)
    if c['batch_size']!=4 or m.get('partial') or not torch.cuda.is_available():raise ValueError('full dataset, batch4 and CUDA required')
    previous=read_json(resume/'state.json')
    if previous['manifest_sha256']!=digest(manifest_path):raise ValueError('resume dataset mismatch')
    for e in m['episodes']:
        if digest(e['file'])!=e['sha256']:raise ValueError('dataset changed')
    train=FrameWindows(m,'train',c['chunk_size']);val=FrameWindows(m,'validation',c['chunk_size'])
    if not len(train) or not len(val):raise ValueError('missing split')
    order=order_for(len(train),c['seed'])
    order_hash=__import__('hashlib').sha256(order.numpy().tobytes()).hexdigest()
    epoch=previous.get('one_pass',{})
    if epoch and (epoch['order_sha256']!=order_hash or epoch['total_frames']!=len(train)):raise ValueError('epoch permutation changed')
    cursor=epoch.get('consumed_frames',0);origin_step=epoch.get('origin_step',previous['step'])
    target=origin_step+(len(train)+3)//4
    if cursor>=len(train):raise ValueError('epoch already completed')
    torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed'])
    policy,pre,post=load(m,resume)
    params=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'])
    restored=torch.load(resume/'training_state.pt',map_location='cpu',weights_only=True)
    optimizer.load_state_dict(restored['optimizer']);torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state_all(restored['cuda_rng'])
    output.mkdir(parents=True,exist_ok=False)
    torch.save(order,output/'epoch_order.pt')
    provenance=dict(config=c,manifest_sha256=digest(manifest_path),initial_weight_sha256=digest(initial_directory()/'model.safetensors'),
                    resume=str(resume),origin_step=origin_step,requested_steps=target,sampling='one shuffled pass without replacement; all episode tail observations included',
                    train_frames=len(train),validation_frames=len(val),order_sha256=order_hash,code_sha256=digest(__file__),
                    entrypoint_sha256=digest(HERE/'run.py'),lock_sha256=digest(HERE/'uv.lock'),smoke=False)
    write_json(output/'run.json',provenance)
    selected=[]
    for key in sorted({e['task'] for e in val.episodes}):
        indices=[];offset=0
        for e in val.episodes:
            if e['task']==key:indices.extend((offset,offset+e['frames']//2,offset+e['frames']-1))
            offset+=e['frames']
        selected.extend(indices[i] for i in np.linspace(0,len(indices)-1,8,dtype=int))
    def validate(full):
        policy.eval();total=0.;count=0;batches=0
        chosen=range(len(val)) if full else selected
        with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(c['seed'])
            for start in range(0,len(chosen),4):
                batch=pre(batch_for(val,chosen[start:start+4]))
                with torch.autocast('cuda',dtype=torch.bfloat16):value,n=masked_loss(policy,batch)
                total+=float(value);count+=n;batches+=1
                if full and batches%1000==0:print(f'full validation {count}/{len(val)}',flush=True)
        return {'validation/loss':total/count,'validation/samples':count,'validation/batches':batches,'validation/full':int(full)}
    mode=mode or os.environ.get('WANDB_MODE',c['wandb_mode'])
    if mode not in ('online','offline','disabled'):raise ValueError('invalid W&B mode')
    with wandb.init(project=os.environ.get('WANDB_PROJECT',c['wandb_project']),entity=os.environ.get('WANDB_ENTITY'),mode=mode,dir=str(output),config=provenance,group=output.parent.name) as tracking:
        while cursor<len(train):
            stop=min(cursor+4,len(train));indices=order[cursor:stop].tolist()
            policy.train();batch=pre(batch_for(train,indices))
            for group in optimizer.param_groups:group['lr']=c['learning_rate']
            with torch.autocast('cuda',dtype=torch.bfloat16):total,n=masked_loss(policy,batch);loss=total/n
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,10.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            cursor=stop;step=origin_step+(cursor+3)//4;full=cursor==len(train)
            metrics={'train/loss':float(loss),'learning_rate':c['learning_rate'],'global_step':step,'epoch':cursor/len(train),
                     'coverage/consumed_frames':cursor,'coverage/total_frames':len(train),'grad_norm':float(grad),'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            if step%c['validation_interval']==0 or full:metrics.update(validate(full))
            tracking.log(metrics,step=step)
            with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step%c['checkpoint_interval']==0 or full:
                path=output/f'ckpt-{step:06d}';policy.save_pretrained(path);pre.save_pretrained(path);post.save_pretrained(path)
                torch.save({'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                            'sampler_rng':torch.Generator().manual_seed(c['seed']).get_state()},path/'training_state.pt')
                write_json(path/'state.json',dict(step=step,manifest_sha256=digest(manifest_path),selected_by_rollout=False,
                           one_pass=dict(order_sha256=order_hash,origin_step=origin_step,consumed_frames=cursor,total_frames=len(train)),
                           validation_full_complete=bool(full and metrics.get('validation/samples')==len(val))))
        if not torch.equal(torch.sort(order).values,torch.arange(len(train))):raise RuntimeError('incomplete coverage')
        write_json(output/'coverage.json',dict(train_frames=len(train),consumed_frames=cursor,unique_observations=len(train),epoch=1,
                   order_sha256=order_hash,validation_frames=len(val),validation_samples=metrics['validation/samples'],
                   validation_loss=metrics['validation/loss'],final_step=step))
    return output

#!/usr/bin/env python3
"""Independent SmolVLA backend. No X-VLA action codec is used."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from physicalai_vla.io import read_json,write_json,digest


def settings():return read_json(Path(os.environ.get('SMOLVLA_CONFIG',str(HERE/'configs/train.json'))))


def initial_directory(c=None):
    return HERE/(c or settings()).get('initial_local_dir','models/base')


def normalization_stats(manifest,c=None):
    import torch
    c=c or settings()
    if c.get('normalization_source')=='initial_checkpoint':
        from safetensors.torch import load_file
        values=load_file(str(initial_directory(c)/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
        result={key:{name:values[f'{key}.{name}'].float() for name in ('mean','std')} for key in ('observation.state','action')}
        for key,dim in [('observation.state',8),('action',7)]:
            if any(v.shape!=(dim,) or not torch.isfinite(v).all() for v in result[key].values()) or not (result[key]['std']>0).all():raise ValueError('invalid checkpoint normalization statistics')
        return result
    return {k:{n:torch.tensor(v,dtype=torch.float32) for n,v in d.items()} for k,d in manifest['stats'].items()}


def download():
    from huggingface_hub import snapshot_download
    c=settings()
    for key,repo,revision in [('base',c['initial_model'],c['initial_revision']),('vlm',c['vlm_model'],c['vlm_revision'])]:
        if not revision:raise ValueError('pin the model revision before downloading')
        target=initial_directory(c) if key=='base' else HERE/'models/vlm'
        snapshot_download(repo,revision=revision,local_dir=target,allow_patterns=['*.json','*.txt','*.model','*.safetensors','*.jinja'])
        files={p.name:digest(p) for p in target.iterdir() if p.is_file()}
        write_json(target/'provenance.json',{'repo':repo,'revision':revision,'files':files})


def prepare(source,output,resume=False):
    """Read original state[t]/action[t]/RGB[t], never reconstructed EE targets."""
    if 'raw_root' in read_json(source):
        from prepare_raw import prepare_raw
        return prepare_raw(source,output,resume)
    import av
    import h5py
    import numpy as np
    c=settings();m=read_json(source);output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    raw_info=read_json(ROOT/'data/raw/meta/info.json')
    rows=[];train_states=[];train_actions=[]
    def video(path):
        with av.open(str(path)) as container:return np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
    for e in m['episodes']:
        with h5py.File(e['file'],'r') as f:
            state=f['source_state'][()];action=f['source_action'][()];instruction=str(f.attrs['instruction'])
        idx=e['episode_index'];values={'episode_chunk':idx//raw_info['chunks_size'],'episode_index':idx}
        front=video(ROOT/'data/raw'/raw_info['video_path'].format(**values,video_key='observation.images.front'))
        wrist=video(ROOT/'data/raw'/raw_info['video_path'].format(**values,video_key='observation.images.wrist'))
        if state.shape!=(len(front),8) or action.shape!=(len(front),7) or wrist.shape!=front.shape:raise ValueError('raw alignment mismatch')
        if not np.isfinite(state).all() or not np.isfinite(action).all():raise ValueError('nonfinite demo')
        target=output/f'episode_{idx:06d}.h5'
        with h5py.File(target,'x') as f:
            for key,value in [('front',front),('wrist',wrist),('state',state),('action',action)]:f.create_dataset(key,data=value,compression='lzf')
            f.attrs['instruction']=instruction
        row={**e,'file':str(target.resolve()),'frames':len(state),'sha256':digest(target)};rows.append(row)
        if e['split']=='train':train_states.append(state);train_actions.append(action)
    if not train_states:raise ValueError('no train episodes')
    stats={}
    for name,values in [('observation.state',train_states),('action',train_actions)]:
        a=np.concatenate(values).astype(np.float64)
        stats[name]={'mean':a.mean(0).tolist(),'std':np.maximum(a.std(0),1e-6).tolist()}
    manifest={'model_family':'smolvla','source_manifest_hash':digest(source),'episodes':rows,'stats':stats,
              'action_representation':'relative_osc','alignment':'state[t],action[t],images[t]',
              'fps':20,'chunk_size':c['chunk_size'],'partial':m.get('partial',True)}
    write_json(output/'manifest.json',manifest,exclusive=True)
    return output/'manifest.json'


def load(manifest,checkpoint=None):
    import torch
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from lerobot.configs.types import FeatureType,PolicyFeature
    c=settings();base=initial_directory(c);vlm=HERE/'models/vlm'
    base_provenance=read_json(base/'provenance.json')
    if base_provenance['repo']!=c['initial_model'] or base_provenance['revision']!=c['initial_revision']:raise ValueError('initial checkpoint identity mismatch')
    for directory in (base,vlm):
        provenance=read_json(directory/'provenance.json')
        for filename,expected in provenance['files'].items():
            if digest(directory/filename)!=expected:raise ValueError(f'model hash mismatch: {filename}')
    from lerobot.configs.policies import PreTrainedConfig
    cfg=PreTrainedConfig.from_pretrained(checkpoint or base)
    cfg.vlm_model_name=str(vlm);cfg.device='cuda';cfg.push_to_hub=False
    cfg.load_vlm_weights=c.get('load_vlm_weights',True);cfg.train_expert_only=True;cfg.freeze_vision_encoder=True;cfg.train_state_proj=True
    cfg.input_features={
        'observation.state':PolicyFeature(type=FeatureType.STATE,shape=(8,)),
        'observation.images.image':PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256)),
        'observation.images.image2':PolicyFeature(type=FeatureType.VISUAL,shape=(3,256,256))}
    cfg.output_features={'action':PolicyFeature(type=FeatureType.ACTION,shape=(7,))}
    cfg.chunk_size=c['chunk_size'];cfg.n_action_steps=c['n_action_steps'];cfg.empty_cameras=0
    policy=SmolVLAPolicy.from_pretrained(str(checkpoint or base),config=cfg,strict=c.get('strict_initial_weights',False)).to('cuda')
    stats=normalization_stats(manifest,c)
    pre,post=make_smolvla_pre_post_processors(cfg,stats)
    return policy,pre,post


def train(manifest_path,output,steps=None,mode=None,smoke=False,resume=None):
    import h5py
    import numpy as np
    import torch
    import wandb
    from torch.utils.data import Dataset,DataLoader,default_collate
    c=settings();m=read_json(manifest_path);steps=steps or c['steps'];output=Path(output)
    start=read_json(Path(resume)/'state.json')['step'] if resume else 0
    if steps<=start:raise ValueError('target steps must exceed resume step')
    if smoke and steps-start>2:raise ValueError('smoke is limited to two additional updates')
    if resume and read_json(Path(resume)/'state.json')['manifest_sha256']!=digest(manifest_path):raise ValueError('resume data mismatch')
    if not smoke and m.get('partial'):raise ValueError('full training requires full dataset manifest')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if c['batch_size']!=4:raise ValueError('agreed batch is 4')
    for row in m['episodes']:
        if digest(row['file'])!=row['sha256']:raise ValueError('dataset changed')
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    policy,pre,post=load(m,resume)
    class Windows(Dataset):
        def __init__(self,split):
            self.items=[(e['file'],i) for e in m['episodes'] if e['split']==split for i in range(max(0,e['frames']-c['chunk_size']+1))]
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            path,t=self.items[i]
            with h5py.File(path,'r') as f:
                return {'observation.state':torch.from_numpy(f['state'][t]).float(),
                        'observation.images.image':torch.from_numpy(f['front'][t].copy()).permute(2,0,1).float()/255,
                        'observation.images.image2':torch.from_numpy(f['wrist'][t].copy()).permute(2,0,1).float()/255,
                        'action':torch.from_numpy(f['action'][t:t+c['chunk_size']]).float(),
                        'action_is_pad':torch.zeros(c['chunk_size'],dtype=torch.bool),'task':str(f.attrs['instruction'])}
    ds,val=Windows('train'),Windows('validation')
    if len(ds)<4 or len(val)<1:raise ValueError('both splits need full action windows')
    validation_limit=4
    if c.get('validation_samples_per_task'):
        from torch.utils.data import Subset
        file_tasks={e['file']:e['task'] for e in m['episodes']}
        groups={}
        for index,(path,t) in enumerate(val.items):groups.setdefault(file_tasks[path],[]).append(index)
        if len(groups)!=8:raise ValueError('validation must cover all eight tasks')
        indices=[group[i] for group in groups.values() for i in np.linspace(0,len(group)-1,c['validation_samples_per_task'],dtype=int)]
        val=Subset(val,indices);validation_limit=(len(indices)+3)//4
    params=[p for p in policy.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'])
    output.mkdir(parents=True,exist_ok=False)
    provenance={'config':c,'manifest_sha256':digest(manifest_path),'lock_sha256':digest(HERE/'uv.lock'),
                'source_sha256':digest(HERE/'source.json'),'code_sha256':digest(__file__),'smoke':smoke,
                'initial_weight_sha256':digest(initial_directory(c)/'model.safetensors'),'normalization_source':c.get('normalization_source','train_split'),'trainable_parameters':sum(p.numel() for p in params),'requested_steps':steps,'resume':str(resume) if resume else None}
    write_json(output/'run.json',provenance)
    generator=torch.Generator().manual_seed(c['seed'])
    if resume:
        state=torch.load(Path(resume)/'training_state.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(state['optimizer']);torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['sampler_rng'])
    mode=mode or os.environ.get('WANDB_MODE',c['wandb_mode'])
    if mode not in ('online','offline','disabled'):raise ValueError('invalid W&B mode')
    with wandb.init(project=os.environ.get('WANDB_PROJECT',c['wandb_project']),entity=os.environ.get('WANDB_ENTITY'),
                    mode=mode,dir=str(output),config=provenance,group=output.name) as run:
        for step in range(start+1,steps+1):
            policy.train();batch=pre(default_collate([ds[i] for i in torch.randint(len(ds),(4,),generator=generator).tolist()]))
            lr=c['learning_rate']*min(1,step/c['warmup_steps'])
            for g in optimizer.param_groups:g['lr']=lr
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=policy.forward(batch)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,10.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
            metrics={'train/loss':loss.item(),'learning_rate':lr,'global_step':step,'epoch':step*4/len(ds),
                     'grad_norm':float(grad),'peak_vram_bytes':torch.cuda.max_memory_allocated()}
            if step%c['validation_interval']==0 or step==steps:
                policy.eval();values=[]
                with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(c['seed'])
                    for i,b in enumerate(DataLoader(val,batch_size=4)):
                        if i==validation_limit:break
                        with torch.autocast('cuda',dtype=torch.bfloat16):v,_=policy.forward(pre(b))
                        values.append(v.item())
                metrics['validation/loss']=float(np.mean(values));metrics['validation/batches']=len(values)
            run.log(metrics,step=step)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics,allow_nan=False)+'\n')
            if step%c['checkpoint_interval']==0 or step==steps:
                path=output/f'ckpt-{step:06d}';policy.save_pretrained(path);pre.save_pretrained(path);post.save_pretrained(path)
                torch.save({'optimizer':optimizer.state_dict(),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':generator.get_state()},path/'training_state.pt')
                write_json(path/'state.json',{'step':step,'manifest_sha256':digest(manifest_path),'selected_by_rollout':False})
                run.log({'checkpoint/step':step},step=step)
    return output


def evaluate(manifest_path,output,checkpoint=None,url='http://127.0.0.1:8011',task_key=None,episodes=10,max_steps=300,resume=False,urls=None):
    import base64,io,time,uuid
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    from urllib.request import Request,urlopen
    m=read_json(manifest_path);policy,pre,post=load(m,checkpoint);policy.eval()
    tasks=read_json(ROOT/'data/tasks.resolved.json')['tasks']
    if task_key:tasks=[t for t in tasks if t['key']==task_key]
    if not tasks or episodes<1 or max_steps<1:raise ValueError('invalid evaluation plan')
    output=Path(output)
    if output.exists() and not resume:raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    protocol={'model_family':'smolvla','initial_revision':settings()['initial_revision'],'checkpoint':str(checkpoint) if checkpoint else None,
              'manifest_sha256':digest(manifest_path),'episodes_per_task':episodes,'max_steps':max_steps,'initial_state_offset':10,
              'action_representation':'relative_osc','action_dt':.05,'execution':'synchronous HTTP simulation worker'}
    if settings().get('preprocessing_profile'):
        protocol.update(preprocessing_profile=settings()['preprocessing_profile'],camera_transforms=settings()['camera_transforms'],state_encoding=settings()['state_encoding'])
    if (output/'protocol.json').exists():
        if read_json(output/'protocol.json')!=protocol:raise ValueError('evaluation resume protocol mismatch')
    else:write_json(output/'protocol.json',protocol,exclusive=True)
    weight=Path(checkpoint)/'model.safetensors' if checkpoint else initial_directory()/'model.safetensors'
    bundle_hash=digest(weight)
    def call(worker_url,route,payload):
        req=Request(worker_url+route,data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=60) as response:return json.load(response)
    def batch(obs):
        images={}
        for name,key in [('front','observation.images.image'),('wrist','observation.images.image2')]:
            a=np.array(Image.open(io.BytesIO(base64.b64decode(obs[name]))).convert('RGB'))
            if settings().get('preprocessing_profile'):
                from input_contract import rgb_tensor
                images[key]=rgb_tensor(a,name,settings(),source='simulator')
            else:
                if name=='front':a=a[::-1,::-1].copy()
                images[key]=torch.from_numpy(a.copy()).permute(2,0,1).float()/255
        if settings().get('state_encoding') and obs.get('state_encoding')!=settings()['state_encoding']:raise ValueError('simulator state convention differs from training contract')
        return pre({**images,'observation.state':torch.tensor(obs['state'],dtype=torch.float32),'task':obs['instruction']})
    from parallel_policy import IsolatedPolicy
    from concurrent.futures import ThreadPoolExecutor
    shared=IsolatedPolicy(policy,batch,post)
    worker_urls=urls or [url]
    if not 1<=len(worker_urls)<=8 or len(set(worker_urls))!=len(worker_urls):raise ValueError('invalid worker URLs')
    trials=[(task,ep) for task in tasks for ep in range(episodes)]
    completed_before=sum((output/f'{task["key"]}_{ep:03d}.json').exists() for task,ep in trials)
    started=time.monotonic()
    def worker(index):
        worker_url=worker_urls[index]
        for task,ep in trials[index::len(worker_urls)]:
            record_path=output/f'{task["key"]}_{ep:03d}.json'
            if record_path.exists():
                existing=read_json(record_path)
                if existing['bundle_hash']!=bundle_hash or existing['protocol_hash']!=digest(output/'protocol.json') or existing['task']!=task['key'] or existing['episode_id']!=ep or not Path(existing['video']).is_file():raise ValueError('inconsistent completed evaluation trial')
                continue
            seed=42+task['dataset_task_index']*100000+ep;context=shared.reset(seed)
            obs=call(worker_url,'/reset',{'task':task['key'],'index':10+ep,'seed':seed})
            video=output/f'{task["key"]}_{ep:03d}.mp4';result={}
            with imageio.get_writer(video,fps=20,codec='libx264') as writer:
                writer.append_data(np.array(Image.open(io.BytesIO(base64.b64decode(obs['front']))))[::-1,::-1])
                for step in range(max_steps):
                    a=shared.select(context,obs)
                    if a.shape!=(7,) or not np.isfinite(a).all():raise ValueError('invalid SmolVLA action')
                    result=call(worker_url,'/step',{'episode':obs['episode'],'action':np.clip(a,-1,1).tolist()});obs=result['observation']
                    writer.append_data(np.array(Image.open(io.BytesIO(base64.b64decode(obs['front']))))[::-1,::-1])
                    if result['success']:break
            record={'trial_id':str(uuid.uuid4()),'task':task['key'],'episode_id':ep,'backend':'simulation',
                    'success':result['success'],'infra_error':False,'intervention':False,'bundle_hash':bundle_hash,
                    'protocol_hash':digest(output/'protocol.json'),'initial_state_hash':result['initial_state_hash'],
                    'seed':seed,'initial_state_index':10+ep,'video':str(video.resolve()),'steps':result['steps'],
                    'collision':result['collision'],'collision_available':True,'collision_free_success':result['success'] and not result['collision']}
            write_json(output/f'{task["key"]}_{ep:03d}.json',record,exclusive=True)
            print(task['key'],ep,record['success'],flush=True)
    with ThreadPoolExecutor(max_workers=len(worker_urls)) as pool:
        futures=[pool.submit(worker,i) for i in range(len(worker_urls))]
        for future in futures:future.result()
    elapsed=time.monotonic()-started
    write_json(output/'execution.json',dict(simulator_workers=len(worker_urls),shared_policy=True,
               independent_trial_rng=True,worker_urls=worker_urls,wall_seconds=elapsed,
               resumed_trials=completed_before,new_trials=len(trials)-completed_before,
               new_trials_per_minute=(len(trials)-completed_before)*60/elapsed))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('download');sub.add_parser('plan')
    a=sub.add_parser('prepare');a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',action='store_true')
    a=sub.add_parser('evaluate');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--worker-urls',nargs='+');a.add_argument('--resume',action='store_true');a.add_argument('--checkpoint',type=Path);a.add_argument('--url',default='http://127.0.0.1:8011');a.add_argument('--task');a.add_argument('--episodes',type=int,default=10);a.add_argument('--max-steps',type=int,default=300)
    a=sub.add_parser('train-epoch');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--resume',type=Path,required=True);a.add_argument('--wandb-mode',choices=['online','offline','disabled'])
    a=sub.add_parser('train');a.add_argument('--manifest',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--resume',type=Path);a.add_argument('--steps',type=int);a.add_argument('--smoke',action='store_true');a.add_argument('--wandb-mode',choices=['online','offline','disabled'])
    args=p.parse_args()
    if args.command=='plan':print(json.dumps(settings(),indent=2))
    elif args.command=='download':download()
    elif args.command=='prepare':print(prepare(args.source,args.output,args.resume))
    elif args.command=='evaluate':print(evaluate(args.manifest,args.output,args.checkpoint,args.url,args.task,args.episodes,args.max_steps,args.resume,args.worker_urls))
    elif args.command=='train-epoch':
        from epoch_training import train_epoch
        print(train_epoch(args.manifest,args.output,args.resume,args.wandb_mode))
    else:print(train(args.manifest,args.output,args.steps,args.wandb_mode,args.smoke,args.resume))
if __name__=='__main__':main()

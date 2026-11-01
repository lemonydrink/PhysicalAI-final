"""Single-GPU PARC-style LoRA SFT, batch 4, validation and W&B."""
from __future__ import annotations
import json
import random
from pathlib import Path
import numpy as np
from .io import ROOT, config, read_json, write_json, digest


def train(output, steps=None, mode=None, smoke=False, resume=None, manifest_path=None):
    import h5py
    import torch
    import wandb
    from PIL import Image
    from peft import LoraConfig,get_peft_model
    from torch.utils.data import Dataset,DataLoader
    from .model import load_model
    import os
    c=config(); steps=c['steps'] if steps is None else steps
    if type(steps) is not int or steps<1:raise ValueError('positive steps required')
    if c['batch_size']!=4:raise ValueError('batch size must be 4')
    codec=read_json(ROOT/'configs/codec.json')
    manifest_path=Path(manifest_path) if manifest_path else ROOT/'data/manifest.json'
    manifest=read_json(manifest_path)
    if manifest['codec_hash']!=digest(ROOT/'configs/codec.json'):raise ValueError('codec/manifest mismatch')
    resume_step=read_json(Path(resume)/'state.json')['global_step'] if resume else 0
    if steps<=resume_step:raise ValueError('target steps must exceed resumed step')
    if smoke and steps-resume_step>2:raise ValueError('smoke limited to two additional updates')
    if not smoke and (codec['validation_status']!='validated' or manifest['partial']):
        raise ValueError('full training requires audited codec and full dataset; --smoke --steps 2 is diagnostic only')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback disabled')
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    for e in manifest['episodes']:
        if digest(e['file'])!=e['sha256']:raise ValueError('dataset hash mismatch')
    model,processor=load_model()
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias='none',target_modules='all-linear',
                 modules_to_save=['transformer.soft_prompt_hub','transformer.action_encoder','transformer.action_decoder']))
    if model.num_actions!=c['chunk_size']:raise ValueError('model horizon mismatch')
    class Episodes(Dataset):
        def __init__(self,split):
            self.items=[]
            for e in manifest['episodes']:
                if e['split']==split:
                    self.items.extend((e['file'],i) for i in range(max(0,e['frames']-c['chunk_size'])))
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            path,start=self.items[i]
            with h5py.File(path,'r') as f:
                front=f['front'][start];wrist=f['wrist'][start]
                target=f['targets'][start:start+c['chunk_size']+1]
                if codec['front_rotate_180']:front=front[::-1,::-1].copy()
                if codec['wrist_rotate_180']:wrist=wrist[::-1,::-1].copy()
                encoded=processor(images=[Image.fromarray(front),Image.fromarray(wrist)],language_instruction=f.attrs['instruction'])
            # Keep upstream action_slice convention: proprio at first target,
            # future targets at following 30 timestamps, with unused arm zeros.
            sequence=np.concatenate((target,np.zeros_like(target)),axis=1)
            out={k:v.squeeze(0) for k,v in encoded.items()}
            out.update(proprio=torch.from_numpy(sequence[0]).float(),action=torch.from_numpy(sequence[1:]).float(),domain_id=torch.tensor(3))
            return out
    dataset,val=Episodes('train'),Episodes('validation')
    if len(dataset)<4:raise ValueError('at least four full train windows required')
    if not smoke and not len(val):raise ValueError('validation split contains no full windows')
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    sample_rng=torch.Generator().manual_seed(c['seed'])
    def batches(start):
        from torch.utils.data import default_collate
        for step in range(start+1,steps+1):
            ids=torch.randint(len(dataset),(4,),generator=sample_rng).tolist()
            yield step,default_collate([dataset[i] for i in ids])
    val_loader=DataLoader(val,batch_size=4,num_workers=0) if len(val) else None
    groups={'vlm':[],'policy':[],'soft_prompt':[],'action':[]}
    for name,param in model.named_parameters():
        if not param.requires_grad:continue
        group='soft_prompt' if 'soft_prompt_hub' in name else 'action' if ('action_encoder' in name or 'action_decoder' in name) else 'vlm' if '.vlm.' in name else 'policy'
        groups[group].append(param)
    optimizer=torch.optim.AdamW([{'params':p,'name':n,'lr':c['learning_rate']} for n,p in groups.items() if p],betas=tuple(c['betas']),weight_decay=c['weight_decay'])
    start=0
    if resume:
        resume=Path(resume)
        state=read_json(resume/'state.json')
        if state['dataset_hash']!=digest(manifest_path) or state['config_hash']!=digest(ROOT/'configs/project.json'):
            raise ValueError('resume provenance mismatch')
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model,load_file(str(resume/'adapter_model.safetensors')))
        payload=torch.load(resume/'optimizer.pt',map_location='cuda',weights_only=True)
        optimizer.load_state_dict(payload['optimizer']);sample_rng.set_state(payload['rng_sampler'].cpu());torch.set_rng_state(payload['rng_cpu'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in payload['rng_cuda']]);start=state['global_step']
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    import subprocess
    git_head=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    git_dirty=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True)
    provenance={'git_commit':git_head.stdout.strip() if git_head.returncode==0 else None,'git_dirty':bool(git_dirty.stdout) if git_dirty.returncode==0 else None,
                'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/physicalai_vla').glob('*.py')},'config':c,'codec':codec,'dataset_hash':digest(manifest_path),'config_hash':digest(ROOT/'configs/project.json'),
                'base_sha256':c['model_sha256'],'smoke':smoke,'requested_steps':steps,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'epoch_definition':'optimizer samples / finite train windows; replacement sampling',
                'source_manifest_sha256':digest(ROOT/'docs/UPSTREAM_SOURCE.json'),'dependency_lock_sha256':digest(ROOT/'uv.lock')}
    write_json(output/'run.json',provenance)
    mode=mode or os.environ.get('WANDB_MODE',c['wandb_mode'])
    if mode not in ('online','offline','disabled'):raise ValueError('invalid W&B mode')
    with wandb.init(project=os.environ.get('WANDB_PROJECT',c['wandb_project']),entity=os.environ.get('WANDB_ENTITY'),
                    mode=mode,group=output.name,config=provenance,dir=str(output)) as run:
        for offset,batch in batches(start):
            if offset>steps:break
            model.train()
            for group in optimizer.param_groups:
                group['lr']=0.0 if offset<=c['freeze_steps'] and group['name'] in ('vlm','policy') else c['learning_rate']
            batch={k:v.cuda() for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):losses=model(**batch);loss=sum(losses.values())
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['max_grad_norm'])
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            logs={'train/loss':loss.item(),'global_step':offset,'epoch':offset*4/len(dataset),'train/grad_norm':float(grad),
                  'gpu/peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            logs.update({f'learning_rate/{g["name"]}':g['lr'] for g in optimizer.param_groups})
            logs.update({f'train/{k}':v.detach().float().item() for k,v in losses.items()})
            if val_loader is not None and (offset%c['validation_interval']==0 or offset==steps):
                model.eval();values=[]
                with torch.no_grad(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(c['seed'])
                    for i,vbatch in enumerate(val_loader):
                        if i==8:break
                        with torch.autocast('cuda',dtype=torch.bfloat16):v=sum(model(**{k:v.cuda() for k,v in vbatch.items()}).values())
                        values.append(v.item())
                logs['validation/loss']=float(np.mean(values))
                logs['validation/batches']=len(values)
            run.log(logs,step=offset)
            with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(logs,allow_nan=False)+'\n')
            if offset%c['checkpoint_interval']==0 or offset==steps:
                ckpt=output/f'ckpt-{offset:06d}';model.save_pretrained(str(ckpt),safe_serialization=True);processor.save_pretrained(str(ckpt))
                torch.save({'optimizer':optimizer.state_dict(),'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_sampler':sample_rng.get_state()},ckpt/'optimizer.pt')
                write_json(ckpt/'state.json',{'global_step':offset,'dataset_hash':provenance['dataset_hash'],'config_hash':provenance['config_hash'],
                          'adapter_sha256':digest(ckpt/'adapter_model.safetensors'),'selected_by_rollout':False})
                run.log({'checkpoint/global_step':offset},step=offset)
    return output

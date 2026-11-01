#!/usr/bin/env python3
"""Capture new ROS2 replays of selected episodes; never overwrite original trials."""
import argparse,base64,io,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py,numpy as np,torch
import imageio.v2 as imageio
from PIL import Image
from run import HERE,ROOT,read_json,write_json,digest,load,settings
from staged_job import environment
from final_ros2 import healthy,rpc
from parallel_policy import IsolatedPolicy
from input_contract import rgb_tensor


def canonical(obs,key):return np.asarray(Image.open(io.BytesIO(base64.b64decode(obs[key]))).convert('RGB'))[::-1,::-1].copy()


def compare_videos(a,b):
 ra=imageio.get_reader(a);rb=imageio.get_reader(b);na=ra.count_frames();nb=rb.count_frames();total=0;max_abs=0;equal=0;count=0
 try:
  for fa,fb in zip(ra,rb):
   diff=np.abs(fa.astype(np.int16)-fb.astype(np.int16));total+=int(diff.sum());max_abs=max(max_abs,int(diff.max()));equal+=int(not diff.any());count+=diff.size
 finally:ra.close();rb.close()
 return dict(original_frames=na,replay_frames=nb,equal_frame_count=equal,all_decoded_frames_identical=na==nb==equal,max_pixel_difference=max_abs,mean_absolute_pixel_difference=total/count if count else None,mp4_bytes_identical=digest(a)==digest(b))


def main(job,out):
 if out.exists():raise FileExistsError('Preserve replay attempts; select a new output directory')
 out.mkdir(parents=True);env=environment(job);os.environ.update(env)
 plan=read_json(job/'plan.json');chapters=read_json(job/'deliverables/final_ros2_eight_tasks_prompted.json')['chapters'];checkpoint=Path(read_json(job/'best_checkpoint.json')['checkpoint']);bundle=digest(checkpoint/'model.safetensors')
 for chapter in chapters:
  if chapter['bundle_hash']!=bundle or digest(chapter['source'])!=chapter['source_sha256']:raise ValueError('original episode/model identity changed')
 policy,pre,post=load(read_json(Path(plan['evaluation_manifest_path'])),checkpoint);policy.eval();config=settings()
 def batch(obs):
  images={name:rgb_tensor(np.asarray(Image.open(io.BytesIO(base64.b64decode(obs[key]))).convert('RGB')),key,config,'simulator') for key,name in [('front','observation.images.image'),('wrist','observation.images.image2')]}
  if obs['state_encoding']!=config['state_encoding']:raise ValueError('state convention mismatch')
  return pre({**images,'observation.state':torch.tensor(obs['state'],dtype=torch.float32),'task':obs['instruction']})
 shared=IsolatedPolicy(policy,batch,post);children=[];logs=[];urls=[]
 ros_env=dict(env,ROS_DOMAIN_ID='64',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');ros_env.pop('VIRTUAL_ENV',None);ros_env.pop('CONDA_PREFIX',None)
 def launch(args,child_env,name):
  log=(out/(name+'.log')).open('w');logs.append(log);p=subprocess.Popen([str(a) for a in args],cwd=ROOT,env=child_env,stdout=log,stderr=subprocess.STDOUT);children.append(p);return p
 try:
  for index in range(2):
   port=18251+index;url=f'http://127.0.0.1:{port}';p=launch([ROOT/'.venv/bin/vla','serve','--kind','sim','--port',port],dict(env,PHYSICALAI_SIM_INSTANCE=str(port),OMP_NUM_THREADS='1'),'sim-'+str(index));healthy(p,url)
   common=['--namespace',f'/physicalai/replay/worker_{index}','--bundle-hash',bundle,'--max-age','2.0']
   launch(['bash',ROOT/'scripts/run_smolvla_ros.sh','--role','robot','--backend',url,*common],ros_env,'robot-'+str(index))
   port=18261+index;url=f'http://127.0.0.1:{port}';p=launch(['bash',ROOT/'scripts/run_smolvla_ros.sh','--role','gateway','--port',port,*common],ros_env,'gateway-'+str(index));healthy(p,url);urls.append(url)
  def worker(index):
   results=[]
   for chapter in chapters[index::2]:
    original=read_json(job/'final-ros2-best80'/f"{chapter['task']}_{chapter['episode_id']:03d}.json");context=shared.reset(original['seed']);url=urls[index]
    obs=rpc(url,'/reset',{'task':chapter['task'],'index':original['initial_state_index'],'seed':original['seed']})
    observations=[obs];actions=[];collisions=[False];successes=[False];result={};video=out/(chapter['task']+'.mp4')
    with imageio.get_writer(video,fps=20,codec='libx264') as writer:
     writer.append_data(canonical(obs,'front'))
     for step in range(300):
      action=np.clip(shared.select(context,obs),-1,1).astype(np.float32);actions.append(action)
      result=rpc(url,'/step',{'episode':obs['episode'],'action':action.tolist(),'observation_time':obs['observation_time'],'bundle_hash':bundle});obs=result['observation']
      observations.append(obs);collisions.append(result['collision']);successes.append(result['success']);writer.append_data(canonical(obs,'front'))
      if result['success']:break
    path=out/(chapter['task']+'.h5')
    with h5py.File(path,'w') as f:
     for key,field in [('state','state'),('joint_position','joint_position'),('joint_velocity','joint_velocity'),('eef_quaternion_xyzw','eef_quaternion_xyzw'),('controller_position','position')]:f[key]=np.asarray([o[field] for o in observations],dtype=np.float64)
     f['action']=np.asarray(actions,dtype=np.float32);f['timestamp']=np.arange(len(observations),dtype=np.float64)/20;f['observation_monotonic_time']=np.asarray([o['observation_time'] for o in observations]);f['success']=np.asarray(successes);f['collision']=np.asarray(collisions)
     for camera in ['front','wrist']:f.create_dataset(camera,data=np.stack([canonical(o,camera) for o in observations]),compression='lzf',chunks=(1,256,256,3))
     f.attrs['instruction']=chapter['prompt'];f.attrs['provenance']='new ROS2 replay; original per-step state/action was not recorded'
    comparison=compare_videos(chapter['source'],video)
    record=dict(task=chapter['task'],prompt=chapter['prompt'],source_trial_id=chapter['trial_id'],source_episode_id=chapter['episode_id'],source_video=chapter['source'],source_video_sha256=chapter['source_sha256'],replay_video=str(video),replay_video_sha256=digest(video),raw_trajectory=str(path),raw_trajectory_sha256=digest(path),seed=original['seed'],initial_state_index=original['initial_state_index'],initial_state_hash_matches=result['initial_state_hash']==original['initial_state_hash'],model_sha256=bundle,steps=result['steps'],original_steps=original['steps'],success=result['success'],collision=result['collision'],ros2_counts=obs['ros2_counts'],comparison=comparison,original_hidden_state_action_identity_not_verifiable=True)
    write_json(out/(chapter['task']+'.json'),record);print(json.dumps({k:record[k] for k in ['task','steps','success','collision','comparison']}),flush=True);results.append(record)
   return results
  import wandb
  with wandb.init(project=os.environ.get('WANDB_PROJECT','physicalai-smolvla-eight'),entity=os.environ.get('WANDB_ENTITY'),mode=os.environ.get('WANDB_MODE','offline'),dir=str(out),job_type='trajectory_replay',config=dict(model_sha256=bundle,source_compilation_sha256=digest(job/'deliverables/final_ros2_eight_tasks_prompted.mp4'),workers=2,original_per_step_data_available=False)) as tracking:
   with ThreadPoolExecutor(max_workers=2) as pool:groups=list(pool.map(worker,range(2)))
   by_task={r['task']:r for group in groups for r in group};results=[by_task[c['task']] for c in chapters]
   summary=dict(provenance='new ROS2 replay, not original recorded telemetry',model_sha256=bundle,episodes=results,all_video_frames_identical=all(r['comparison']['all_decoded_frames_identical'] for r in results),all_initial_state_hashes_match=all(r['initial_state_hash_matches'] for r in results),original_benchmark_results_unchanged=True)
   write_json(out/'replay_manifest.json',summary);tracking.log({'replay/episodes':8,'replay/pixel_exact_episodes':sum(r['comparison']['all_decoded_frames_identical'] for r in results),'replay/successes':sum(r['success'] for r in results)})
 finally:
  for child in reversed(children):
   if child.poll() is None:child.terminate()
  for child in reversed(children):
   try:child.wait(timeout=15)
   except subprocess.TimeoutExpired:child.kill();child.wait()
  for log in logs:log.close()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.job.resolve(),a.output.resolve())

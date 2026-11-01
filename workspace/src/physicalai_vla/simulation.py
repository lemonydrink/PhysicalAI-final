"""Unperturbed LIBERO-plus environments, fixed initial states and videos."""
from __future__ import annotations
import os
import sys
import time
import uuid
import re
from pathlib import Path
import numpy as np
from .io import ROOT,config,read_json,write_json,digest
from .codec import absolute_to_delta


def setup_libero():
    source=ROOT/'third_party/libero'
    os.environ.setdefault('MUJOCO_GL','egl')
    private=ROOT/'runs'/('libero_config_'+os.environ.get('PHYSICALAI_SIM_INSTANCE','default'));private.mkdir(parents=True,exist_ok=True)
    os.environ['LIBERO_CONFIG_PATH']=str(private)
    import yaml
    base=source/'libero/libero'
    paths={'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),'init_states':str(base/'init_files'),
           'datasets':str(ROOT/'data'),'assets':str(base/'assets')}
    (private/'config.yaml').write_text(yaml.safe_dump(paths))
    sys.path.insert(0,str(source))
    return base

class Environment:
    def __init__(self,task,pose_frame="body"):
        base=setup_libero()
        from libero.libero.envs import OffScreenRenderEnv
        bddl=base/'bddl_files'/task['suite']/(task['environment']+'.bddl')
        init=base/'init_files'/task['suite']/(task['environment']+'.pruned_init')
        if digest(bddl)!=task['bddl_sha256'] or digest(init)!=task['initial_states_sha256']:
            raise ValueError('task assets changed since resolution')
        import torch
        self.initial=torch.load(init,map_location='cpu',weights_only=False)
        self.task=task;self.pose_frame=pose_frame
        self.state_encoding=os.environ.get("SMOLVLA_STATE_ENCODING","legacy_scipy")
        if self.state_encoding not in ("legacy_scipy","libero_quat2axisangle"):raise ValueError("unknown state encoding")
        text=bddl.read_text();match=re.search(r'\(:obj_of_interest(.*?)\)\s*\(:init',text,re.S)
        self.interest=set(match.group(1).split()) if match else set()
        self.env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256,control_freq=20)
        self.steps=0

    def reset(self,index,seed):
        if index<0 or index>=len(self.initial):raise ValueError('initial state reuse/out of range forbidden')
        from hashlib import sha256
        state=np.asarray(self.initial[index])
        self.initial_hash=sha256(state.tobytes()).hexdigest()
        self.env.seed(seed);self.env.reset();self.raw=self.env.set_init_state(self.initial[index]);self.steps=0
        for _ in range(10):self.raw,_,_,_=self.env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32))
        self.object_initial={k[:-4]:np.asarray(v).copy() for k,v in self.raw.items() if k.endswith('_pos') and not k.startswith(('robot0','xvla','robo_')) and not k.endswith('_to_robot0_eef_pos') and k[:-4] not in self.interest}
        self.max_object_displacement={name:0.0 for name in self.object_initial}
        return self.observation()

    def observation(self):
        from scipy.spatial.transform import Rotation
        position=self.raw['robot0_eef_pos'];matrix=Rotation.from_quat(self.raw['robot0_eef_quat']).as_matrix()
        if self.pose_frame=='controller':
            controller=self.env.env.robots[0].controller;controller.update(force=True)
            position=np.asarray(controller.ee_pos).copy();matrix=np.asarray(controller.ee_ori_mat).copy()
        rotation=Rotation.from_quat(self.raw['robot0_eef_quat']).as_rotvec()
        if self.state_encoding=='libero_quat2axisangle':
            from robosuite.utils.transform_utils import quat2axisangle
            rotation=quat2axisangle(np.asarray(self.raw['robot0_eef_quat']).copy())
        return {'front':self.raw['agentview_image'],'wrist':self.raw['robot0_eye_in_hand_image'],
                'position':position,'matrix':matrix,
                'instruction':self.task['dataset_instruction'],
                'state_encoding':self.state_encoding,
                'state':np.r_[self.raw['robot0_eef_pos'],rotation,self.raw['robot0_gripper_qpos']]} 

    def step(self,action):
        a=np.asarray(action,dtype=np.float32)
        if a.shape!=(7,) or not np.isfinite(a).all() or np.any(np.abs(a)>1):raise ValueError('invalid simulation action')
        self.raw,_,done,_=self.env.step(a);self.steps+=1
        for name,initial in self.object_initial.items():
            if name+'_pos' in self.raw:self.max_object_displacement[name]=max(self.max_object_displacement[name],float(np.abs(np.asarray(self.raw[name+'_pos'])-initial).sum()))
        return self.observation(),bool(done)

    def close(self):self.env.close()


def evaluate(output,adapter=None,per_task=None,task_key=None,offset=None,pose_frame="body"):
    from .model import Policy
    import imageio.v2 as imageio
    c=config();tasks=read_json(ROOT/'data/tasks.resolved.json')['tasks']
    if task_key:tasks=[t for t in tasks if t['key']==task_key]
    if not tasks:raise ValueError('unknown task')
    per_task=c['episodes_per_task'] if per_task is None else per_task
    offset=c['initial_state_offset'] if offset is None else offset
    if type(per_task) is not int or per_task<1:raise ValueError('positive episode count required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    policy=Policy(adapter=adapter)
    protocol={'backend':'simulation','tasks':tasks,'per_task':per_task,'initial_state_offset':offset,
              'action_dt':c['action_dt'],'seed':c['seed'],'execution_profile':f'synchronous_{pose_frame}_previous_chunk',
              'codec_hash':digest(ROOT/'configs/codec.json'),'inference_profile':read_json(ROOT/'configs/inference.json'),'success_definition':'LIBERO task_completed; collisions reported separately when available',
              'libero_source_hash':digest(ROOT/'docs/LIBERO_SOURCE.json')}
    write_json(output/'protocol.json',protocol);protocol_hash=digest(output/'protocol.json')
    for tindex,task in enumerate(tasks):
        env=None
        try:
            env=Environment(task,pose_frame)
            for episode in range(per_task):
                trial=str(uuid.uuid4());video=output/f'{task["key"]}_{episode:03d}.mp4'
                record={'trial_id':trial,'task':task['key'],'episode_id':episode,'initial_state_index':offset+episode,
                        'backend':'simulation','bundle_hash':policy.bundle_hash,'protocol_hash':protocol_hash,
                        'success':False,'infra_error':False,'intervention':False,'video':str(video.resolve())}
                started=time.monotonic();latencies=[]
                try:
                    seed=c['seed']+task['dataset_task_index']*100000+episode
                    obs=env.reset(offset+episode,seed);record.update(seed=seed,initial_state_hash=env.initial_hash)
                    policy.reset(seed);queue=[];done=False
                    with imageio.get_writer(str(video),fps=20,codec='libx264',quality=7) as writer:
                        writer.append_data(obs['front'][::-1,::-1])
                        for step in range(task['max_steps']):
                            if not queue:
                                tick=time.monotonic();queue=list(policy.predict(obs));latencies.append(time.monotonic()-tick)
                            target=queue.pop(0)
                            action=absolute_to_delta(target,obs['position'],obs['matrix'])
                            obs,done=env.step(action);writer.append_data(obs['front'][::-1,::-1])
                            if done:break
                    record.update(success=done,task_completed=done,steps=env.steps,timeout=not done,collision=any(x>.001 for x in env.max_object_displacement.values()),
                                  collision_free_success=done and not any(x>.001 for x in env.max_object_displacement.values()),
                                  max_non_target_displacement_m=env.max_object_displacement,collision_available=True,latency_p50=float(np.median(latencies)),latency_p95=float(np.quantile(latencies,.95)))
                except Exception as exc:
                    record.update(infra_error=True,error=f'{type(exc).__name__}: {exc}')
                record['wall_time_seconds']=time.monotonic()-started
                write_json(output/f'{task["key"]}_{episode:03d}.json',record,exclusive=True)
                print(f'{task["key"]} {episode}: success={record["success"]} infra_error={record["infra_error"]}',flush=True)
                if record['infra_error']:raise RuntimeError(record['error'])
        finally:
            if env is not None:env.close()
    return output

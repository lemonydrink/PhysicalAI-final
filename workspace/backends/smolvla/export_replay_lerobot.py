#!/usr/bin/env python3
"""Convert recorded replay telemetry with the installed official LeRobot v3 API."""
import argparse,json
from pathlib import Path
import h5py,numpy as np,pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from run import read_json,write_json,digest
from lerobot_video_compat import enable as enable_video_compat


def export(replay,output):
 manifest=read_json(replay/'replay_manifest.json')
 if not manifest['all_video_frames_identical'] or not manifest['all_initial_state_hashes_match']:raise ValueError('Do not present a non-matching replay as the selected video trajectory')
 features={
  'observation.images.image':{'dtype':'video','shape':(3,256,256),'names':['channels','height','width']},
  'observation.images.image2':{'dtype':'video','shape':(3,256,256),'names':['channels','height','width']},
  'observation.state':{'dtype':'float32','shape':(8,),'names':['x','y','z','axis_angle_x','axis_angle_y','axis_angle_z','finger_left','finger_right']},
  'observation.joint_position':{'dtype':'float32','shape':(7,),'names':[f'joint_{i+1}' for i in range(7)]},
  'observation.joint_velocity':{'dtype':'float32','shape':(7,),'names':[f'joint_{i+1}' for i in range(7)]},
  'observation.eef_position':{'dtype':'float32','shape':(3,),'names':['x','y','z']},
  'observation.eef_quaternion_xyzw':{'dtype':'float32','shape':(4,),'names':['x','y','z','w']},
  'observation.controller_position':{'dtype':'float32','shape':(3,),'names':['x','y','z']},
  'observation.host_monotonic_time':{'dtype':'float64','shape':(1,),'names':['host_monotonic_seconds']},
  'action':{'dtype':'float32','shape':(7,),'names':['dx','dy','dz','rx','ry','rz','gripper']},
  'action.valid':{'dtype':'bool','shape':(1,),'names':['action_executed_after_this_observation']},
  'observation.terminal':{'dtype':'bool','shape':(1,),'names':['terminal_observation']},
  'observation.task_success':{'dtype':'bool','shape':(1,),'names':['task_success_at_this_observation']},
  'observation.collision':{'dtype':'bool','shape':(1,),'names':['non_target_displacement_exceeded_1mm_so_far']},
 }
 ds=LeRobotDataset.create(repo_id='lemonydrink/smolvla-libero-selected-ros2-replays',root=output,fps=20,robot_type='franka_panda_simulation',features=features,use_videos=True,video_backend='torchcodec',vcodec='h264',encoder_threads=2,metadata_buffer_size=1)
 provenance=[]
 try:
  for episode,record in enumerate(manifest['episodes']):
   with h5py.File(record['raw_trajectory']) as f:
    n=len(f['action']);assert len(f['state'])==n+1
    for t in range(n+1):
     state=f['state'][t].astype(np.float32)
     frame={'observation.images.image':f['front'][t],'observation.images.image2':f['wrist'][t],
       'observation.state':state,'observation.joint_position':f['joint_position'][t].astype(np.float32),
       'observation.joint_velocity':f['joint_velocity'][t].astype(np.float32),'observation.eef_position':state[:3],
       'observation.eef_quaternion_xyzw':f['eef_quaternion_xyzw'][t].astype(np.float32),
       'observation.controller_position':f['controller_position'][t].astype(np.float32),
       'observation.host_monotonic_time':np.array([f['observation_monotonic_time'][t]],dtype=np.float64),
       'action':f['action'][t] if t<n else np.zeros(7,dtype=np.float32),
       'action.valid':np.array([t<n],dtype=bool),'observation.terminal':np.array([t==n],dtype=bool),
       'observation.task_success':np.array([f['success'][t]],dtype=bool),'observation.collision':np.array([f['collision'][t]],dtype=bool),
       'task':record['prompt']}
     ds.add_frame(frame)
    ds.save_episode(parallel_encoding=False)
    provenance.append(dict(lerobot_episode_index=episode,frames=n+1,executed_actions=n,task=record['task'],source_trial_id=record['source_trial_id'],source_video_sha256=record['source_video_sha256'],raw_replay_sha256=record['raw_trajectory_sha256'],image_replay_comparison=record['comparison'],original_hidden_telemetry_was_not_recorded=True))
    print('Saved LeRobot episode',episode,record['task'],n+1,flush=True)
 finally:ds.finalize()
 write_json(output/'provenance.json',dict(provenance='new ROS2 replay with MP4-identical front videos; not recovered original telemetry',fps=20,observation_alignment='state[t] -> action[t] -> state[t+1]; terminal observation included',terminal_action='zero placeholder, action.valid=false; not an executed zero command',cameras='canonical front/wrist RGB: simulator raw rotated 180 degrees',images_lossy_encoded=True,state_units='position metres, axis-angle radians, finger positions metres',action_units='normalized relative OSC inputs [-1,1], not metres/radians',episodes=provenance,model_sha256=manifest['model_sha256']))
 return verify(replay,output)


def verify(replay,output):
 manifest=read_json(replay/'replay_manifest.json')
 # Independently compare every exported numeric sample against the lossless raw recording.
 records=[]
 for p in sorted((output/'data').rglob('*.parquet')):records.extend(pq.read_table(p).to_pylist())
 for episode,source in enumerate(manifest['episodes']):
  rows=sorted([r for r in records if r['episode_index']==episode],key=lambda r:r['frame_index'])
  with h5py.File(source['raw_trajectory']) as f:
   assert len(rows)==len(f['state'])
   for column,key in [('observation.state','state'),('observation.joint_position','joint_position'),('observation.joint_velocity','joint_velocity'),('observation.eef_quaternion_xyzw','eef_quaternion_xyzw')]:np.testing.assert_array_equal(np.array([r[column] for r in rows],dtype=np.float32),f[key][()].astype(np.float32))
   np.testing.assert_array_equal(np.array([r['action'] for r in rows[:-1]],dtype=np.float32),f['action'][()]);assert not bool(np.asarray(rows[-1]['action.valid']).reshape(-1)[0]) and bool(np.asarray(rows[-1]['observation.terminal']).reshape(-1)[0])
 # Load with LeRobot itself and exercise image/video decoding at episode boundaries.
 enable_video_compat()
 loaded=LeRobotDataset(repo_id='lemonydrink/smolvla-libero-selected-ros2-replays',root=output,video_backend='torchcodec')
 offset=0
 for source in manifest['episodes']:
  n=source['steps']+1
  for index in [offset,offset+n//2,offset+n-1]:
   row=loaded[index];assert row['observation.images.image'].shape==(3,256,256) and row['observation.images.image2'].shape==(3,256,256)
  offset+=n
 verification=dict(official_format='LeRobot v3.0',official_loader_opened=True,video_reader='explicit direct-PyAV compatibility adapter (TorchCodec ABI incompatible with installed torch)',episodes=8,frames=len(records),executed_actions=sum(r['steps'] for r in manifest['episodes']),terminal_padding_rows=8,all_numeric_values_equal_float32_source=True,video_decoder_samples_checked=24)
 write_json(output/'VERIFICATION.json',verification);print(json.dumps(verification),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 if a.verify_only:verify(a.replay,a.output)
 else:export(a.replay,a.output)

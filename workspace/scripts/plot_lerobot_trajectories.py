#!/usr/bin/env python3
"""Static trajectory figures from exported LeRobot parquet, not from guessed video motion."""
import argparse,json,csv
from pathlib import Path
import numpy as np,pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(dataset,out):
 out.mkdir(parents=True,exist_ok=True);provenance=json.loads((dataset/'provenance.json').read_text());rows=[]
 for p in sorted((dataset/'data').rglob('*.parquet')):rows.extend(pq.read_table(p).to_pylist())
 colors=['#1665A6','#C76314','#298449','#9D3C92','#8A7428','#6A64AE','#665752'];summary=[]
 overview=plt.figure(figsize=(15,8),layout='constrained')
 for index,episode in enumerate(provenance['episodes']):
  r=sorted([r for r in rows if r['episode_index']==index],key=lambda r:r['frame_index']);t=np.asarray([v['timestamp'] for v in r]).reshape(-1);pos=np.asarray([v['observation.eef_position'] for v in r]);q=np.asarray([v['observation.joint_position'] for v in r]);vel=np.asarray([v['observation.joint_velocity'] for v in r]);state=np.asarray([v['observation.state'] for v in r]);actions=np.asarray([v['action'] for v in r]);valid=np.asarray([v['action.valid'] for v in r]).reshape(-1).astype(bool);a=actions[valid];ta=t[valid]
  assert len(r)==episode['frames'] and valid.sum()==episode['executed_actions'] and not valid[-1]
  length=float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum());displacement=float(np.linalg.norm(pos[-1]-pos[0]));peak=float(np.abs(vel).max())
  summary.append(dict(episode_index=index,task=episode['task'],observations=len(r),executed_actions=len(a),last_observation_time_s=float(t[-1]),eef_path_length_m=length,eef_displacement_m=displacement,max_abs_joint_velocity_rad_s=peak,max_abs_normalized_action=float(np.abs(a).max()),original_trial_id=episode['source_trial_id'],original_video_frames_identical=episode['image_replay_comparison']['all_decoded_frames_identical']))
  def pathplot(ax):
   ax.plot(pos[:,0],pos[:,1],pos[:,2],color=colors[0],linewidth=1.7);ax.scatter(*pos[0],c='#298449',marker='o',s=35,label='start');ax.scatter(*pos[-1],c='#B53932',marker='*',s=60,label='end')
   ax.set_xlabel('x (m)');ax.set_ylabel('y (m)');ax.set_zlabel('z (m)');ax.tick_params(labelsize=8);ax.view_init(25,-55)
  oa=overview.add_subplot(2,4,index+1,projection='3d');pathplot(oa);oa.set_title(f"{index}: {episode['task']}\npath {length:.3f} m",fontsize=10)
  fig=plt.figure(figsize=(15,12),layout='constrained');gs=fig.add_gridspec(3,2)
  ax=fig.add_subplot(gs[0,0],projection='3d');pathplot(ax);ax.set_title('End-effector path (LIBERO world frame)');ax.legend(fontsize=8)
  ax=fig.add_subplot(gs[0,1])
  for k,key in enumerate('xyz'):ax.plot(t,pos[:,k],label=key,color=colors[k])
  ax.set_title('Measured end-effector position');ax.set_ylabel('Position (m)');ax.legend(ncol=3);ax.grid(alpha=.2)
  ax=fig.add_subplot(gs[1,0])
  for k in range(7):ax.plot(t,q[:,k],label=f'joint {k+1}',color=colors[k],linestyle='--' if k>3 else '-')
  ax.set_title('Measured arm joint positions (7 DOF)');ax.set_ylabel('Angle (rad)');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
  ax=fig.add_subplot(gs[1,1])
  for k in range(7):ax.plot(t,vel[:,k],label=f'joint {k+1}',color=colors[k],linestyle='--' if k>3 else '-')
  ax.set_title('Measured arm joint velocities');ax.set_ylabel('Velocity (rad/s)');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
  ax=fig.add_subplot(gs[2,0])
  for k,key in enumerate(['dx','dy','dz','rx','ry','rz']):ax.step(ta,a[:,k],where='post',label=key,color=colors[k],linestyle='--' if k>2 else '-')
  ax.set_title('Executed relative OSC commands (not metres/radians)');ax.set_ylim(-1.1,1.1);ax.set_ylabel('Normalized command');ax.legend(ncol=6,fontsize=8);ax.grid(alpha=.2)
  ax=fig.add_subplot(gs[2,1]);ax.step(ta,a[:,6],where='post',color=colors[0],label='gripper command');ax.set_ylim(-1.1,1.1);ax.set_ylabel('Normalized gripper command',color=colors[0]);ax.set_title('Gripper command and measured finger positions');ax.grid(alpha=.2)
  right=ax.twinx();right.plot(t,state[:,6]*1000,color=colors[1],label='left finger');right.plot(t,state[:,7]*1000,color=colors[2],linestyle='--',label='right finger');right.set_ylabel('Finger joint position (mm)');ax.legend(loc='upper left',fontsize=8);right.legend(loc='lower right',fontsize=8)
  for axis in fig.axes:
   if axis is not fig.axes[0]:axis.set_xlabel('Simulated time (s), sampled at 20 Hz')
  fig.suptitle(f"Episode {index}: {episode['task']}\nNew ROS 2 replay; front MP4 bytes match original selected clip",fontsize=14)
  fig.savefig(out/(episode['task']+'.png'),dpi=140);plt.close(fig)
 overview.suptitle('Selected-video trajectories exported to LeRobot: measured end-effector paths\nEach subplot has its own LIBERO world origin/scale; reconstructed by a new matching replay',fontsize=14);overview.savefig(out/'overview.png',dpi=160);plt.close(overview)
 (out/'motion_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (out/'motion_summary.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
 print('Generated 8 per-task figures and overview from',len(rows),'LeRobot rows')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.dataset,a.output)

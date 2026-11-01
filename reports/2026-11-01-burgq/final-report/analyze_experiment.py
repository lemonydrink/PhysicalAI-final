#!/usr/bin/env python3
"""Recompute the report from trial JSON; no training or model modification."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,shutil

def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wilson(k,n):
 z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def rows(path):
 vals=[read(p) for p in path.glob('*.json')];vals=[v for v in vals if 'trial_id' in v];keys=[(r['task'],r['episode_id']) for r in vals]
 assert len(keys)==80 and len(set(keys))==80
 assert all(n==10 for n in collections.Counter(r['task'] for r in vals).values())
 assert len({r['bundle_hash'] for r in vals})==len({r['protocol_hash'] for r in vals})==1
 assert all(type(r['success']) is bool and r['collision_available'] for r in vals)
 assert all(r['collision_free_success']==(r['success'] and not r['collision']) for r in vals)
 return {k:r for k,r in zip(keys,vals)}
def main(job,out):
 out.mkdir(parents=True,exist_ok=True);(out/'evidence').mkdir(exist_ok=True);stages=[];allrows={}
 names=['baseline80',*[f'eval-{s:06d}' for s in [2000,4000,6000,8000,10000,11434]],'final-ros2-best80']
 for name in names:
  d=job/name;rs=rows(d);allrows[name]=rs;summary=read(d/'summary.json');success=sum(r['success'] for r in rs.values());safe=sum(r['collision_free_success'] for r in rs.values())
  assert abs(summary['macro_success_rate']-success/80)<1e-10
  assert sum(r['successes'] for r in summary['tasks'])==success
  assert round(sum(t['collision_free_success_rate']*10 for t in summary['tasks']))==safe
  assert not any(r.get('intervention') for r in rs.values())
  stage=dict(stage=name,step=0 if name=='baseline80' else 11434 if name=='final-ros2-best80' else int(name[5:]),successes=success,trials=80,success_rate=success/80,collision_free_successes=safe,collision_free_rate=safe/80,infra_errors=sum(r['infra_error'] for r in rs.values()),wilson95=wilson(success,80))
  stages.append(stage);dest=out/'evidence'/name;dest.mkdir(exist_ok=True)
  for p in d.glob('*.json'):shutil.copy2(p,dest/p.name)
 a=allrows['baseline80'];b=allrows['eval-011434'];z=allrows['final-ros2-best80'];paired={}
 for metric in ['success','collision_free_success']:
  tab={str(x)+str(y):sum(r[metric]==bool(x) and b[k][metric]==bool(y) for k,r in a.items()) for x in [0,1] for y in [0,1]};n=tab['01']+tab['10'];p=min(1.,2*sum(math.comb(n,i) for i in range(min(tab['01'],tab['10'])+1))/2**n) if n else 1
  paired[metric]=dict(transitions=tab,exact_mcnemar_p_exploratory=p)
 assert all(r['initial_state_hash']==b[k]['initial_state_hash'] and r['seed']==b[k]['seed'] and r['initial_state_index']==b[k]['initial_state_index'] for k,r in a.items())
 assert all(r['transport']=='ros2_dds' and r['ros2_counts']['actions']==r['steps'] for r in z.values())
 assert {r['initial_state_index'] for r in z.values()}==set(range(20,30))
 assert {r['initial_state_index'] for r in b.values()}==set(range(10,20))
 tasks=read(job.parents[3]/'data/tasks.resolved.json')['tasks'];taskrows=[]
 for t in tasks:
  row=dict(task=t['key'],prompt=t['dataset_instruction'])
  for key,rs in [('baseline',a),('model_sft',b),('ros2_final',z)]:
   rr=[r for r in rs.values() if r['task']==t['key']];row[key+'_successes']=sum(r['success'] for r in rr);row[key+'_collision_free_successes']=sum(r['collision_free_success'] for r in rr)
  row['model_delta_pp']=(row['model_sft_successes']-row['baseline_successes'])*10;taskrows.append(row)
 with (out/'checkpoint_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(stages[0]));w.writeheader();w.writerows(stages)
 with (out/'task_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(taskrows[0]));w.writeheader();w.writerows(taskrows)
 pa=read(job/'baseline80/protocol.json');pb=read(job/'eval-011434/protocol.json');pa.pop('checkpoint');pb.pop('checkpoint');assert pa==pb
 metrics=[json.loads(line) for name in ['train-001','train-002'] for line in (job/name/'metrics.jsonl').read_text().splitlines()];assert [r['global_step'] for r in metrics]==list(range(1,11435))
 coverage=read(job/'train-002/coverage.json');assert coverage['consumed_frames']==coverage['unique_observations']==365887 and coverage['optimizer_steps']==11434
 import torch
 order=torch.load(job/'train-002/epoch_order.pt',map_location='cpu',weights_only=True);assert torch.equal(torch.sort(order).values,torch.arange(365887))
 assert hashlib.sha256(order.numpy().tobytes()).hexdigest()==coverage['order_sha256']
 assert read(job/'train-001/run.json')['order_sha256']==read(job/'train-002/run.json')['order_sha256']
 videos=read(job/'deliverables/final_ros2_eight_tasks_prompted.json');assert len(videos['chapters'])==8 and videos['collision_free_task_count']==8 and videos['all_selected_episodes_successful']
 assert digest(job/'deliverables/final_ros2_eight_tasks_prompted.mp4')==videos['output_sha256']
 for c in videos['chapters']:
  rr=z[c['task'],c['episode_id']];assert rr['success'] and rr['collision_free_success'] and rr['trial_id']==c['trial_id'] and rr['bundle_hash']==c['bundle_hash'] and digest(c['source'])==c['source_sha256']
 metadata=dict(stages=stages,tasks=taskrows,paired_comparison=paired,all_80_initial_state_hashes_match=True,all_80_ros2_dds_action_counts_match=True,ros2_action_count=sum(r['steps'] for r in z.values()),coverage=coverage,video=videos,limitations=['Checkpoint selected using the same 80 HTTP trials; post-selection comparison.','ROS2 final uses different initial states and lacks a same-state HTTP comparison; no causal ROS transport improvement claim.','LIBERO pretraining overlap and simulator-only scope.'])
 write(out/'analysis.json',metadata)
 for name in ['best_checkpoint.json','final_ros2_result.json','training-complete.json','status.json','video-status.json','plan.json']:
  if (job/name).exists():shutil.copy2(job/name,out/'evidence'/name)
 for name in ['run.json','coverage.json']:shutil.copy2(job/'train-002'/name,out/'evidence'/('train-'+name))
 shutil.copy2(job/'deliverables/final_ros2_eight_tasks_prompted.json',out/'evidence/video_manifest.json')
 # Standalone static scientific figure for the Markdown report.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 model=stages[:-1];fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained')
 ax.plot([s['step'] for s in model],[s['success_rate']*100 for s in model],marker='o',color='#1559A7',label='Task success (HTTP selection set)')
 ax.plot([s['step'] for s in model],[s['collision_free_rate']*100 for s in model],marker='s',linestyle='--',color='#BB5900',label='Collision-free success (same set)')
 for s in model:ax.annotate(f"{s['successes']}/80",(s['step'],s['success_rate']*100),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
 ax.axhline(70,color='#777777',linestyle=':',label='Target: 70% task success');ax.set_ylim(0,108);ax.set_xlim(-500,12000);ax.set_xticks([0,2000,4000,6000,8000,10000,11434]);ax.set_xlabel('Optimizer step');ax.set_ylabel('Success rate (%)');ax.set_title('Model evaluation: identical initial states 10-19, 80 trials per checkpoint');ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right');fig.savefig(out/'model_performance.png',dpi=180);plt.close(fig)
 print(json.dumps({'paired':paired,'stages':[(v['stage'],v['successes'],v['collision_free_successes']) for v in stages],'ros_actions':metadata['ros2_action_count']},ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.job,a.out)

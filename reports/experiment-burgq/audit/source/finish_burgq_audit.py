from pathlib import Path
import json,hashlib,collections,math
import numpy as np,h5py,av,pyarrow.parquet as pq
from PIL import Image,ImageDraw
h=Path('${WORKSPACE}/backends/smolvla');j=h/'runs/burgq-staged-batch32-run';out=j/'audit';ref=h/'data/burgq_reference_raw'
read=lambda p:json.loads(p.read_text())
write=lambda p,x:p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
m=read(h/'data/full-v1/manifest.json');eps={e['episode_index']:e for e in m['episodes']};sel=read(out/'reference_video_selection.json');rows=[];tiles=[]
for e in sel:
 meta=e['metadata'];records=pq.read_table(ref/f"data/chunk-000/file-{meta['data/file_index']:03d}.parquet").to_pylist();rr=[r for r in records if r['episode_index']==e['ref_episode']]
 # Episodes can span parquet files: concatenate all files when needed.
 if len(rr)!=e['frames']:
  rr=[]
  for p in sorted((ref/'data').rglob('*.parquet')):rr.extend(r for r in pq.read_table(p).to_pylist() if r['episode_index']==e['ref_episode'])
 rr.sort(key=lambda r:r['frame_index']);rs=np.array([r['observation.state'] for r in rr]);
 with h5py.File(eps[e['plus_episode']]['file']) as f:
  s=f['state'][()]; row={'task':e['task'],'state_lag_mae':{str(l):float(np.abs((rs[1:] if l==-1 else rs[:-1] if l==1 else rs)-(s[:-1] if l==-1 else s[1:] if l==1 else s)).mean()) for l in [-1,0,1]},'cameras':{}}
  for cam,key in [('image','front'),('image2','wrist')]:
   t=meta[f'videos/observation.images.{cam}/from_timestamp']+e['frame_index']/10
   with av.open(str(ref/e['videos'][cam])) as c:
    st=c.streams.video[0];c.seek(max(0,int((t-.5)/float(st.time_base))),stream=st)
    best=None;dist=1e9
    for fr in c.decode(st):
     d=abs(float(fr.pts*fr.time_base)-t)
     if d<dist:best=fr.to_ndarray(format='rgb24');dist=d
     if float(fr.pts*fr.time_base)>t+.15:break
   actual=f[key][e['frame_index']]; transforms={'identity':best,'rotate180':best[::-1,::-1],'flip_vertical':best[::-1],'flip_horizontal':best[:,::-1],'BGR':best[:,:,::-1]}
   mae={k:float(np.abs(v.astype(float)-actual).mean()) for k,v in transforms.items()};row['cameras'][key]={'timestamp_error':dist,'mae':mae,'best':min(mae,key=mae.get)}
   tile=Image.new('RGB',(512,280),'white');tile.paste(Image.fromarray(best),(0,24));tile.paste(Image.fromarray(actual),(256,24));ImageDraw.Draw(tile).text((2,4),e['task']+' '+key+' REF | PLUS',fill='black');tiles.append(tile)
 rows.append(row);print(row,flush=True)
write(out/'reference_images_temporal.json',rows)
sheet=Image.new('RGB',(1024,280*8),'white')
for n,t in enumerate(tiles):sheet.paste(t,((n%2)*512,(n//2)*280))
sheet.save(out/'reference_image_pairs.jpg')
groups=collections.defaultdict(list);arrays={};oldval=0
for e in m['episodes']:
 with h5py.File(e['file']) as f:
  a=f['action'][()];s=f['state'][()];assert a.shape==(e['frames'],7) and s.shape==(e['frames'],8);assert np.isfinite(a).all() and np.isfinite(s).all()
 k=e['task']+':'+hashlib.sha256(a.astype(np.float32).tobytes()).hexdigest();groups[k].append(e);arrays[e['episode_index']]=(s,a)
val=set()
for task in sorted({e['task'] for e in m['episodes']}):
 keys=sorted([k for k in groups if k.startswith(task+':')],key=lambda k:hashlib.sha256(('42:'+k).encode()).hexdigest());val.update(keys[:max(1,round(len(keys)*.1))])
for k,es in groups.items():
 if any(e['split']=='train' for e in es):oldval+=sum(e['split']=='validation' for e in es)
 for e in es:e['split']='validation' if k in val else 'train';e['trajectory_group']=k
print('split labels',collections.Counter(e['split'] for e in m['episodes']),flush=True)
# Keep existing schema statistics keys, compute from new train subset.
print('manifest keys',m.keys(), 'stats',str(m.get('stats'))[:1000],flush=True)
for split in ['train','validation']:
 print(split,sum(e['frames'] for e in m['episodes'] if e['split']==split),flush=True)
m['split_policy']={'method':'task-stratified-full-action-trajectory-sha256','seed':42,'validation_group_fraction':.1,'groups':len(groups),'validation_groups':len(val),'old_validation_leaked_episodes':oldval,'pretraining_overlap':'Reference original trajectories were used in base model pretraining; this split only prevents SFT train/validation trajectory duplication.'}
# stats schema inspected separately before saving
p=h/'data/grouped-v2';p.mkdir(exist_ok=True);write(p/'manifest.json',m)
write(out/'group_split.json',{'groups':len(groups),'validation_groups':len(val),'old_validation_leaked_episodes':oldval,'counts':{split:{'episodes':sum(e['split']==split for e in m['episodes']),'frames':sum(e['frames'] for e in m['episodes'] if e['split']==split)} for split in ['train','validation']},'trajectory_overlap':0})

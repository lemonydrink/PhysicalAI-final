from pathlib import Path
import numpy as np,json,pyarrow.parquet as pq,h5py,hashlib,collections
h=Path('${WORKSPACE}/backends/smolvla');root=h.parents[1];p=h/'data/burgq_reference_raw';out=h/'runs/burgq-staged-batch32-run/audit'
tasks=json.loads((root/'data/tasks.resolved.json').read_text())['tasks'];wanted={t['dataset_instruction']:t['key'] for t in tasks}
ref_labels={x['task_index']:x['__index_level_0__'] for x in pq.read_table(p/'meta/tasks.parquet').to_pylist()};ref={};records=[]
for f in sorted((p/'data').rglob('*.parquet')):records.extend(pq.read_table(f).to_pylist())
print('reference_rows',len(records),flush=True)
for r in records:
 if ref_labels[r['task_index']] in wanted:ref.setdefault(r['episode_index'],[]).append(r)
groups={};hashes={}
for eid,rows in ref.items():
 rows.sort(key=lambda r:r['frame_index']);s=np.asarray([r['observation.state'] for r in rows],dtype=np.float32);a=np.asarray([r['action'] for r in rows],dtype=np.float32)
 key=wanted[ref_labels[rows[0]['task_index']]];groups.setdefault(key,{'ref':[],'plus':[]})['ref'].append(s)
 hashes.setdefault((key,len(a),hashlib.sha256(a.tobytes()).hexdigest()),[]).append((eid,s,a))
m=json.loads((h/'data/full-v1/manifest.json').read_text());matches=[]
for e in m['episodes']:
 with h5py.File(e['file']) as f:s=f['state'][()];a=f['action'][()]
 groups[e['task']]['plus'].append(s)
 k=(e['task'],len(a),hashlib.sha256(a.tobytes()).hexdigest())
 for rid,rs,ra in hashes.get(k,[]):
  diff=rs-s
  matches.append(dict(task=e['task'],plus_episode=e['episode_index'],ref_episode=rid,frames=len(a),delta_mean=diff.mean(0).tolist(),delta_std=diff.std(0).tolist(),max_abs=float(np.abs(diff).max())))
summary={}
for key,g in groups.items():
 row={}
 for source in ['ref','plus']:
  x=np.concatenate(g[source]);row[source]={'episodes':len(g[source]),'frames':len(x),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'min':x.min(0).tolist(),'max':x.max(0).tolist()}
 summary[key]=row
(out/'reference_numeric_comparison.json').write_text(json.dumps(dict(reference_rows=len(records),tasks=summary,exact_action_matches=matches),indent=2)+'\n')
print('matches',len(matches),flush=True)
for k,v in summary.items():print(k,'ref',v['ref']['mean'][:3],'plus',v['plus']['mean'][:3],flush=True)
for match in matches[:10]:print(match,flush=True)

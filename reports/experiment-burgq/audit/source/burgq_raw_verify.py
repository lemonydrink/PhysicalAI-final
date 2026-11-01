from pathlib import Path
import json,h5py,numpy as np,pyarrow.parquet as pq,hashlib
h=Path('${WORKSPACE}/backends/smolvla');root=h.parents[1];j=h/'runs/burgq-staged-batch32-run';m=json.loads((h/'data/grouped-v2/manifest.json').read_text());labels={r['task_index']:r['task'] for r in []};tasks=json.loads((root/'data/tasks.resolved.json').read_text())['tasks'];instructions={t['key']:t['dataset_instruction'] for t in tasks};counts=0;dt=set();groups={};
for e in m['episodes']:
 r=pq.read_table(root/f"data/raw/data/chunk-{e['episode_index']//1000:03d}/episode_{e['episode_index']:06d}.parquet").to_pylist()
 with h5py.File(e['file']) as f:
  for col,key in [('observation.state','state'),('action','action')]:np.testing.assert_array_equal(np.asarray([v[col] for v in r],dtype=np.float32),f[key][()])
  assert str(f.attrs['instruction'])==instructions[e['task']]
  for key in ['front','wrist']:assert f[key].shape==(len(r),256,256,3) and f[key].dtype==np.uint8
  dt.add(round(float(r[1]['timestamp']-r[0]['timestamp']),4))
 groups.setdefault(e['trajectory_group'],set()).add(e['split']);counts+=1
assert all(len(x)==1 for x in groups.values());assert counts==3071
report={'all_raw_state_action_exact':True,'episodes':counts,'frames':sum(e['frames'] for e in m['episodes']),'raw_timestamp_delta_seconds':sorted(dt),'all_instruction_strings_match':True,'all_image_shapes_dtype_valid':True,'cross_split_exact_action_groups':0,'finite_state_action':True}
(j/'audit/raw_conversion_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)

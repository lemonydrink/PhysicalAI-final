import importlib.util
import json
from pathlib import Path
import h5py
import numpy as np

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]


def test_raw_conversion_has_disjoint_split_train_only_stats_and_resume(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    from prepare_raw import prepare_raw
    m=json.loads((ROOT/'data/eight-tasks-local.manifest.json').read_text())
    selected=[];counts={key:0 for key in m['counts']}
    for ep in m['episodes']:
        if counts[ep['task']]<2:
            selected.append(ep);counts[ep['task']]+=1
    m.update(episodes=selected,counts=counts)
    source=tmp_path/'source.json';source.write_text(json.dumps(m))
    result=prepare_raw(source,tmp_path/'converted')
    prepared=json.loads(result.read_text())
    train=[e for e in prepared['episodes'] if e['split']=='train']
    val=[e for e in prepared['episodes'] if e['split']=='validation']
    assert len(train)==len(val)==8
    assert {e['episode_index'] for e in train}.isdisjoint(e['episode_index'] for e in val)
    states=[]
    for row in train:
        with h5py.File(row['file']) as f:
            states.append(f['state'][()])
            assert len(f['front'])==len(f['wrist'])==len(f['state'])==len(f['action'])
            assert f['front'].chunks==(1,256,256,3)
    np.testing.assert_allclose(prepared['stats']['observation.state']['mean'],np.concatenate(states).astype(np.float64).mean(0))
    before=result.read_bytes()
    assert prepare_raw(source,tmp_path/'converted',resume=True)==result
    assert result.read_bytes()==before

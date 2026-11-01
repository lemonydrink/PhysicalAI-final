from pathlib import Path
import pytest
from physicalai_vla.evaluation import summarize

def records():
    return [{'task':str(t),'episode_id':i,'success':i<7,'infra_error':False,'intervention':False,
             'backend':'simulation','bundle_hash':'bundle','protocol_hash':'protocol'} for t in range(8) for i in range(10)]

def test_seventy_percent_requires_all_trials_and_eight_video_files(tmp_path):
    rows=records();keys=list(map(str,range(8)))
    assert not summarize(rows,keys)['accepted']
    for t in range(8):
        p=tmp_path/f'{t}.mp4';p.write_bytes(b'fixture')
        rows[t*10]['video']=str(p)
    r=summarize(rows,keys)
    assert r['macro_success_rate']==pytest.approx(.7) and r['accepted']
    assert not summarize(rows[:-1],keys)['accepted']

def test_duplicate_mixed_models_and_intervened_success_refused():
    keys=list(map(str,range(8)));r=records()
    with pytest.raises(ValueError):summarize(r+[r[0]],keys)
    r[0]['bundle_hash']='other'
    with pytest.raises(ValueError):summarize(r,keys)
    r=records();r[0]['intervention']=True
    with pytest.raises(ValueError):summarize(r,keys)

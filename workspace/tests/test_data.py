import json
from pathlib import Path
import pytest
from physicalai_vla.data import split_manifest
from physicalai_vla.io import read_json,write_json


def test_stratified_split_keeps_each_episode_whole_and_both_splits(tmp_path):
    episodes=[{'task':str(t),'episode_index':t*10+i,'split':'train','file':'fixture','sha256':'fixture'} for t in range(8) for i in range(2)]
    source=tmp_path/'source.json';out=tmp_path/'split.json'
    write_json(source,{'episodes':episodes,'partial':True})
    m=split_manifest(source,out)
    for task in map(str,range(8)):
        assert {e['split'] for e in m['episodes'] if e['task']==task}=={'train','validation'}
    assert read_json(source)['episodes'][0]['split']=='train'
    with pytest.raises(FileExistsError):split_manifest(source,out)

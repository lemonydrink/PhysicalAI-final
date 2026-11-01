import importlib.util
from pathlib import Path
import numpy as np
import h5py

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('smolvla_backend',HERE/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_model_is_generic_and_batch_four_with_relative_actions():
    c=m.settings()
    assert c['initial_model']=='lerobot/smolvla_base'
    assert len(c['initial_revision'])==40
    assert c['batch_size']==4 and c['action_dim']==7 and c['state_dim']==8
    assert c['state_action_alignment']=='same_state_action'

def test_prepared_smoke_uses_raw_same_timestamp_state_action():
    manifest=m.read_json(HERE/'data/smoke/manifest.json')
    source=m.read_json(m.ROOT/'data/smoke-split/manifest.json')
    source_by_id={e['episode_index']:e for e in source['episodes']}
    for row in manifest['episodes']:
        with h5py.File(row['file'],'r') as target,h5py.File(source_by_id[row['episode_index']]['file'],'r') as original:
            np.testing.assert_array_equal(target['state'][0],original['source_state'][0])
            np.testing.assert_array_equal(target['action'][0],original['source_action'][0])
            assert target['action'].shape[1]==7 and len(target['front'])==len(target['state'])


def test_libero_initializer_and_checkpoint_normalization(monkeypatch):
    from safetensors.torch import load_file
    monkeypatch.setenv('SMOLVLA_CONFIG',str(HERE/'configs/libero-plus-sft.json'))
    c=m.settings()
    assert c['initial_model']=='lerobot/smolvla_libero_plus'
    assert m.initial_directory(c)==HERE/'models/libero-plus-7bb70aa5'
    provenance=m.read_json(m.initial_directory(c)/'provenance.json')
    assert provenance['revision']==c['initial_revision']
    stats=m.normalization_stats({'stats':{}},c)
    source=load_file(str(m.initial_directory(c)/'policy_preprocessor_step_5_normalizer_processor.safetensors'))
    for key,dim in [('observation.state',8),('action',7)]:
        for name in ('mean','std'):
            np.testing.assert_array_equal(stats[key][name].numpy(),source[f'{key}.{name}'].numpy())
            assert tuple(stats[key][name].shape)==(dim,)
    assert c['batch_size']==4 and c['validation_samples_per_task']==8

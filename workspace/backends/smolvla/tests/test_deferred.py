from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from deferred_experiment import predecessor_ready
from run import HERE,initial_directory,normalization_stats


def test_deferred_experiment_waits_for_completion_and_process_exit():
    assert not predecessor_ready({'stage':'training'},'active')
    assert not predecessor_ready({'stage':'complete'},'active')
    assert not predecessor_ready({'stage':'complete'},'deactivating')
    assert not predecessor_ready({'stage':'failed'},'failed')
    assert not predecessor_ready({'stage':'training'},'inactive')
    assert predecessor_ready({'stage':'complete'},'inactive')
    assert not predecessor_ready({'stage':'stopped_by_user'},'inactive')
    assert predecessor_ready({'stage':'stopped_by_user'},'inactive',allow_user_stop=True)
    assert not predecessor_ready({'stage':'stopped_by_user'},'active',allow_user_stop=True)


def test_burgq_uses_its_own_fixed_weights_and_normalization():
    c=json.loads((HERE/'configs/burgq-staged-batch32.json').read_text())
    assert c['initial_model']=='BurgQ/smolvla-libero-bs32-100k'
    assert c['batch_size']==32 and c['epochs']==1 and c['load_vlm_weights'] is False
    directory=initial_directory(c);p=json.loads((directory/'provenance.json').read_text())
    assert p['revision']==c['initial_revision']
    assert p['files']['model.safetensors']=='88638a6aa1399327d79418265b224b883e7af6b5a7855c3b0aa6b7929ff1f971'
    assert p['compatibility_transform']['removed_config_fields']=={'pretrained_revision':None}
    stats=normalization_stats({},c)
    assert stats['observation.state']['mean'].shape==(8,)
    assert stats['action']['mean'].shape==(7,)

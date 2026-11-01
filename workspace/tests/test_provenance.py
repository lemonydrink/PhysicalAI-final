from physicalai_vla.io import ROOT,read_json,digest

def test_upstream_models_are_byte_identical_to_recorded_copy():
    manifest=read_json(ROOT/'docs/UPSTREAM_SOURCE.json')
    for row in manifest['files']:
        assert digest(ROOT/row['target'])==row['sha256']

def test_batch_and_confirmed_task_selection():
    assert read_json(ROOT/'configs/project.json')['batch_size']==4
    tasks=read_json(ROOT/'configs/tasks.json')
    assert tasks['selection_confirmed']
    assert {t['dataset_task_index'] for t in tasks['tasks']}=={4,6,7,8,9,16,20,36}

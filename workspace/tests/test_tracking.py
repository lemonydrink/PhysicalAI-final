import pytest
import wandb

@pytest.mark.parametrize('mode',['offline','disabled'])
def test_tracking_modes_log_training_and_validation(tmp_path,mode):
    with wandb.init(project='physicalai-local-test',mode=mode,dir=str(tmp_path),config={'batch_size':4}) as run:
        run.log({'train/loss':.5,'validation/loss':.6,'learning_rate':1e-4,'epoch':.1,'global_step':1},step=1)
        assert run is not None

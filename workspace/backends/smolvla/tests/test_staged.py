from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from input_contract import rgb_tensor
from staged_training import scheduled_lr,reduce_valid_seven


def test_camera_canonicalization_matches_dataset_and_both_simulator_cameras():
    canonical=np.zeros((256,256,3),dtype=np.uint8);canonical[2,9]=[13,80,211]
    raw_sim=canonical[::-1,::-1].copy()
    c={'camera_transforms':{'front':'rot180','wrist':'rot180'},'dataset_camera_transforms':{'front':'identity','wrist':'identity'}}
    for camera in ('front','wrist'):
        assert torch.equal(rgb_tensor(canonical,camera,c,'dataset'),rgb_tensor(raw_sim,camera,c,'simulator'))


def test_only_seven_real_dimensions_and_unpadded_steps_contribute_gradient():
    loss=torch.ones((4,50,32),requires_grad=True)
    with torch.no_grad():loss[...,7:]=1e6
    pad=torch.ones((4,50),dtype=torch.bool);pad[0,:2]=False
    total,n=reduce_valid_seven(loss,pad)
    assert n==1 and torch.isclose(total,torch.tensor(1.))
    total.backward()
    assert torch.count_nonzero(loss.grad[...,7:])==0
    assert torch.count_nonzero(loss.grad[1:])==0
    assert torch.count_nonzero(loss.grad[0,2:])==0
    assert torch.isclose(loss.grad.sum(),torch.tensor(1.))


def test_schedule_one_epoch_and_accumulation_weighting():
    total=(364505+31)//32;warmup=570
    assert total==11391
    assert scheduled_lr(1,total,warmup)<scheduled_lr(warmup,total,warmup)
    assert abs(scheduled_lr(warmup,total,warmup)-3e-5)<1e-12
    assert abs(scheduled_lr(total,total,warmup)-3e-6)<1e-12
    p=torch.tensor(1.,requires_grad=True);target=torch.arange(9.)
    for i in range(0,9,4):
        losses=(p-target[i:i+4])[:,None,None].square().expand(-1,50,32)
        value,_=reduce_valid_seven(losses,torch.zeros((len(losses),50),dtype=torch.bool))
        (value/9).backward()
    grad=p.grad.clone();p.grad=None
    ((p-target).square().mean()).backward()
    torch.testing.assert_close(grad,p.grad)


def test_batch32_configuration_preserves_effective_batch_and_recipe():
    import json
    h=Path(__file__).resolve().parents[1]
    old=json.loads((h/'configs/staged-one-epoch.json').read_text())
    new=json.loads((h/'configs/staged-batch32.json').read_text())
    assert new['batch_size']==32 and new['gradient_accumulation_steps']==1
    assert old['batch_size']*old['gradient_accumulation_steps']==new['batch_size']*new['gradient_accumulation_steps']==32
    changed={key for key in old if old[key]!=new[key]}
    assert changed=={'batch_size','gradient_accumulation_steps'}

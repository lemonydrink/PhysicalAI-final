import sys
from pathlib import Path
import h5py
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from epoch_training import FrameWindows,batch_for,order_for,masked_loss


def test_every_frame_including_episode_tail_and_dummy_batch_mask(tmp_path):
    episodes=[]
    for episode,length in enumerate((5,2)):
        path=tmp_path/f'{episode}.h5'
        with h5py.File(path,'w') as f:
            f['state']=np.zeros((length,8),dtype=np.float32)
            f['action']=np.full((length,7),episode+1,dtype=np.float32)
            f['front']=np.zeros((length,256,256,3),dtype=np.uint8)
            f['wrist']=np.zeros((length,256,256,3),dtype=np.uint8)
            f.attrs['instruction']='task'
        episodes.append({'file':str(path),'frames':length,'split':'train'})
    ds=FrameWindows({'episodes':episodes},'train',50)
    assert len(ds)==7
    assert int((~ds[4]['action_is_pad']).sum())==1
    assert int((~ds[5]['action_is_pad']).sum())==2
    assert torch.all(ds[4]['action']==1) and torch.all(ds[5]['action']==2)
    batch=batch_for(ds,[6])
    assert batch['action'].shape==(4,50,7)
    assert (~batch['action_is_pad']).sum(dim=1).tolist()==[1,0,0,0]
    class Policy:
        def forward(self,batch,reduction):
            # Constant unit error on each valid action, as in policy's time mean.
            assert torch.equal(batch['actions_id_pad'],batch['action_is_pad'])
            return (~batch['actions_id_pad']).float().mean(dim=1),{}
    loss,count=masked_loss(Policy(),batch)
    assert count==1 and torch.isclose(loss,torch.tensor(1.))


def test_full_dataset_permutation_and_resume_cover_every_observation_once():
    order=order_for(364505,42)
    assert torch.equal(torch.sort(order).values,torch.arange(364505))
    assert torch.equal(order,order_for(364505,42))
    cursor=20000
    assert len(set(order[:cursor].tolist()) & set(order[cursor:].tolist()))==0
    assert sum(len(order[i:i+4]) for i in range(0,len(order),4))==364505
    assert (len(order)+3)//4==91127

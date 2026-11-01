"""Shared dataset/simulator image conversion; preserves LIBERO raw-camera convention."""
import numpy as np
import torch


def rgb_tensor(rgb,camera,config,source='dataset'):
    a=np.asarray(rgb)
    if a.shape!=(256,256,3) or a.dtype!=np.uint8:raise ValueError('requires 256x256 RGB uint8')
    if source not in ('dataset','simulator'):raise ValueError('unknown RGB source')
    field='dataset_camera_transforms' if source=='dataset' else 'camera_transforms'
    transform=config.get(field,{}).get(camera,'identity')
    if transform=='rot180':a=a[::-1,::-1]
    elif transform!='identity':raise ValueError('unknown image transform')
    return torch.from_numpy(a.copy()).permute(2,0,1).float()/255

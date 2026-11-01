"""2toINF model loader shared by training and inference; no remote code."""
from __future__ import annotations
import numpy as np
from .io import ROOT, config, read_json, digest
from .codec import matrix_to_rotation6d, finite


def load_model(base=None, adapter=None, device='cuda'):
    import torch
    from .upstream.models.modeling_xvla import XVLA
    from .upstream.models.processing_xvla import XVLAProcessor
    base = ROOT/'models/base' if base is None else base
    if digest(base/'model.safetensors') != config()['model_sha256']:
        raise ValueError('base model checksum mismatch')
    processor = XVLAProcessor.from_pretrained(str(base),local_files_only=True)
    model = XVLA.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model,str(adapter),local_files_only=True,is_trainable=False)
    return model.to(device),processor

class Policy:
    def __init__(self, adapter=None, device='cuda'):
        self.model,self.processor = load_model(adapter=adapter,device=device)
        self.model.eval(); self.device=device
        self.codec=read_json(ROOT/'configs/codec.json')
        self.bundle_hash = digest(ROOT/'models/base/model.safetensors')
        if adapter:
            import hashlib
            self.bundle_hash=hashlib.sha256((self.bundle_hash+digest(adapter/'adapter_model.safetensors')).encode()).hexdigest()
        self.reset()

    def reset(self, seed=42):
        import torch
        torch.manual_seed(seed)
        self.previous=None

    def predict(self, observation):
        import torch
        from PIL import Image
        position=finite(observation['position'],(3,)); matrix=np.asarray(observation['matrix'])
        rotation=matrix_to_rotation6d(matrix)
        front=np.asarray(observation['front'],dtype=np.uint8)
        wrist=np.asarray(observation['wrist'],dtype=np.uint8)
        if front.shape!=(256,256,3) or wrist.shape!=front.shape:
            raise ValueError('expected two 256x256 RGB images')
        if self.codec['front_rotate_180']:front=front[::-1,::-1].copy()
        if self.codec['wrist_rotate_180']:wrist=wrist[::-1,::-1].copy()
        # PARC baseline uses last predicted chunk pose; asynchronous ROS path
        # uses actual observed pose and is evaluated as a separate execution profile.
        proprio=np.r_[position,rotation,0,np.zeros(10)].astype(np.float32)
        if self.previous is not None and observation.get('proprio_mode','previous_chunk')=='previous_chunk':
            proprio[:9]=self.previous
        inputs=self.processor(images=[Image.fromarray(front),Image.fromarray(wrist)],language_instruction=observation['instruction'])
        inputs['proprio']=torch.from_numpy(proprio).unsqueeze(0)
        inputs['domain_id']=torch.tensor([3],dtype=torch.long)
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode():a=self.model.generate_actions(**inputs,steps=10).squeeze(0).float().cpu().numpy()
        if a.shape!=(30,20) or not np.isfinite(a).all():raise ValueError('invalid model chunk')
        self.previous=a[-1,:9].copy()
        return a[:,:10]

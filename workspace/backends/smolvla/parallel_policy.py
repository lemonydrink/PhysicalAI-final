"""Serialize shared inference while preserving each rollout's RNG and action queue."""
import threading
import torch


class IsolatedPolicy:
    def __init__(self,policy,preprocess,postprocess,devices=None):
        self.policy=policy;self.preprocess=preprocess;self.postprocess=postprocess
        self.devices=([torch.cuda.current_device()] if torch.cuda.is_available() else []) if devices is None else devices
        self.lock=threading.Lock()

    def reset(self,seed):
        with self.lock,torch.random.fork_rng(devices=self.devices):
            torch.manual_seed(seed);self.policy.reset()
            return {'queues':self.policy._queues,'cpu_rng':torch.get_rng_state(),
                    'cuda_rng':[torch.cuda.get_rng_state(d) for d in self.devices]}

    def select(self,context,observation):
        with self.lock,torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(context['cpu_rng'])
            for d,rng in zip(self.devices,context['cuda_rng']):torch.cuda.set_rng_state(rng,d)
            self.policy._queues=context['queues']
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=bool(self.devices)):
                action=self.postprocess(self.policy.select_action(self.preprocess(observation)))
            context.update(queues=self.policy._queues,cpu_rng=torch.get_rng_state(),
                           cuda_rng=[torch.cuda.get_rng_state(d) for d in self.devices])
            return action.squeeze(0).float().cpu().numpy()

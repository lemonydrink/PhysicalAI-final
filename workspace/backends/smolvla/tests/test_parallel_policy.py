from collections import deque
import threading
import numpy as np
import torch
from parallel_policy import IsolatedPolicy


class QueuedPolicy:
    def reset(self):self._queues={'action':deque()}
    def select_action(self,batch):
        if not self._queues['action']:
            self._queues['action'].extend(torch.rand(1,7)+batch for _ in range(3))
        return self._queues['action'].popleft()


def test_parallel_trials_preserve_serial_rng_and_action_queue():
    shared=IsolatedPolicy(QueuedPolicy(),lambda x:x,lambda x:x,devices=[])
    def sequence(seed):
        c=shared.reset(seed)
        return np.stack([shared.select(c,seed) for _ in range(9)])
    expected={seed:sequence(seed) for seed in range(4)}
    contexts={seed:shared.reset(seed) for seed in range(4)};actual={seed:[] for seed in range(4)}
    barrier=threading.Barrier(4)
    def worker(seed):
        for _ in range(9):
            barrier.wait();actual[seed].append(shared.select(contexts[seed],seed))
    threads=[threading.Thread(target=worker,args=(seed,)) for seed in range(4)]
    for thread in threads:thread.start()
    for thread in threads:thread.join()
    for seed in range(4):np.testing.assert_array_equal(actual[seed],expected[seed])


def test_rollouts_do_not_modify_callers_training_rng():
    shared=IsolatedPolicy(QueuedPolicy(),lambda x:x,lambda x:x,devices=[])
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    c=shared.reset(91)
    for _ in range(5):shared.select(c,0)
    assert torch.equal(before,torch.get_rng_state())

import numpy as np
from physicalai_vla.scheduler import Chunk,Scheduler

def chunk(episode='e',sequence=0,time=10):
    return Chunk(episode,sequence,time,.05,np.tile([0,0,0,1,0,0,0,1,0,1],(4,1)),'codec')

def test_old_episode_duplicate_and_timeout_do_not_extend_motion():
    s=Scheduler('codec');s.reset('e')
    assert not s.accept(chunk(episode='old'),10)[0]
    assert s.accept(chunk(),10)[0]
    assert s.next(10) is not None
    assert s.next(10.01) is None
    assert not s.accept(chunk(),10.02)[0]
    assert s.next(10.3) is None and s.state=='PAUSED'
    assert s.next(10.4) is None

def test_reset_clears_chunk_and_future_or_wrong_codec_rejected():
    s=Scheduler('codec');s.reset('e');assert not s.accept(chunk(time=11),10)[0]
    assert s.accept(chunk(),10)[0]
    s.reset('new');assert s.next(10.05) is None
    c=chunk(episode='new');c.codec='other'
    assert not s.accept(c,10)[0] and s.state=='PAUSED'

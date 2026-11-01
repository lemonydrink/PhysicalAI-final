"""Simulation-only absolute waypoint scheduler; monotonic deadlines."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .codec import rotation6d_to_matrix

@dataclass
class Chunk:
    episode: str
    sequence: int
    observation_time: float
    dt: float
    actions: np.ndarray
    codec: str

class Scheduler:
    def __init__(self, codec, max_age=0.25):
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError('max_age must be positive')
        self.codec, self.max_age = codec, max_age
        self.reset('')

    def reset(self, episode):
        self.episode, self.last_sequence = episode, -1
        self.chunk, self.index = None, 0
        self.state, self.reason = 'READY', 'reset'

    def pause(self, reason):
        self.chunk = None
        self.state, self.reason = 'PAUSED', reason

    def accept(self, chunk, now):
        if chunk.episode != self.episode:
            return False, 'old_episode'
        if type(chunk.sequence) is not int or chunk.sequence <= self.last_sequence:
            return False, 'out_of_order'
        a = np.asarray(chunk.actions)
        if chunk.codec != self.codec:
            self.pause('codec_mismatch')
            return False, self.reason
        if (a.ndim != 2 or a.shape[1] != 10 or not 1 <= len(a) <= 128
                or not np.isfinite(a).all() or not math.isfinite(chunk.dt) or chunk.dt <= 0
                or not math.isfinite(now) or not math.isfinite(chunk.observation_time)):
            self.pause('invalid_chunk')
            return False, self.reason
        age = now - chunk.observation_time
        if age < 0 or age > self.max_age:
            return False, 'expired_or_future'
        try:
            for step in a:
                rotation6d_to_matrix(step[3:9])
        except ValueError:
            self.pause('invalid_rotation')
            return False, self.reason
        # Conservative initial path: no prefix skipping, no temporal fusion.
        # Start accepted absolute sequence at present time, under age deadline.
        self.chunk, self.index, self.start = chunk, 0, now
        self.last_sequence = chunk.sequence
        self.state, self.reason = 'RUNNING', 'accepted'
        return True, self.reason

    def next(self, now):
        if self.chunk is None:
            return None
        if now < self.start or now - self.chunk.observation_time > self.max_age:
            self.pause('observation_expired')
            return None
        if self.index >= len(self.chunk.actions):
            self.pause('buffer_underrun')
            return None
        if now + 1e-9 < self.start + self.index * self.chunk.dt:
            return None
        result = np.asarray(self.chunk.actions[self.index]).copy()
        self.index += 1
        return result

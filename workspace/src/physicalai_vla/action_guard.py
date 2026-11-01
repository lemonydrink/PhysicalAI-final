"""Fail-closed contract for normalized, controller-frame relative OSC commands.

This guard does not turn normalized actions into hardware-safe motor commands.
A hardware driver must supply measured state, physical limits, calibration and stop.
"""
import math
import time
from dataclasses import dataclass
from typing import Protocol

CODEC='smolvla-relative-osc7-v1'


class RobotDriver(Protocol):
    def reset(self,payload:dict)->dict: ...
    def step(self,payload:dict)->dict: ...
    def stop(self,reason:str)->None: ...
    def health(self)->dict: ...


@dataclass
class ActionGuard:
    bundle_hash:str
    max_age:float=2.0
    episode:str=''
    sequence:int=-1
    observation_time:float=0.0
    fault:str=''

    def reset(self,episode,sequence,observation_time):
        if not episode or not math.isfinite(observation_time):raise ValueError('invalid reset observation')
        self.episode=episode;self.sequence=sequence;self.observation_time=observation_time;self.fault=''

    def accept(self,*,episode,sequence,observation_time,codec,bundle_hash,dt,actions,now=None):
        now=time.monotonic() if now is None else now
        try:
            if self.fault:raise ValueError('fault latched; explicit reset required')
            if episode!=self.episode:raise ValueError('episode mismatch')
            if sequence!=self.sequence+1:raise ValueError('out of order or duplicate command')
            if codec!=CODEC or bundle_hash!=self.bundle_hash:raise ValueError('codec/model mismatch')
            if not math.isfinite(dt) or abs(dt-.05)>1e-8:raise ValueError('invalid action period')
            if len(actions)!=7 or not all(math.isfinite(x) and -1.000001<=x<=1.000001 for x in actions):raise ValueError('invalid normalized action')
            if not math.isfinite(observation_time) or not math.isfinite(now):raise ValueError('invalid timestamp')
            if abs(observation_time-self.observation_time)>1e-6 or not 0<=now-observation_time<=self.max_age:raise ValueError('stale/future observation')
        except (ValueError,TypeError) as exc:
            self.fault=str(exc);raise ValueError(self.fault) from exc
        self.sequence=sequence
        return list(actions)

    def watchdog(self,now=None):
        now=time.monotonic() if now is None else now
        if self.episode and now-self.observation_time>self.max_age:
            self.fault='observation watchdog expired';return True
        return False


def require_hardware_readiness(capabilities):
    """Reject real motion without verified adapter-specific commissioning evidence."""
    required=('measured_state','camera_calibration','robot_camera_extrinsics','action_frame_calibration',
              'physical_workspace_limits','velocity_acceleration_limits','collision_monitor',
              'gripper_calibration','independent_watchdog','emergency_stop','controller_acknowledgement')
    missing=[name for name in required if capabilities.get(name) is not True]
    if capabilities.get('mode')!='hardware' or missing:raise ValueError('hardware commissioning incomplete: '+', '.join(missing))

"""Measured Cartesian setpoint adapter for a separately commissioned robot driver.

No vendor driver or physical calibration is fabricated here. The controller must
implement read_state(), command(target), stop(reason) and an independent watchdog.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .action_guard import require_hardware_readiness


@dataclass
class CartesianLimits:
    workspace_min:tuple
    workspace_max:tuple
    max_linear_speed:float
    max_angular_speed:float
    gripper_min_width:float
    gripper_max_width:float
    max_gripper_speed:float
    position_gain:float
    rotation_gain:float
    policy_to_base_rotation:object
    feedback_max_age:float=.1


class MeasuredCartesianAdapter:
    def __init__(self,controller,capabilities,limits):
        require_hardware_readiness(capabilities)
        self.controller=controller;self.limits=limits;self.fault=None;self.last_sequence=-1
        self.rotation=np.asarray(limits.policy_to_base_rotation,dtype=float)
        if self.rotation.shape!=(3,3) or not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(self.rotation),1.):raise ValueError('invalid frame rotation')
        lo=np.asarray(limits.workspace_min);hi=np.asarray(limits.workspace_max)
        positive=(limits.max_linear_speed,limits.max_angular_speed,limits.max_gripper_speed,limits.position_gain,limits.rotation_gain,limits.feedback_max_age)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([*lo,*hi,*positive,limits.gripper_min_width,limits.gripper_max_width]).all() or not (lo<hi).all() or min(positive)<=0 or not 0<=limits.gripper_min_width<limits.gripper_max_width:raise ValueError('invalid physical limits')

    def stop(self,reason):
        self.fault=reason;self.controller.stop(reason)

    def execute(self,action,sequence,dt=.05,now=None):
        now=time.monotonic() if now is None else now
        try:
            if self.fault:raise ValueError('fault latched; recommission/reset required')
            if type(sequence) is not int or sequence!=self.last_sequence+1:raise ValueError('command order')
            a=np.asarray(action,dtype=float);l=self.limits
            if a.shape!=(7,) or not np.isfinite(a).all() or (np.abs(a)>1).any() or not np.isfinite(dt) or abs(dt-.05)>1e-8:raise ValueError('invalid OSC7 command')
            state=self.controller.read_state()
            if state.get('collision') or not 0<=now-state['monotonic_time']<=l.feedback_max_age:raise ValueError('collision/stale feedback')
            position=np.asarray(state['position_base'],dtype=float);quat=np.asarray(state['quaternion_xyzw'],dtype=float);width=float(state['gripper_width'])
            if position.shape!=(3,) or quat.shape!=(4,) or not np.isfinite([*position,*quat,width]).all() or not np.isclose(np.linalg.norm(quat),1.,atol=1e-5):raise ValueError('invalid measured pose')
            if not l.gripper_min_width<=width<=l.gripper_max_width:raise ValueError('invalid measured gripper width')
            if ((position<l.workspace_min)|(position>l.workspace_max)).any():raise ValueError('measured pose outside workspace')
            dp=self.rotation@a[:3]*l.position_gain;dr=self.rotation@a[3:6]*l.rotation_gain
            dp*=min(1.,l.max_linear_speed*dt/max(np.linalg.norm(dp),1e-12));dr*=min(1.,l.max_angular_speed*dt/max(np.linalg.norm(dr),1e-12))
            target=position+dp
            if ((target<l.workspace_min)|(target>l.workspace_max)).any():raise ValueError('target outside workspace')
            orientation=(Rotation.from_rotvec(dr)*Rotation.from_quat(quat)).as_quat()
            goal=l.gripper_min_width+(1-a[6])*.5*(l.gripper_max_width-l.gripper_min_width)
            goal=float(np.clip(width+np.clip(goal-width,-l.max_gripper_speed*dt,l.max_gripper_speed*dt),l.gripper_min_width,l.gripper_max_width))
            command=dict(sequence=sequence,position_base=target.tolist(),quaternion_xyzw=orientation.tolist(),gripper_width=goal,dt=dt,deadline=now+dt)
            ack=self.controller.command(command)
            if not ack.get('accepted') or ack.get('sequence')!=sequence:raise ValueError('controller did not acknowledge setpoint')
            self.last_sequence=sequence;return command
        except Exception as exc:
            self.stop(str(exc));raise

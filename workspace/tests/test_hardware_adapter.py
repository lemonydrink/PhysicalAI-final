import numpy as np
import pytest
from physicalai_vla.hardware_adapter import CartesianLimits,MeasuredCartesianAdapter

CAP={k:True for k in ('measured_state','camera_calibration','robot_camera_extrinsics','action_frame_calibration','physical_workspace_limits','velocity_acceleration_limits','collision_monitor','gripper_calibration','independent_watchdog','emergency_stop','controller_acknowledgement')};CAP['mode']='hardware'

class Controller:
    def __init__(self):self.stopped=False;self.state=dict(position_base=[.4,0.,.3],quaternion_xyzw=[0.,0.,0.,1.],gripper_width=.04,monotonic_time=100.,collision=False)
    def read_state(self):return self.state
    def command(self,target):return dict(accepted=True,sequence=target['sequence'])
    def stop(self,reason):self.stopped=True


def adapter():
 c=Controller();limits=CartesianLimits((.2,-.2,.1),(.6,.2,.5),.05,.25,0.,.08,.02,.05,.5,np.eye(3));return c,MeasuredCartesianAdapter(c,CAP,limits)


def test_physical_velocity_and_gripper_limits_use_measured_pose():
 c,a=adapter();target=a.execute([1.,1.,1.,1.,1.,1.,1.],0,now=100.01)
 assert np.linalg.norm(np.asarray(target['position_base'])-c.state['position_base'])<=.05*.05+1e-10
 assert abs(target['gripper_width']-.04)<=.02*.05+1e-10
 assert not c.stopped


@pytest.mark.parametrize('update',[{'collision':True},{'monotonic_time':90.},{'position_base':[2.,0.,.3]}])
def test_unsafe_measurements_stop_without_replaying_commands(update):
 c,a=adapter();c.state.update(update)
 with pytest.raises(ValueError):a.execute([0.]*7,0,now=100.01)
 assert c.stopped and a.fault

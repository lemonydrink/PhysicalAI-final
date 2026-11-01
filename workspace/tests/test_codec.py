import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from physicalai_vla.codec import matrix_to_rotation6d,rotation6d_to_matrix,absolute_to_delta,reconstruct

def test_noncommuting_rotation_and_osc_roundtrip():
    current=Rotation.from_euler('xyz',[.3,-.5,.8]).as_matrix()
    increment=Rotation.from_rotvec([.02,-.07,.04]).as_matrix()
    goal=increment@current
    np.testing.assert_allclose(rotation6d_to_matrix(matrix_to_rotation6d(goal)),goal,atol=1e-12)
    target=np.r_[[.02,-.01,.03],matrix_to_rotation6d(goal),1]
    action=absolute_to_delta(target,np.zeros(3),current)
    np.testing.assert_allclose(action[:3],[.4,-.2,.6],atol=1e-6)
    np.testing.assert_allclose(action[3:6],[.04,-.14,.08],atol=1e-6)

@pytest.mark.parametrize('value',[[0]*6,[1,0,0,2,0,0],[float('nan'),0,0,0,1,0]])
def test_invalid_rotation_refused(value):
    with pytest.raises(ValueError):rotation6d_to_matrix(value)

def test_temporal_alignments_differ_explicitly():
    state=np.zeros((3,8));actions=np.zeros((3,7));actions[:,0]=[0,.5,1]
    old,idx_old=reconstruct(state,actions,'previous_state_next_action')
    same,idx_same=reconstruct(state,actions,'same_state_action')
    np.testing.assert_allclose(old[:,0],[.025,.05]);np.testing.assert_allclose(same[:,0],[0,.025])
    assert idx_old.tolist()==[1,2] and idx_same.tolist()==[0,1]

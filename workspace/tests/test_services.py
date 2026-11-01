import numpy as np
import pytest
from physicalai_vla.services import encode_observation,decode_observation


def test_rgb_images_and_poses_survive_cross_interpreter_payload():
    image=np.zeros((256,256,3),dtype=np.uint8);image[10,20]=[255,17,34]
    original={'front':image,'wrist':image.copy(),'position':np.array([.1,.2,.3]),'matrix':np.eye(3),'instruction':'fixture'}
    encoded=encode_observation(original,'episode',7);decoded=decode_observation(encoded)
    assert encoded['episode']=='episode' and encoded['sequence']==7
    np.testing.assert_array_equal(decoded['front'],image)
    assert decoded['proprio_mode']=='measured'
    encoded['front']='not base64'
    with pytest.raises(ValueError):decode_observation(encoded)

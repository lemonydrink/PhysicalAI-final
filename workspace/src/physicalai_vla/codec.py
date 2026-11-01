"""Validated absolute EE6D <-> normalized LIBERO OSC boundary."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'expected finite {shape}, got {a.shape}')
    return a


def rotation6d_to_matrix(value):
    v = finite(value, (6,))
    a, b = v[:3].copy(), v[3:].copy()
    if np.linalg.norm(a) < 1e-8:
        raise ValueError('degenerate rotation first column')
    a /= np.linalg.norm(a)
    b -= np.dot(a, b) * a
    if np.linalg.norm(b) < 1e-8:
        raise ValueError('collinear rotation columns')
    b /= np.linalg.norm(b)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def matrix_to_rotation6d(value):
    m = finite(value, (3, 3))
    if not np.allclose(m.T @ m, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(m), 1, atol=1e-6):
        raise ValueError('not an SO(3) rotation')
    return np.concatenate((m[:, 0], m[:, 1]))


def absolute_to_delta(target, position, matrix, position_scale=0.05, rotation_scale=0.5):
    if not np.isfinite([position_scale, rotation_scale]).all() or min(position_scale, rotation_scale) <= 0:
        raise ValueError('scales must be finite and positive')
    t, p = finite(target, (10,)), finite(position, (3,))
    matrix_to_rotation6d(matrix)
    relative = rotation6d_to_matrix(t[3:9]) @ np.asarray(matrix).T
    delta = np.r_[(t[:3] - p) / position_scale, Rotation.from_matrix(relative).as_rotvec() / rotation_scale]
    # PARC-compatible SIMULATION saturation; never a real robot safety filter.
    return np.r_[np.clip(delta, -1, 1), 1.0 if t[9] > 0.5 else -1.0].astype(np.float32)


def reconstruct(state, actions, alignment):
    state, actions = np.asarray(state), np.asarray(actions)
    if state.ndim != 2 or state.shape[1] != 8 or actions.shape != (len(state), 7) or len(state) < 2:
        raise ValueError('expected state[T,8], action[T,7], T>=2')
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError('nonfinite demonstration')
    if alignment == 'previous_state_next_action':
        pose, rel, image_indices = state[:-1, :6], actions[1:], np.arange(1, len(state))
    elif alignment == 'same_state_action':
        pose, rel, image_indices = state[:-1, :6], actions[:-1], np.arange(len(state) - 1)
    else:
        raise ValueError('unknown temporal alignment')
    goals = []
    for p, a in zip(pose, rel):
        a6 = np.clip(a[:6], -1, 1)
        m = Rotation.from_rotvec(a6[3:] * 0.5).as_matrix() @ Rotation.from_rotvec(p[3:6]).as_matrix()
        goals.append(np.r_[p[:3] + a6[:3] * 0.05, matrix_to_rotation6d(m), float(a[6] > 0)])
    return np.asarray(goals, dtype=np.float32), image_indices

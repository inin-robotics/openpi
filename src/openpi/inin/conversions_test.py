import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.inin import conversions


def _quat(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, dtype=np.float64) * angle).as_quat()


def test_quat_rotvec_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        quat = rng.normal(size=4)
        quat /= np.linalg.norm(quat)
        back = conversions.rotvec_to_quat(conversions.quat_to_rotvec(quat))
        # Quaternions double-cover rotations; compare up to sign.
        assert np.allclose(back, quat, atol=1e-9) or np.allclose(back, -quat, atol=1e-9)


def test_state14_to_state7_layout():
    quat = _quat([0.0, 0.0, 1.0], 0.5)
    state = np.concatenate(
        [
            np.arange(6, dtype=np.float64),  # joints (dropped)
            [1.0, 2.0, 3.0],  # tcp xyz
            quat,
            [0.7],  # gripper
        ]
    )
    state7 = conversions.state14_to_state7(state)
    assert state7.shape == (7,)
    assert state7.dtype == np.float32
    np.testing.assert_allclose(state7[:3], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(state7[3:6], [0.0, 0.0, 0.5], atol=1e-6)
    assert state7[6] == pytest.approx(0.7)


def test_nearest_equivalent_rotvec_wraps_across_pi():
    axis = np.array([0.0, 0.0, 1.0])
    anchor = axis * 3.1  # just below pi
    # A rotation slightly past pi comes back from as_rotvec as ~-(2*pi - 3.2)
    # about the flipped axis; the nearest equivalent should be ~3.2 * axis.
    raw = conversions.quat_to_rotvec(_quat(axis, 3.2))
    wrapped = conversions.nearest_equivalent_rotvec(raw, anchor)
    np.testing.assert_allclose(wrapped, axis * 3.2, atol=1e-9)


def test_nearest_equivalent_rotvec_identity_near_two_pi_anchor():
    anchor = np.array([0.0, 0.0, 6.0])  # > pi, close to a full turn
    wrapped = conversions.nearest_equivalent_rotvec(np.zeros(3), anchor)
    np.testing.assert_allclose(wrapped, [0.0, 0.0, 2.0 * np.pi], atol=1e-9)


def test_quat_action_chunk_is_continuous_across_pi_boundary():
    axis = np.array([0.0, 0.0, 1.0])
    angles = np.linspace(2.9, 3.5, 8)  # crosses pi
    chunk = np.stack([np.concatenate([[0.1, 0.2, 0.3], _quat(axis, a), [0.5]]) for a in angles])
    anchor = axis * 2.9
    rotvec_chunk = conversions.quat_action_chunk_to_rotvec_chunk(chunk, anchor)
    assert rotvec_chunk.shape == (8, 7)
    np.testing.assert_allclose(rotvec_chunk[:, 3:6], np.outer(angles, axis), atol=1e-6)
    # No jump larger than the true angular step anywhere in the chunk.
    steps = np.linalg.norm(np.diff(rotvec_chunk[:, 3:6], axis=0), axis=1)
    assert np.all(steps < 0.2)


def test_rotvec_chunk_round_trip_to_quat():
    axis = np.array([1.0, 0.0, 0.0])
    angles = np.linspace(0.1, 1.0, 5)
    quat_chunk = np.stack([np.concatenate([[1.0, 2.0, 3.0], _quat(axis, a), [0.9]]) for a in angles])
    rotvec_chunk = conversions.quat_action_chunk_to_rotvec_chunk(quat_chunk, np.zeros(3))
    back = conversions.rotvec_chunk_to_quat_chunk(rotvec_chunk)
    assert back.shape == quat_chunk.shape
    np.testing.assert_allclose(back[:, :3], quat_chunk[:, :3], atol=1e-6)
    np.testing.assert_allclose(back[:, 7], quat_chunk[:, 7], atol=1e-6)
    for i in range(len(angles)):
        q_in, q_out = quat_chunk[i, 3:7], back[i, 3:7]
        assert np.allclose(q_out, q_in, atol=1e-6) or np.allclose(q_out, -q_in, atol=1e-6)

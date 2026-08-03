"""Representation conversions between the inin bc-ur5-v2 schema and openpi.

On-disk / on-wire representation (inin-stream, schema ``bc-ur5-v2``):
    state  (14,): [joint_pos(6), tcp_xyz(3), tcp_quat_xyzw(4), gripper(1)]
    action (8,):  [tcp_xyz(3), tcp_quat_xyzw(4), gripper(1)]  (absolute)

Model representation (openpi pi0/pi0.5 with DeltaActions):
    state  (7,): [tcp_xyz(3), tcp_rotvec(3), gripper(1)]
    action (7,): [tcp_xyz(3), tcp_rotvec(3), gripper(1)]  (absolute)

Quaternions cannot be used with openpi's element-wise ``DeltaActions``, so
rotations are converted to axis-angle rotation vectors. Rotation vectors have
equivalent representations that differ by 2*pi about the same axis (and the
quaternion double cover flips the axis sign); ``nearest_equivalent_rotvec``
selects the representation closest to an anchor so that action chunks stay
continuous relative to the state they will be differenced against.
"""

import numpy as np
from scipy.spatial.transform import Rotation

STATE_DIM = 14
STATE7_DIM = 7
ACTION_QUAT_DIM = 8
ACTION_ROTVEC_DIM = 7

_TWO_PI = 2.0 * np.pi
_EPS = 1e-12


def quat_to_rotvec(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert an ``xyzw`` quaternion to an axis-angle rotation vector."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"expected quaternion shape (4,), got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if norm <= _EPS:
        raise ValueError("quaternion norm must be positive")
    return Rotation.from_quat(quat / norm).as_rotvec()


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Convert an axis-angle rotation vector to an ``xyzw`` quaternion."""
    vec = np.asarray(rotvec, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"expected rotation vector shape (3,), got {vec.shape}")
    return Rotation.from_rotvec(vec).as_quat()


def nearest_equivalent_rotvec(rotvec: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Return the rotation-vector representation of ``rotvec`` closest to ``anchor``.

    All vectors ``(theta + 2*pi*k) * axis`` (integer ``k``) describe the same
    rotation. Without this, a trajectory crossing the pi-rotation boundary
    flips sign (a jump of ~2*pi), which poisons delta actions and norm stats.
    """
    vec = np.asarray(rotvec, dtype=np.float64)
    ref = np.asarray(anchor, dtype=np.float64)
    theta = float(np.linalg.norm(vec))
    if theta <= _EPS:
        # Identity rotation: equivalents are 2*pi*k about any axis. Only the
        # anchor direction can beat the zero vector.
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm > np.pi:
            return _TWO_PI * ref / ref_norm
        return vec
    axis = vec / theta
    k = round((float(ref @ axis) - theta) / _TWO_PI)
    candidates = [(theta + _TWO_PI * i) * axis for i in (k - 1, k, k + 1)]
    return min(candidates, key=lambda c: float(np.linalg.norm(c - ref)))


def state14_to_state7(state: np.ndarray) -> np.ndarray:
    """[joint(6), tcp_xyz(3), tcp_quat(4), grip(1)] -> [tcp_xyz(3), rotvec(3), grip(1)]."""
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape != (STATE_DIM,):
        raise ValueError(f"expected state shape ({STATE_DIM},), got {arr.shape}")
    xyz = arr[6:9]
    rotvec = quat_to_rotvec(arr[9:13])
    return np.concatenate([xyz, rotvec, arr[13:14]]).astype(np.float32)


def quat_action_chunk_to_rotvec_chunk(chunk: np.ndarray, anchor_rotvec: np.ndarray) -> np.ndarray:
    """Convert an absolute (N, 8) xyz+quat+gripper chunk to (N, 7) xyz+rotvec+gripper.

    Each row's rotation vector is wrapped to the nearest equivalent of the
    previous row (starting from ``anchor_rotvec``, typically the state's
    rotation vector) so the chunk is continuous with the state.
    """
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_QUAT_DIM:
        raise ValueError(f"expected chunk shape (N, {ACTION_QUAT_DIM}), got {arr.shape}")
    out = np.empty((arr.shape[0], ACTION_ROTVEC_DIM), dtype=np.float64)
    prev = np.asarray(anchor_rotvec, dtype=np.float64)
    for i, row in enumerate(arr):
        rotvec = nearest_equivalent_rotvec(quat_to_rotvec(row[3:7]), prev)
        out[i, :3] = row[:3]
        out[i, 3:6] = rotvec
        out[i, 6] = row[7]
        prev = rotvec
    return out.astype(np.float32)


def rotvec_chunk_to_quat_chunk(chunk: np.ndarray) -> np.ndarray:
    """Convert an absolute (N, 7) xyz+rotvec+gripper chunk to (N, 8) xyz+quat+gripper."""
    arr = np.asarray(chunk, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_ROTVEC_DIM:
        raise ValueError(f"expected chunk shape (N, {ACTION_ROTVEC_DIM}), got {arr.shape}")
    out = np.empty((arr.shape[0], ACTION_QUAT_DIM), dtype=np.float64)
    for i, row in enumerate(arr):
        out[i, :3] = row[:3]
        out[i, 3:7] = rotvec_to_quat(row[3:6])
        out[i, 7] = row[6]
    return out.astype(np.float32)

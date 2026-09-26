"""Small NumPy-only SO(3) helpers used by the standalone V3 scorers."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def rotation_magnitude(matrices: np.ndarray) -> np.ndarray:
    """Return the principal rotation angle in radians for each 3x3 matrix."""
    values = np.asarray(matrices, dtype=np.float64)
    if values.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must end in shape (3, 3)")
    cosine = (np.trace(values, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert one rotation matrix to a normalized (w, x, y, z) quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * scale,
            (m[2, 1] - m[1, 2]) / scale,
            (m[0, 2] - m[2, 0]) / scale,
            (m[1, 0] - m[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            scale = np.sqrt(max(_EPS, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
            quaternion = np.array([
                (m[2, 1] - m[1, 2]) / scale,
                0.25 * scale,
                (m[0, 1] + m[1, 0]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(max(_EPS, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
            quaternion = np.array([
                (m[0, 2] - m[2, 0]) / scale,
                (m[0, 1] + m[1, 0]) / scale,
                0.25 * scale,
                (m[1, 2] + m[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(max(_EPS, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
            quaternion = np.array([
                (m[1, 0] - m[0, 1]) / scale,
                (m[0, 2] + m[2, 0]) / scale,
                (m[1, 2] + m[2, 1]) / scale,
                0.25 * scale,
            ])
    norm = float(np.linalg.norm(quaternion))
    if norm <= _EPS:
        raise ValueError("rotation matrix produced a zero quaternion")
    return quaternion / norm


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= _EPS:
        raise ValueError("zero quaternion cannot represent a rotation")
    w, x, y, z = q / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _slerp(left: np.ndarray, right: np.ndarray, fraction: float) -> np.ndarray:
    q0, q1 = left.copy(), right.copy()
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = q0 + fraction * (q1 - q0)
        return value / np.linalg.norm(value)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    return (np.sin((1.0 - fraction) * angle) / sine * q0
            + np.sin(fraction * angle) / sine * q1)


def interpolate_rotations(
    sample_positions: np.ndarray,
    matrices: np.ndarray,
    target_positions: np.ndarray,
) -> np.ndarray:
    """Quaternion SLERP over a strictly increasing scalar sample axis."""
    positions = np.asarray(sample_positions, dtype=np.float64)
    rotations = np.asarray(matrices, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if positions.ndim != 1 or len(positions) == 0:
        raise ValueError("sample_positions must be a nonempty vector")
    if rotations.shape != (len(positions), 3, 3):
        raise ValueError("matrices must have shape (len(sample_positions), 3, 3)")
    if len(positions) > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("sample_positions must be strictly increasing")
    if not (np.isfinite(positions).all() and np.isfinite(rotations).all()
            and np.isfinite(targets).all()):
        raise ValueError("rotation interpolation inputs must be finite")

    quaternions = np.stack([_matrix_to_quaternion(matrix) for matrix in rotations])
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0
    if len(positions) == 1:
        return np.repeat(rotations[:1], targets.size, axis=0)

    output = []
    for target in targets.reshape(-1):
        if target <= positions[0]:
            quaternion = quaternions[0]
        elif target >= positions[-1]:
            quaternion = quaternions[-1]
        else:
            right = int(np.searchsorted(positions, target, side="right"))
            left = right - 1
            fraction = float((target - positions[left]) / (positions[right] - positions[left]))
            quaternion = _slerp(quaternions[left], quaternions[right], fraction)
        output.append(_quaternion_to_matrix(quaternion))
    return np.stack(output).reshape(targets.shape + (3, 3))

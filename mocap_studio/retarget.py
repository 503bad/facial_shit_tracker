"""Quaternion / retargeting math (numpy, quaternions as (x, y, z, w))."""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking direction a to direction b."""
    a = normalize(np.asarray(a, dtype=np.float64))
    b = normalize(np.asarray(b, dtype=np.float64))
    d = float(np.dot(a, b))
    if d > 0.999999:
        return IDENTITY.copy()
    if d < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = normalize(axis)
        return np.array([axis[0], axis[1], axis[2], 0.0])
    axis = np.cross(a, b)
    q = np.array([axis[0], axis[1], axis[2], 1.0 + d])
    return q / np.linalg.norm(q)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3, columns are basis vectors) -> quaternion."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


def quat_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    m = np.column_stack([normalize(x), normalize(y), normalize(z)])
    return quat_from_matrix(m)


def frame_rotation(ref_axes: tuple[np.ndarray, np.ndarray],
                   cur_axes: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Rotation taking an orthonormalized (primary, secondary) reference
    frame onto the current frame."""

    def build(primary, secondary):
        p = normalize(np.asarray(primary, dtype=np.float64))
        s = np.asarray(secondary, dtype=np.float64)
        s = normalize(s - p * np.dot(s, p))
        t = np.cross(p, s)
        return np.column_stack([p, s, t])

    m_ref = build(*ref_axes)
    m_cur = build(*cur_axes)
    return quat_from_matrix(m_cur @ m_ref.T)


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(np.asarray(axis, dtype=np.float64))
    half = angle_rad / 2.0
    s = np.sin(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


def twist_angle(q: np.ndarray, axis: np.ndarray) -> float:
    """Signed rotation of q about the given (unit) axis, in radians
    (-pi, pi].  This is the twist part of a swing/twist decomposition."""
    axis = normalize(np.asarray(axis, dtype=np.float64))
    proj = float(np.dot(q[:3], axis))
    theta = 2.0 * np.arctan2(proj, q[3])
    if theta > np.pi:
        theta -= 2.0 * np.pi
    elif theta <= -np.pi:
        theta += 2.0 * np.pi
    return theta


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(np.asarray(a, dtype=np.float64))
    b = normalize(np.asarray(b, dtype=np.float64))
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))

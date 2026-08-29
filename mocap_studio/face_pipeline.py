"""Face pipeline: NVIDIA FaceExpressions output -> iFacialMocap v1 fields.

- Maps the 53 NVIDIA coefficients onto the 52 iFacialMocap keys
  (browInnerUp / cheekPuff L+R merged, tongueOut = 0).
- Optional neutral-face calibration (same scheme as NVIDIA's ExpressionApp).
- One Euro smoothing for expressions, slerp smoothing for head pose.
- Converts the head pose quaternion (OpenGL convention: X-right, Y-up,
  Z-back) into Unity Euler degrees as consumed by iFM receivers.
"""

from __future__ import annotations

import math

import numpy as np

from .nvar import EXPRESSION_NAMES
from .smoothing import QuaternionSmoother, SmoothedChannels

_IDX = {name: i for i, name in enumerate(EXPRESSION_NAMES)}

# iFM key -> list of NVIDIA coefficient indices to average.
_MAP: dict[str, list[int]] = {}
for _name in EXPRESSION_NAMES:
    _MAP[_name] = [_IDX[_name]]
_MAP["browInnerUp"] = [_IDX["browInnerUp_L"], _IDX["browInnerUp_R"]]
_MAP["cheekPuff"] = [_IDX["cheekPuff_L"], _IDX["cheekPuff_R"]]
for _k in ("browInnerUp_L", "browInnerUp_R", "cheekPuff_L", "cheekPuff_R"):
    del _MAP[_k]

_EYE_RANGE_DEG = 30.0


def _mirror_key(key: str) -> str:
    if key.endswith("_L"):
        return key[:-2] + "_R"
    if key.endswith("_R"):
        return key[:-2] + "_L"
    if key == "jawLeft":
        return "jawRight"
    if key == "jawRight":
        return "jawLeft"
    if key == "mouthLeft":
        return "mouthRight"
    if key == "mouthRight":
        return "mouthLeft"
    return key


def _quat_to_unity_euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """NvAR pose quat (x,y,z,w, GL convention) -> Unity Euler degrees.

    GL (right-handed, Z toward camera) -> Unity (left-handed, Z away) is a
    mirror in Z: (x, y, z, w) -> (-x, -y, z, w).  Then decompose as Unity's
    Quaternion.Euler order (q = Qy * Qx * Qz).
    """
    x, y, z, w = -q[0], -q[1], q[2], q[3]
    # rotation matrix from quaternion
    m00 = 1 - 2 * (y * y + z * z)
    m02 = 2 * (x * z + w * y)
    m10 = 2 * (x * y + w * z)
    m11 = 1 - 2 * (x * x + z * z)
    m12 = 2 * (y * z - w * x)
    m22 = 1 - 2 * (x * x + y * y)
    # R = Ry(b) Rx(a) Rz(c):  m12 = -sin a, m02/m22 -> b, m10/m11 -> c
    a = math.asin(max(-1.0, min(1.0, -m12)))
    b = math.atan2(m02, m22)
    c = math.atan2(m10, m11)
    return (math.degrees(a), math.degrees(b), math.degrees(c))


class FaceCalibration:
    """Neutral-face calibration (ExpressionApp scheme)."""

    def __init__(self) -> None:
        self.zero = np.zeros(len(EXPRESSION_NAMES))
        self.scale = np.ones(len(EXPRESSION_NAMES))
        self.enabled = False

    def calibrate(self, neutral: np.ndarray) -> None:
        self.zero = neutral.copy()
        self.scale = 1.0 / np.maximum(1.0 - self.zero, 0.25)
        self.enabled = True

    def clear(self) -> None:
        self.enabled = False

    def apply(self, expr: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return expr
        return np.clip((expr - self.zero) * self.scale, 0.0, 1.0)


class FacePipeline:
    def __init__(self, expr_strength: float = 0.3,
                 head_strength: float = 0.35) -> None:
        self.expr_smoother = SmoothedChannels(expr_strength)
        self.head_rot_smoother = QuaternionSmoother(head_strength)
        self.head_pos_smoother = SmoothedChannels(head_strength)
        self.calibration = FaceCalibration()
        self._pending_calibration = False
        self.latest_expr: dict[str, float] = {}
        self.mirror = True

    def request_calibration(self) -> None:
        self._pending_calibration = True

    def clear_calibration(self) -> None:
        self.calibration.clear()

    def set_strengths(self, expr: float, head: float) -> None:
        self.expr_smoother.set_strength(expr)
        self.head_rot_smoother.set_strength(head)
        self.head_pos_smoother.set_strength(head)

    def process(self, expr53: np.ndarray, pose_quat: np.ndarray,
                translation: np.ndarray, t: float):
        """Returns (blendshapes dict, head_euler_deg, head_pos_m,
        right_eye_deg, left_eye_deg) ready for IfmSender.send()."""
        if self._pending_calibration:
            self.calibration.calibrate(expr53)
            self._pending_calibration = False

        expr = self.calibration.apply(expr53)
        expr = self.expr_smoother.apply(expr, t)

        bs = {key: float(np.mean(expr[idxs])) for key, idxs in _MAP.items()}
        bs["tongueOut"] = 0.0
        if self.mirror:
            bs = {key: bs[_mirror_key(key)] for key in bs}
        self.latest_expr = bs

        q = self.head_rot_smoother.apply(pose_quat, t)
        head_euler = _quat_to_unity_euler_deg(q)
        # NvAR PoseTranslation is in centimeters; iFacialMocap expects
        # meters (a raw value puts the head ~80 m away at the receiver).
        pos = self.head_pos_smoother.apply(translation * 0.01, t)
        head_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        if self.mirror:
            head_euler = (head_euler[0], -head_euler[1], -head_euler[2])
            head_pos = (-head_pos[0], head_pos[1], head_pos[2])

        # Eye euler from look blendshapes (receivers mostly use the
        # blendshapes; these keep full compatibility).  X: down positive.
        def _eye(up: str, down: str, inn: str, out: str, out_sign: float):
            pitch = (bs[down] - bs[up]) * _EYE_RANGE_DEG
            yaw = (bs[out] - bs[inn]) * _EYE_RANGE_DEG * out_sign
            return (pitch, yaw, 0.0)

        right_eye = _eye("eyeLookUp_R", "eyeLookDown_R",
                         "eyeLookIn_R", "eyeLookOut_R", 1.0)
        left_eye = _eye("eyeLookUp_L", "eyeLookDown_L",
                        "eyeLookIn_L", "eyeLookOut_L", -1.0)
        return bs, head_euler, head_pos, right_eye, left_eye

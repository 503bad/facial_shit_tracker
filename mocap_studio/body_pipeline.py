"""Body retargeting: tracked 3D points -> VMC local bone rotations.

All input points are in Unity space (left-handed, Y-up, avatar facing +Z,
avatar's left = -X).  Bone local rotations are relative to the humanoid
T-pose, quaternions (x, y, z, w).

Seated use: legs are pinned to T-pose, root stays fixed; spine lean/twist
and full arm chains are driven from tracking.  Fingers come either from
MediaPipe hand landmarks (articulated) or from a grip estimate (NVIDIA).
"""

from __future__ import annotations

import numpy as np

from .retarget import (IDENTITY, angle_between, frame_rotation, normalize,
                       quat_from_axis_angle, quat_from_axes,
                       quat_from_two_vectors, quat_inv, quat_mul,
                       twist_angle)
from .smoothing import QuaternionSmoother, SmoothedChannels, quat_slerp

# T-pose reference directions in Unity space.
_REF_UP = np.array([0.0, 1.0, 0.0])
_REF_SHOULDER_LINE = np.array([1.0, 0.0, 0.0])   # left shoulder -> right
_REF_ARM = {"left": np.array([-1.0, 0.0, 0.0]),
            "right": np.array([1.0, 0.0, 0.0])}

# Default bind-pose local offsets (from the user's VRM); replaced when a
# model is loaded via vrm_loader.
DEFAULT_BONE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "Hips": (0.0, 1.0795, 0.0), "Spine": (0.0, 0.0571, 0.0),
    "Chest": (0.0, 0.0987, 0.0), "Neck": (0.0, 0.3303, 0.0),
    "Head": (0.0, 0.0715, 0.0),
    "LeftShoulder": (0.0265, 0.3129, 0.0),
    "LeftUpperArm": (0.1866, -0.0363, 0.0),
    "LeftLowerArm": (0.2132, -0.0351, 0.0),
    "LeftHand": (0.2987, 0.0, 0.0),
    "RightShoulder": (-0.0265, 0.3129, 0.0),
    "RightUpperArm": (-0.1866, -0.0363, 0.0),
    "RightLowerArm": (-0.2132, -0.0351, 0.0),
    "RightHand": (-0.2987, 0.0, 0.0),
    "LeftUpperLeg": (0.0989, -0.03, 0.0),
    "LeftLowerLeg": (0.0, -0.5372, 0.0),
    "LeftFoot": (0.0, -0.4222, 0.0), "LeftToes": (0.0, -0.09, 0.1146),
    "RightUpperLeg": (-0.0989, -0.03, 0.0),
    "RightLowerLeg": (0.0, -0.5372, 0.0),
    "RightFoot": (0.0, -0.4222, 0.0), "RightToes": (0.0, -0.09, 0.1146),
    "LeftThumbProximal": (0.0469, 0.0, 0.0378),
    "LeftThumbIntermediate": (0.024, -0.014, 0.013),
    "LeftThumbDistal": (0.0274, 0.0, 0.0001),
    "LeftIndexProximal": (0.1042, 0.0, 0.0235),
    "LeftIndexIntermediate": (0.0388, 0.0, 0.0),
    "LeftIndexDistal": (0.0301, 0.0, 0.0),
    "LeftMiddleProximal": (0.1092, 0.0, 0.0015),
    "LeftMiddleIntermediate": (0.0391, 0.0, 0.0),
    "LeftMiddleDistal": (0.0326, 0.0, 0.0),
    "LeftRingProximal": (0.1095, 0.0, -0.0213),
    "LeftRingIntermediate": (0.0312, 0.0, 0.0),
    "LeftRingDistal": (0.029, 0.0, 0.0),
    "LeftLittleProximal": (0.1063, 0.0, -0.0427),
    "LeftLittleIntermediate": (0.021, 0.0, 0.0),
    "LeftLittleDistal": (0.0185, 0.0, 0.0),
    "RightThumbProximal": (-0.0469, 0.0, 0.0378),
    "RightThumbIntermediate": (-0.024, -0.014, 0.013),
    "RightThumbDistal": (-0.0274, 0.0, 0.0001),
    "RightIndexProximal": (-0.1042, 0.0, 0.0235),
    "RightIndexIntermediate": (-0.0388, 0.0, 0.0),
    "RightIndexDistal": (-0.0301, 0.0, 0.0),
    "RightMiddleProximal": (-0.1092, 0.0, 0.0015),
    "RightMiddleIntermediate": (-0.0391, 0.0, 0.0),
    "RightMiddleDistal": (-0.0326, 0.0, 0.0),
    "RightRingProximal": (-0.1095, 0.0, -0.0213),
    "RightRingIntermediate": (-0.0312, 0.0, 0.0),
    "RightRingDistal": (-0.029, 0.0, 0.0),
    "RightLittleProximal": (-0.1063, 0.0, -0.0427),
    "RightLittleIntermediate": (-0.021, 0.0, 0.0),
    "RightLittleDistal": (-0.0185, 0.0, 0.0),
}

# Bones owned by the face tracker (iFacialMocap side); never sent over VMC
# unless the body pipeline actually computes them.
_FACE_DRIVEN_BONES = frozenset(
    {"Neck", "Head", "LeftEye", "RightEye", "Jaw"})

_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
_SEGMENTS = ("Proximal", "Intermediate", "Distal")
# MediaPipe hand landmark chains per finger: [base, mcp, pip/ip, dip, tip]
_MP_FINGER_CHAIN = {
    "Thumb": [0, 1, 2, 3, 4],
    "Index": [0, 5, 6, 7, 8],
    "Middle": [0, 9, 10, 11, 12],
    "Ring": [0, 13, 14, 15, 16],
    "Little": [0, 17, 18, 19, 20],
}
_MAX_CURL_RAD = np.deg2rad(100.0)
_MAX_SPLAY_RAD = np.deg2rad(32.0)

# Canonical relaxed-hand rest values.  These are the default zero points;
# the calibration button re-captures them from the user's actual hand.
# Curl rests per finger segment (radians), _FINGERS order.
_CANON_CURL_REST = np.array([
    0.35, 0.25, 0.15,   # thumb (used only in the axis-based fallback)
    0.15, 0.12, 0.08,   # index
    0.15, 0.12, 0.08,   # middle
    0.15, 0.12, 0.08,   # ring
    0.15, 0.12, 0.08,   # little
])
# In-palm splay rests (radians, positive = toward thumb side).
_CANON_SPLAY_REST = np.array([0.75, 0.10, 0.0, -0.12, -0.30])
# Thumb segment rest directions in the avatar hand-local frame.
_DEFAULT_THUMB_REST = {
    "Left": np.array([[-0.68, -0.30, 0.66], [-0.58, -0.25, 0.77],
                      [-0.55, -0.25, 0.80]]),
    "Right": np.array([[0.68, -0.30, 0.66], [0.58, -0.25, 0.77],
                       [0.55, -0.25, 0.80]]),
}


class BodyRetargeter:
    def __init__(self, body_strength: float = 0.5,
                 finger_strength: float = 0.6,
                 send_head: bool = False) -> None:
        self.send_head = send_head
        self.bone_offsets = dict(DEFAULT_BONE_OFFSETS)
        self._rot_smoothers: dict[str, QuaternionSmoother] = {}
        self._body_strength = body_strength
        # 30 channels: 2 hands x 5 fingers x 3 segment angles
        self._finger_smoother = SmoothedChannels(finger_strength)
        self._finger_angles = np.zeros(30)
        # curl axes in Unity local space per hand (sign flips tunable)
        self.curl_sign = {"Left": 1.0, "Right": -1.0}
        # Thumb curls in a tilted plane: +Z (T-pose thumb dir) swings toward
        # the palm center (-X,-Y for the left hand; +X,-Y for the right).
        self.thumb_axis = {
            "Left": normalize(np.array([0.53, -0.85, 0.0])),
            "Right": normalize(np.array([0.53, 0.85, 0.0])),
        }

        # Neutral-posture / finger-rest calibration (auto at start; can be
        # re-triggered with start_calibration()).
        self._CALIB_FRAMES = 40
        self._chest_neutral: np.ndarray | None = None
        self._chest_samples = 0
        # Finger rest starts from canonical relaxed-hand values; capturing
        # from "first frames the hand is visible" is unreliable because the
        # hand usually appears mid-gesture.  Only the calibration button
        # re-captures (start_calibration).
        self._rest_angles = np.tile(_CANON_CURL_REST, 2)
        self._rest_splay = np.tile(_CANON_SPLAY_REST, 2)
        self._rest_counts = [self._CALIB_FRAMES, self._CALIB_FRAMES]
        self._FINGER_GAIN = 1.3
        # Splay (finger spread): 10 channels = 2 hands x 5 fingers,
        # signed angle in the palm plane toward the thumb side.
        self._splay_smoother = SmoothedChannels(finger_strength)
        self._splay_angles = np.zeros(10)
        self.splay_sign = {"Left": 1.0, "Right": -1.0}
        # Thumb: direction-based 3DOF retarget (the saddle joint can't be
        # represented by a fixed rotation axis).  Rest directions per hand,
        # 3 segments each, expressed in the avatar hand-local frame.
        self._thumb_rest: list[np.ndarray | None] = [None, None]
        self._thumb_smoothers: dict[str, QuaternionSmoother] = {}
        self._finger_strength = finger_strength
        # Head pose fed from the face tracker (Unity space, mirror applied),
        # sent over VMC split across Neck/Head for receivers whose body
        # tracking owns the whole skeleton.
        self._head_quat: np.ndarray | None = None
        self._head_neutral: np.ndarray | None = None
        self._head_samples = 0
        self._NECK_SHARE = 0.35
        # Micro-tremor gate: each bone trails its target by up to gate_rad,
        # so sub-threshold jitter produces zero output motion while larger
        # movements pass through without popping.
        self.gate_enabled = True
        self.gate_rad = np.deg2rad(2.0)
        self._gate_held: dict[str, np.ndarray] = {}

    def start_calibration(self) -> None:
        """Re-capture the neutral torso pose and, for any hand that is
        visible during the capture window, the relaxed finger rest angles.
        The user should sit relaxed with open hands in view."""
        self._chest_neutral = None
        self._chest_samples = 0
        self._rest_angles = np.tile(_CANON_CURL_REST, 2)
        self._rest_splay = np.tile(_CANON_SPLAY_REST, 2)
        self._rest_counts = [0, 0]
        self._thumb_rest = [None, None]
        self._head_neutral = None
        self._head_samples = 0

    def set_bone_offsets(
            self, offsets: dict[str, tuple[float, float, float]]) -> None:
        merged = dict(DEFAULT_BONE_OFFSETS)
        merged.update(offsets)
        self.bone_offsets = merged

    def set_strengths(self, body: float, finger: float) -> None:
        self._body_strength = body
        for s in self._rot_smoothers.values():
            s.set_strength(body)
        self._finger_smoother.set_strength(finger)
        self._splay_smoother.set_strength(finger)
        self._finger_strength = finger
        for s in self._thumb_smoothers.values():
            s.set_strength(finger)

    def _smooth(self, bone: str, quat: np.ndarray, t: float) -> np.ndarray:
        sm = self._rot_smoothers.get(bone)
        if sm is None:
            sm = QuaternionSmoother(self._body_strength)
            self._rot_smoothers[bone] = sm
        return sm.apply(quat, t)

    # ------------------------------------------------------------------
    def process(self, pose: dict[str, np.ndarray] | None,
                left_hand: np.ndarray | None,
                right_hand: np.ndarray | None,
                t: float) -> dict[str, tuple[tuple, tuple]]:
        """Build the full VMC bone dict {name: (pos, quat)}."""
        rotations: dict[str, np.ndarray] = {}

        if pose is not None:
            rotations.update(self._solve_torso_and_arms(
                pose, left_hand, right_hand))

        # Smooth every driven rotation.
        for bone, q in rotations.items():
            rotations[bone] = self._smooth(bone, q, t)

        rotations.update(self._solve_head())
        rotations.update(self._solve_fingers(left_hand, right_hand, t))
        rotations = self._apply_gate(rotations)

        bones: dict[str, tuple[tuple, tuple]] = {}
        for bone, offset in self.bone_offsets.items():
            if bone in _FACE_DRIVEN_BONES and bone not in rotations:
                # Not driven by the body tracker: sending identity would
                # override the iFacialMocap-driven head/eyes at the
                # receiver, pinning the head to face forward.
                continue
            q = rotations.get(bone, IDENTITY)
            bones[bone] = (offset, (float(q[0]), float(q[1]),
                                    float(q[2]), float(q[3])))
        return bones

    # ------------------------------------------------------------------
    def _solve_torso_and_arms(self, p: dict[str, np.ndarray],
                              left_hand, right_hand):
        rot: dict[str, np.ndarray] = {}
        ls, rs = p["left_shoulder"], p["right_shoulder"]
        lh, rh = p["left_hip"], p["right_hip"]
        hips_c = (lh + rh) / 2.0
        shoulder_c = (ls + rs) / 2.0

        torso_up = normalize(shoulder_c - hips_c)
        shoulder_line = normalize(rs - ls)  # current left->right line
        chest_raw = frame_rotation((_REF_UP, _REF_SHOULDER_LINE),
                                   (torso_up, shoulder_line))

        # Neutral calibration: average the first frames after (re)start and
        # output only the rotation relative to that neutral pose.  This
        # cancels the systematic lean caused by camera angle / seated pose.
        if self._chest_samples < self._CALIB_FRAMES:
            if self._chest_neutral is None:
                self._chest_neutral = chest_raw.copy()
            else:
                self._chest_neutral = quat_slerp(
                    self._chest_neutral, chest_raw,
                    1.0 / (self._chest_samples + 1))
            self._chest_samples += 1
        chest_out = quat_mul(chest_raw, quat_inv(self._chest_neutral)) \
            if self._chest_neutral is not None else IDENTITY.copy()

        # Distribute torso rotation over Spine and Chest (half each, as a
        # quaternion "square root" via slerp from identity).
        half = quat_slerp(IDENTITY, chest_out, 0.5)
        rot["Spine"] = half
        rot["Chest"] = quat_mul(quat_inv(half), chest_out)
        # Arm-swing math stays in the RAW (measured) chest frame so arms
        # remain correct relative to the real torso.
        chest_global = chest_raw

        if self.send_head and "nose" in p and "left_ear" in p:
            ear_line = normalize(p["right_ear"] - p["left_ear"])
            head_fwd = normalize(
                p["nose"] - (p["left_ear"] + p["right_ear"]) / 2.0)
            head_global = frame_rotation(
                (_REF_SHOULDER_LINE, np.array([0.0, 0.0, 1.0])),
                (ear_line, head_fwd))
            rot["Neck"] = quat_mul(quat_inv(chest_global), head_global)

        for side, hand_lm in (("left", left_hand), ("right", right_hand)):
            cap = side.capitalize()
            sh = p[f"{side}_shoulder"]
            el = p[f"{side}_elbow"]
            wr = p[f"{side}_wrist"]
            ref = _REF_ARM[side]

            upper_local = _swing_in_parent(chest_global, ref, el - sh)
            upper_g = quat_mul(chest_global, upper_local)
            lower_local = _swing_in_parent(upper_g, ref, wr - el)
            lower_g = quat_mul(upper_g, lower_local)

            hand_g = self._hand_orientation(side, hand_lm, ref, lower_g,
                                            wr, el)
            if hand_g is not None:
                # Twist distribution: humans pronate with the whole forearm,
                # not the wrist joint.  Take the hand's twist about the
                # forearm axis and spread it up the chain (upper arm 25%,
                # forearm 50%, remainder stays at the wrist), re-solving the
                # swings so elbow/wrist positions are unchanged.
                local_hand = quat_mul(quat_inv(lower_g), hand_g)
                tw = twist_angle(local_hand, ref)
                fu, fl = 0.25, 0.50
                upper_local = quat_mul(
                    upper_local, quat_from_axis_angle(ref, tw * fu))
                upper_g = quat_mul(chest_global, upper_local)
                lower_local = quat_mul(
                    _swing_in_parent(upper_g, ref, wr - el),
                    quat_from_axis_angle(ref, tw * fl))
                lower_g = quat_mul(upper_g, lower_local)
                rot[f"{cap}Hand"] = quat_mul(quat_inv(lower_g), hand_g)

            rot[f"{cap}UpperArm"] = upper_local
            rot[f"{cap}LowerArm"] = lower_local
        return rot

    def _hand_orientation(self, side: str, hand_lm, ref, lower_g, wr, el):
        if hand_lm is None:
            return None
        wrist = hand_lm[0]
        middle_mcp = hand_lm[9]
        index_mcp = hand_lm[5]
        little_mcp = hand_lm[17]
        hand_dir = normalize(middle_mcp - wrist)
        # Palm normal, pointing out of the palm (T-pose palms face down, -Y).
        # Verified for Unity LH space: left hand palm-down has index_mcp on
        # +Z of little_mcp, so cross(index, little) (algebraic formula)
        # points -Y; mirrored for the right hand.
        if side == "left":
            palm = normalize(np.cross(index_mcp - wrist, little_mcp - wrist))
        else:
            palm = normalize(np.cross(little_mcp - wrist, index_mcp - wrist))
        hand_global = frame_rotation(
            (ref, np.array([0.0, -1.0, 0.0])), (hand_dir, palm))
        return hand_global

    # ------------------------------------------------------------------
    def _solve_fingers(self, left_hand, right_hand, t: float):
        angles = self._finger_angles.copy()
        splay = self._splay_angles.copy()
        for hi, (side, lm) in enumerate((("Left", left_hand),
                                         ("Right", right_hand))):
            if lm is None:
                continue
            # Hand-plane frame for splay measurement: along-hand axis and
            # thumb-side axis (little_mcp -> index_mcp, orthogonalized).
            wrist, index_mcp, middle_mcp, little_mcp = (
                lm[0], lm[5], lm[9], lm[17])
            hand_dir = normalize(middle_mcp - wrist)
            t_side = np.asarray(index_mcp - little_mcp, dtype=np.float64)
            t_side = normalize(t_side - hand_dir * np.dot(t_side, hand_dir))
            for fi, finger in enumerate(_FINGERS):
                chain = _MP_FINGER_CHAIN[finger]
                pts = lm[chain]
                for si in range(3):
                    a = pts[si + 1] - pts[si]
                    b = pts[si + 2] - pts[si + 1]
                    ang = angle_between(a, b)
                    angles[hi * 15 + fi * 3 + si] = min(ang, _MAX_CURL_RAD)
                # Splay: signed in-plane angle of the proximal segment
                # (mcp -> pip) relative to the along-hand axis.
                d = pts[2] - pts[1]
                splay[hi * 5 + fi] = float(np.arctan2(
                    np.dot(d, t_side), max(np.dot(d, hand_dir), 1e-6)))
            # Rest-angle calibration: the relaxed hand at start defines the
            # per-joint zero point (the thumb in particular has a large
            # constant structural angle that must not read as curl, and
            # every finger has a natural rest splay).
            if self._rest_counts[hi] < self._CALIB_FRAMES:
                n = self._rest_counts[hi]
                sl = slice(hi * 15, hi * 15 + 15)
                sp = slice(hi * 5, hi * 5 + 5)
                self._rest_angles[sl] = (
                    self._rest_angles[sl] * n + angles[sl]) / (n + 1)
                self._rest_splay[sp] = (
                    self._rest_splay[sp] * n + splay[sp]) / (n + 1)
                self._rest_counts[hi] = n + 1
        self._finger_angles = angles
        self._splay_angles = splay
        effective = np.clip((angles - self._rest_angles) * self._FINGER_GAIN,
                            0.0, _MAX_CURL_RAD)
        smoothed = self._finger_smoother.apply(effective, t)
        eff_splay = np.clip(splay - self._rest_splay,
                            -_MAX_SPLAY_RAD, _MAX_SPLAY_RAD)
        smoothed_splay = self._splay_smoother.apply(eff_splay, t)

        rot: dict[str, np.ndarray] = {}
        hands = {"Left": left_hand, "Right": right_hand}
        for hi, side in enumerate(("Left", "Right")):
            for fi, finger in enumerate(_FINGERS):
                if finger == "Thumb" and hands[side] is not None:
                    continue  # direction-based solve below
                for si, seg in enumerate(_SEGMENTS):
                    ang = float(smoothed[hi * 15 + fi * 3 + si])
                    if finger == "Thumb":
                        axis = self.thumb_axis[side]
                    else:
                        axis = np.array([0.0, 0.0, self.curl_sign[side]])
                    q = quat_from_axis_angle(axis, ang)
                    if si == 0 and finger != "Thumb":
                        # Spread is applied at the proximal joint only,
                        # rotating in the palm plane (local Y).
                        sp = float(smoothed_splay[hi * 5 + fi])
                        q_splay = quat_from_axis_angle(
                            np.array([0.0, self.splay_sign[side], 0.0]), sp)
                        q = quat_mul(q_splay, q)
                    rot[f"{side}{finger}{seg}"] = q
            if hands[side] is not None:
                self._solve_thumb(side, hi, hands[side], rot, t)
        return rot

    # ------------------------------------------------------------------
    def _apply_gate(self, rotations: dict[str, np.ndarray]
                    ) -> dict[str, np.ndarray]:
        if not self.gate_enabled:
            self._gate_held.clear()
            return rotations
        out: dict[str, np.ndarray] = {}
        for bone, q in rotations.items():
            held = self._gate_held.get(bone)
            if held is None:
                out[bone] = q
            else:
                dot = abs(float(np.dot(held, q)))
                ang = 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))
                if ang <= self.gate_rad:
                    out[bone] = held
                else:
                    out[bone] = quat_slerp(q, held, self.gate_rad / ang)
            self._gate_held[bone] = out[bone]
        return out

    # ------------------------------------------------------------------
    def set_head_pose(self, quat_unity: np.ndarray) -> None:
        """Feed the face tracker's head quaternion (Unity space, already
        smoothed and mirrored on the face side)."""
        self._head_quat = np.asarray(quat_unity, dtype=np.float64)

    def _solve_head(self) -> dict[str, np.ndarray]:
        if self._head_quat is None:
            return {}
        q = self._head_quat
        if self._head_samples < self._CALIB_FRAMES:
            if self._head_neutral is None:
                self._head_neutral = q.copy()
            else:
                self._head_neutral = quat_slerp(
                    self._head_neutral, q, 1.0 / (self._head_samples + 1))
            self._head_samples += 1
        if self._head_neutral is None:
            return {}
        rel = quat_mul(q, quat_inv(self._head_neutral))
        neck = quat_slerp(IDENTITY, rel, self._NECK_SHARE)
        head = quat_mul(quat_inv(neck), rel)
        return {"Neck": neck, "Head": head}

    # ------------------------------------------------------------------
    def _solve_thumb(self, side: str, hi: int, lm: np.ndarray,
                     rot: dict[str, np.ndarray], t: float) -> None:
        """Direction-based 3DOF thumb retarget.

        The thumb's saddle joint (opposition, thumbs-up, spreading) can't be
        expressed with a fixed rotation axis, so each segment direction is
        measured in the hand's own frame, mapped into the avatar hand-local
        frame, and applied as a chained swing from the rest direction."""
        ref = _REF_ARM[side.lower()]
        wrist, index_mcp, middle_mcp, little_mcp = (
            lm[0], lm[5], lm[9], lm[17])
        e1 = normalize(np.asarray(middle_mcp - wrist, dtype=np.float64))
        if side == "Left":
            pn = np.cross(index_mcp - wrist, little_mcp - wrist)
        else:
            pn = np.cross(little_mcp - wrist, index_mcp - wrist)
        e2 = normalize(pn - e1 * np.dot(pn, e1))
        e3 = np.cross(e1, e2)
        a1 = ref
        a2 = np.array([0.0, -1.0, 0.0])
        a3 = np.cross(a1, a2)

        def to_avatar(d):
            d = np.asarray(d, dtype=np.float64)
            return normalize(a1 * np.dot(d, e1) + a2 * np.dot(d, e2)
                             + a3 * np.dot(d, e3))

        dirs = np.array([to_avatar(lm[i + 1] - lm[i]) for i in (1, 2, 3)])

        if self._rest_counts[hi] <= self._CALIB_FRAMES:
            prev = self._thumb_rest[hi]
            if prev is None:
                self._thumb_rest[hi] = dirs.copy()
            else:
                n = max(self._rest_counts[hi], 1)
                mixed = (prev * n + dirs) / (n + 1)
                self._thumb_rest[hi] = np.array(
                    [normalize(v) for v in mixed])

        rest = self._thumb_rest[hi]
        if rest is None:
            rest = _DEFAULT_THUMB_REST[side]

        g = IDENTITY.copy()
        for seg, d_rest, d_cur in zip(_SEGMENTS, rest, dirs):
            d_in_parent = _rotate_vec(quat_inv(g), d_cur)
            q = quat_from_two_vectors(normalize(d_rest), d_in_parent)
            name = f"{side}Thumb{seg}"
            sm = self._thumb_smoothers.get(name)
            if sm is None:
                sm = QuaternionSmoother(self._finger_strength)
                self._thumb_smoothers[name] = sm
            q = sm.apply(q, t)
            rot[name] = q
            g = quat_mul(g, q)

    def set_grip(self, side: str, curl01: float) -> None:
        """Fallback for backends without hand landmarks: set a uniform
        curl (0=open, 1=fist) for one hand ('Left'/'Right')."""
        hi = 0 if side == "Left" else 1
        ang = float(np.clip(curl01, 0.0, 1.0)) * np.deg2rad(80.0)
        for fi in range(5):
            for si in range(3):
                a = ang * (0.5 if fi == 0 else 1.0)  # thumb curls less
                self._finger_angles[hi * 15 + fi * 3 + si] = a


def _swing_in_parent(parent_global: np.ndarray, ref_dir: np.ndarray,
                     cur_dir_world: np.ndarray) -> np.ndarray:
    """Swing rotation in parent-local space taking ref_dir to the current
    world direction expressed in the parent's frame."""
    from .retarget import quat_from_two_vectors
    inv = quat_inv(parent_global)
    d = _rotate_vec(inv, normalize(np.asarray(cur_dir_world,
                                              dtype=np.float64)))
    return quat_from_two_vectors(ref_dir, d)


def _rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    r = quat_mul(quat_mul(q, qv), quat_inv(q))
    return r[:3]

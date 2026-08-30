"""MediaPipe Holistic backend: pose + articulated hands from a BGR frame.

Outputs points converted to Unity space (left-handed, Y-up, avatar facing
+Z toward the render camera): unity = (-x, -y, -z) of MediaPipe world coords
(MediaPipe: x image-right, y down, z away from camera).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / \
    "holistic_landmarker.task"

# MediaPipe pose landmark indices
MP_POSE = {
    "nose": 0, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
    "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20,
    "left_thumb": 21, "right_thumb": 22,
    "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28,
    "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}
# Landmarks whose visibility is exported (legs may be out of frame).
_VIS_NAMES = ("left_hip", "right_hip",
              "left_knee", "right_knee", "left_ankle", "right_ankle",
              "left_foot_index", "right_foot_index")

_CONV = np.array([-1.0, -1.0, -1.0])
_MAX_WRIST_DIST = 0.12   # normalized image units, hand wrist vs pose wrist


def _to_unity(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks]) * _CONV


def _hand_to_unity(landmarks, w: int, h: int) -> np.ndarray:
    """Normalized image-space hand landmarks -> Unity-oriented points.

    The holistic task's hand *world* landmarks do not carry the hand
    model's articulation (fingers read as a generic flat hand), so hand
    geometry is built from the normalized landmarks instead, scaled by the
    image size for correct aspect (z is normalized against width).
    Orientation flips match _to_unity; absolute scale is irrelevant since
    only directions/angles are consumed downstream.
    """
    return np.array([[-p.x * w, -p.y * h, -p.z * w] for p in landmarks])


class MediaPipeBodyTracker:
    """Wraps HolisticLandmarker (VIDEO mode) for single-person tracking."""

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as tp
        from mediapipe.tasks.python import vision
        self._mp = mp
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipeモデルがありません: {MODEL_PATH}")
        opts = vision.HolisticLandmarkerOptions(
            base_options=tp.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.3,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.6,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opts)
        self._ts_ms = 0
        self._last_ts = 0

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Returns (pose_points dict name->unity xyz | None,
        left_hand[21,3] | None, right_hand[21,3] | None).

        Hands are anatomical left/right, assigned by proximity to the pose
        wrists — the task's own left/right labels follow the mirrored-selfie
        convention and are wrong for a plain webcam feed.
        """
        # Moderate downscale: cuts CPU cost while keeping the hand ROI crop
        # sharp enough for reliable finger landmarks on a fist.
        h, w = frame_bgr.shape[:2]
        if w > 960:
            scale = 960.0 / w
            frame_bgr = cv2.resize(frame_bgr, (960, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires monotonically increasing timestamps.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms
        res = self._landmarker.detect_for_video(img, timestamp_ms)

        pose = None
        if res.pose_world_landmarks:
            pts = _to_unity(res.pose_world_landmarks)
            pose = {name: pts[i] for name, i in MP_POSE.items()}
            # per-landmark visibility for the leg joints (0..1)
            pose["_vis"] = {
                name: float(res.pose_world_landmarks[MP_POSE[name]].visibility)
                for name in _VIS_NAMES}
            if res.pose_landmarks:
                # Image-space torso anchors (normalized) for pelvis
                # translation: world landmarks are hip-centred and carry
                # no absolute position.
                lm = res.pose_landmarks
                fh0, fw0 = frame_bgr.shape[:2]
                pose["_img"] = {
                    "hip_center": ((lm[23].x + lm[24].x) / 2,
                                   (lm[23].y + lm[24].y) / 2),
                    "shoulder_center": ((lm[11].x + lm[12].x) / 2,
                                        (lm[11].y + lm[12].y) / 2),
                    "aspect": fw0 / fh0,
                }

        fh, fw = frame_bgr.shape[:2]
        left, right = self._assign_hands(res, fw, fh)

        # Normalized 2D points for the preview overlay (resolution-free).
        debug2d = []
        if res.pose_landmarks:
            # kind 0 = pose (visible), kind 2 = pose but low visibility
            debug2d += [(p.x, p.y, 0 if p.visibility >= 0.5 else 2)
                        for p in res.pose_landmarks]
        for hand_lm in (res.left_hand_landmarks, res.right_hand_landmarks):
            if hand_lm:
                debug2d += [(p.x, p.y, 1) for p in hand_lm]
        return pose, left, right, debug2d

    @staticmethod
    def _assign_hands(res, w: int, h: int):
        """Assign detected hands to anatomical sides via pose wrists
        (normalized image coords). Falls back to swapping the task's
        mirrored labels when no pose is available.  Geometry comes from
        the normalized hand landmarks (see _hand_to_unity)."""
        candidates = [lm for lm in (res.left_hand_landmarks,
                                    res.right_hand_landmarks) if lm]
        if not candidates:
            return None, None

        if not res.pose_landmarks:
            # labels are mirrored for a non-selfie feed -> swap them
            left = _hand_to_unity(res.right_hand_landmarks, w, h) \
                if res.right_hand_landmarks else None
            right = _hand_to_unity(res.left_hand_landmarks, w, h) \
                if res.left_hand_landmarks else None
            return left, right

        lw = res.pose_landmarks[15]  # left_wrist (normalized)
        rw = res.pose_landmarks[16]  # right_wrist

        def dists(norm):
            w0 = norm[0]  # hand wrist landmark
            dl = (w0.x - lw.x) ** 2 + (w0.y - lw.y) ** 2
            dr = (w0.x - rw.x) ** 2 + (w0.y - rw.y) ** 2
            return dl, dr

        # A hand whose wrist is far from both pose wrists is a bad crop
        # (edge of frame, fast motion): treat it as not detected.
        candidates = [c for c in candidates
                      if min(dists(c)) <= _MAX_WRIST_DIST ** 2]
        if not candidates:
            return None, None

        if len(candidates) == 1:
            dl, dr = dists(candidates[0])
            hand = _hand_to_unity(candidates[0], w, h)
            return (hand, None) if dl <= dr else (None, hand)

        d0 = dists(candidates[0])
        d1 = dists(candidates[1])
        if d0[0] + d1[1] <= d0[1] + d1[0]:
            return (_hand_to_unity(candidates[0], w, h),
                    _hand_to_unity(candidates[1], w, h))
        return (_hand_to_unity(candidates[1], w, h),
                _hand_to_unity(candidates[0], w, h))

    def close(self) -> None:
        self._landmarker.close()

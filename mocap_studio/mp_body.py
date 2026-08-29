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
}

_CONV = np.array([-1.0, -1.0, -1.0])


def _to_unity(landmarks) -> np.ndarray:
    return np.array([[p.x, p.y, p.z] for p in landmarks]) * _CONV


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
            min_hand_landmarks_confidence=0.4,
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
        # Downscale for speed: landmark output is normalized/world-space,
        # so tracking quality barely changes but CPU cost drops a lot.
        h, w = frame_bgr.shape[:2]
        if w > 640:
            scale = 640.0 / w
            frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)),
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

        left, right = self._assign_hands(res)
        return pose, left, right

    @staticmethod
    def _assign_hands(res):
        """Assign detected hands to anatomical sides via pose wrists
        (normalized image coords). Falls back to swapping the task's
        mirrored labels when no pose is available."""
        candidates = []  # (normalized landmarks, world landmarks)
        if res.left_hand_landmarks:
            candidates.append((res.left_hand_landmarks,
                               res.left_hand_world_landmarks))
        if res.right_hand_landmarks:
            candidates.append((res.right_hand_landmarks,
                               res.right_hand_world_landmarks))
        if not candidates:
            return None, None

        if not res.pose_landmarks:
            # labels are mirrored for a non-selfie feed -> swap them
            left = _to_unity(res.right_hand_world_landmarks) \
                if res.right_hand_world_landmarks else None
            right = _to_unity(res.left_hand_world_landmarks) \
                if res.left_hand_world_landmarks else None
            return left, right

        lw = res.pose_landmarks[15]  # left_wrist (normalized)
        rw = res.pose_landmarks[16]  # right_wrist

        def dists(norm):
            w0 = norm[0]  # hand wrist landmark
            dl = (w0.x - lw.x) ** 2 + (w0.y - lw.y) ** 2
            dr = (w0.x - rw.x) ** 2 + (w0.y - rw.y) ** 2
            return dl, dr

        if len(candidates) == 1:
            norm, world = candidates[0]
            dl, dr = dists(norm)
            hand = _to_unity(world)
            return (hand, None) if dl <= dr else (None, hand)

        d0 = dists(candidates[0][0])
        d1 = dists(candidates[1][0])
        if d0[0] + d1[1] <= d0[1] + d1[0]:
            return (_to_unity(candidates[0][1]),
                    _to_unity(candidates[1][1]))
        return (_to_unity(candidates[1][1]),
                _to_unity(candidates[0][1]))

    def close(self) -> None:
        self._landmarker.close()

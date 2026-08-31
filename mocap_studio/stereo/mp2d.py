"""Per-camera MediaPipe observer for the stereo extension.

Each camera gets its *own* HolisticLandmarker instance in VIDEO mode
(the design doc's requirement: never feed alternating cameras into one
tracking instance).  Unlike ``mp_body.MediaPipeBodyTracker`` - which
returns Unity-space world points for the legacy path - this wrapper
returns *structured 2D observations* (normalized image coordinates per
joint) plus the raw world landmarks, so the fusion stage can pair the
same joint across cameras and triangulate.

Hand side assignment reuses the same pose-wrist-proximity logic as the
legacy tracker so behaviour matches what the user already knows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..mp_body import MODEL_PATH, MP_POSE, _MAX_WRIST_DIST

N_POSE = 33
N_HAND = 21


@dataclass
class Observation:
    """One camera's landmarks for one frame (source-timestamped)."""

    ts: float                                  # receive-time estimate [s]
    seq: int
    img_size: tuple[int, int]                  # (w, h) of the full frame
    pose_norm: np.ndarray | None = None        # (33, 2) normalized xy
    pose_vis: np.ndarray | None = None         # (33,) visibility 0..1
    pose_world: np.ndarray | None = None       # (33, 3) MP world coords
    lhand_norm: np.ndarray | None = None       # (21, 3) normalized xyz
    rhand_norm: np.ndarray | None = None
    debug2d: list = field(default_factory=list)  # (x, y, kind) overlay

    def pose_uv(self) -> np.ndarray | None:
        """Pose joints in full-frame pixels, (33, 2)."""
        if self.pose_norm is None:
            return None
        w, h = self.img_size
        return self.pose_norm * np.array([w, h], dtype=np.float64)


class CameraObserver:
    """HolisticLandmarker (VIDEO mode) -> Observation."""

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as tp
        from mediapipe.tasks.python import vision
        self._mp = mp
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"MediaPipeモデルがありません: {MODEL_PATH}")
        model_bytes = MODEL_PATH.read_bytes()
        opts = vision.HolisticLandmarkerOptions(
            base_options=tp.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.3,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.6,
        )
        self._landmarker = vision.HolisticLandmarker.create_from_options(opts)
        self._last_ts_ms = 0
        self._seq = 0

    def process(self, frame_bgr: np.ndarray, ts: float) -> Observation:
        h, w = frame_bgr.shape[:2]
        full_size = (w, h)
        # Same moderate downscale as the legacy backend (landmarks are
        # normalized, so pixel mapping to the full frame is unaffected).
        if w > 960:
            scale = 960.0 / w
            frame_bgr = cv2.resize(frame_bgr, (960, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(ts * 1000.0)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        res = self._landmarker.detect_for_video(img, ts_ms)

        self._seq += 1
        obs = Observation(ts=ts, seq=self._seq, img_size=full_size)

        if res.pose_landmarks:
            lm = res.pose_landmarks
            obs.pose_norm = np.array([[p.x, p.y] for p in lm])
            obs.pose_vis = np.array([float(p.visibility) for p in lm])
            obs.debug2d += [(p.x, p.y, 0 if p.visibility >= 0.5 else 2)
                            for p in lm]
        if res.pose_world_landmarks:
            obs.pose_world = np.array(
                [[p.x, p.y, p.z] for p in res.pose_world_landmarks])
            # World landmarks carry the visibility used by the legacy
            # path; keep the image-landmark one if both exist.
            if obs.pose_vis is None:
                obs.pose_vis = np.array(
                    [float(p.visibility) for p in res.pose_world_landmarks])

        left, right = self._assign_hands(res)
        obs.lhand_norm, obs.rhand_norm = left, right
        for hand in (left, right):
            if hand is not None:
                obs.debug2d += [(p[0], p[1], 1) for p in hand]
        return obs

    # -- hand side assignment (mirrors mp_body logic) ------------------
    @staticmethod
    def _assign_hands(res):
        def arr(lm):
            return np.array([[p.x, p.y, p.z] for p in lm])

        candidates = [lm for lm in (res.left_hand_landmarks,
                                    res.right_hand_landmarks) if lm]
        if not candidates:
            return None, None
        if not res.pose_landmarks:
            # task labels follow the mirrored-selfie convention -> swap
            left = arr(res.right_hand_landmarks) \
                if res.right_hand_landmarks else None
            right = arr(res.left_hand_landmarks) \
                if res.left_hand_landmarks else None
            return left, right

        lw = res.pose_landmarks[MP_POSE["left_wrist"]]
        rw = res.pose_landmarks[MP_POSE["right_wrist"]]

        def dists(lm):
            w0 = lm[0]
            dl = (w0.x - lw.x) ** 2 + (w0.y - lw.y) ** 2
            dr = (w0.x - rw.x) ** 2 + (w0.y - rw.y) ** 2
            return dl, dr

        candidates = [c for c in candidates
                      if min(dists(c)) <= _MAX_WRIST_DIST ** 2]
        if not candidates:
            return None, None
        if len(candidates) == 1:
            dl, dr = dists(candidates[0])
            hand = arr(candidates[0])
            return (hand, None) if dl <= dr else (None, hand)
        d0, d1 = dists(candidates[0]), dists(candidates[1])
        if d0[0] + d1[1] <= d0[1] + d1[0]:
            return arr(candidates[0]), arr(candidates[1])
        return arr(candidates[1]), arr(candidates[0])

    def close(self) -> None:
        self._landmarker.close()

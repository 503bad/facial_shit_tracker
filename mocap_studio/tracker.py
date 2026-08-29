"""Tracking worker: camera -> estimators -> pipelines -> UDP senders.

Runs on a plain Python thread; the GUI polls `TrackerWorker.status` (lock
protected) for preview frames and stats.
"""

from __future__ import annotations

import threading
import time
import traceback

import cv2
import numpy as np

from .body_pipeline import BodyRetargeter
from .camera import Camera
from .config import Settings
from .face_pipeline import FacePipeline
from .ifm_sender import IfmSender
from .vmc_sender import VmcSender

# NVIDIA keypoint -> Unity conversion (X-right/Y-up/Z-toward-camera ->
# avatar space): (-x, y, z), plus mm -> m if magnitudes suggest mm.
_NV_CONV = np.array([-1.0, 1.0, 1.0])


def _mirror_body(pose, lhand, rhand):
    """Mirror tracked points: flip X and swap anatomical sides."""

    def flip(p):
        q = np.array(p, dtype=np.float64, copy=True)
        q[..., 0] *= -1.0
        return q

    new_pose = None
    if pose is not None:
        new_pose = {}
        for k, v in pose.items():
            nk = k.replace("left_", "@").replace(
                "right_", "left_").replace("@", "right_")
            new_pose[nk] = flip(v)
    new_l = flip(rhand) if rhand is not None else None
    new_r = flip(lhand) if lhand is not None else None
    return new_pose, new_l, new_r


class TrackerStatus:
    def __init__(self) -> None:
        self.running = False
        self.fps = 0.0
        self.face_found = False
        self.body_found = False
        self.hands_found = (False, False)
        self.error: str | None = None
        self.info: str = ""
        self.preview: np.ndarray | None = None  # BGR with overlay


class TrackerWorker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.status = TrackerStatus()
        self._status_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.face_pipeline = FacePipeline(settings.face_smoothing,
                                          settings.head_smoothing)
        self.body_retargeter = BodyRetargeter(settings.body_smoothing,
                                              settings.finger_smoothing)
        self.body_backend = "mediapipe"  # or "nvidia"
        self._preview_enabled = True

    # -- GUI-facing controls (thread-safe by value assignment) ----------
    def set_preview_enabled(self, on: bool) -> None:
        self._preview_enabled = on

    def request_face_calibration(self) -> None:
        self.face_pipeline.request_calibration()

    def request_body_calibration(self) -> None:
        self.body_retargeter.start_calibration()

    def apply_smoothing(self) -> None:
        s = self.settings
        self.face_pipeline.set_strengths(s.face_smoothing, s.head_smoothing)
        self.body_retargeter.set_strengths(s.body_smoothing,
                                           s.finger_smoothing)

    def get_status(self) -> TrackerStatus:
        with self._status_lock:
            st = TrackerStatus()
            st.__dict__.update(self.status.__dict__)
            return st

    def _set_status(self, **kw) -> None:
        with self._status_lock:
            self.status.__dict__.update(kw)

    # -------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    # -------------------------------------------------------------------
    def _run(self) -> None:
        s = self.settings
        camera = None
        face = None
        body_nv = None
        body_mp = None
        ifm = None
        vmc = None
        try:
            self._set_status(running=True, error=None,
                             info="カメラを起動中...")
            camera = Camera(s.camera_index, s.camera_width,
                            s.camera_height, s.camera_fps)
            if not camera.start():
                raise RuntimeError(camera.last_error or "カメラ起動失敗")
            # wait for first frame to know the actual size
            for _ in range(100):
                frame, fid = camera.latest()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                raise RuntimeError("カメラからフレームが届きません")
            h, w = frame.shape[:2]

            if s.face_enabled:
                self._set_status(info="NVIDIA FaceExpressions を初期化中"
                                      "（初回は数十秒かかります）...")
                from .nvar import FaceExpressionEstimator
                face = FaceExpressionEstimator(w, h)
                ifm = IfmSender(s.face_host, s.face_port)

            if s.body_enabled:
                if self.body_backend == "nvidia":
                    self._set_status(info="NVIDIA BodyPose を初期化中...")
                    from .nvar import BodyPoseEstimator
                    body_nv = BodyPoseEstimator(
                        w, h, high_quality=s.body_mode_high_quality)
                else:
                    self._set_status(info="MediaPipe Holistic を初期化中...")
                    from .mp_body import MediaPipeBodyTracker
                    body_mp = MediaPipeBodyTracker()
                vmc = VmcSender(s.vmc_host, s.vmc_port)

            self._set_status(info="トラッキング中")
            last_fid = -1
            t_start = time.perf_counter()
            fps_n, fps_t = 0, time.perf_counter()
            last_face_send = 0.0
            last_body_send = 0.0

            while not self._stop.is_set():
                frame, fid = camera.latest()
                if frame is None or fid == last_fid:
                    time.sleep(0.002)
                    continue
                last_fid = fid
                now = time.perf_counter()
                t = now - t_start
                overlay = frame.copy() if self._preview_enabled else None

                face_found = False
                if face is not None and ifm is not None:
                    ifm.set_destination(s.face_host, s.face_port)
                    self.face_pipeline.mirror = s.mirror_tracking
                    found, expr, pose_q, trans, lm = face.process(frame)
                    face_found = bool(found)
                    if found and now - last_face_send >= 1.0 / max(
                            1, s.face_send_rate):
                        last_face_send = now
                        bs, head_e, head_p, reye, leye = \
                            self.face_pipeline.process(expr, pose_q, trans, t)
                        ifm.send(bs, head_e, head_p, reye, leye)
                    if overlay is not None and found:
                        for x, y in lm:
                            cv2.circle(overlay, (int(x), int(y)), 1,
                                       (0, 255, 128), -1)

                body_found = False
                l_found = r_found = False
                if vmc is not None:
                    vmc.set_destination(s.vmc_host, s.vmc_port)
                    pose_pts = None
                    lhand = rhand = None
                    if body_mp is not None:
                        pose_pts, lhand, rhand = body_mp.process(
                            frame, int(t * 1000))
                        if overlay is not None:
                            pass  # world landmarks aren't in pixel space
                    elif body_nv is not None:
                        kp3d, kp2d, conf = body_nv.process(frame)
                        if float(np.mean(conf)) > 0.2:
                            pose_pts = self._nv_pose_points(kp3d)
                            self._nv_grip(kp3d, s.mirror_tracking)
                        if overlay is not None:
                            for (x, y), c in zip(kp2d, conf):
                                if c > 0.3:
                                    cv2.circle(overlay, (int(x), int(y)), 3,
                                               (0, 128, 255), -1)
                    if s.mirror_tracking:
                        pose_pts, lhand, rhand = _mirror_body(
                            pose_pts, lhand, rhand)
                    body_found = pose_pts is not None
                    l_found, r_found = lhand is not None, rhand is not None
                    if body_found and now - last_body_send >= 1.0 / max(
                            1, s.body_send_rate):
                        last_body_send = now
                        bones = self.body_retargeter.process(
                            pose_pts, lhand, rhand, t)
                        vmc.send_frame(bones)

                fps_n += 1
                if now - fps_t >= 1.0:
                    self._set_status(fps=fps_n / (now - fps_t))
                    fps_n, fps_t = 0, now
                self._set_status(face_found=face_found,
                                 body_found=body_found,
                                 hands_found=(l_found, r_found),
                                 preview=overlay)
        except Exception:
            self._set_status(error=traceback.format_exc(),
                             info="エラーで停止しました")
        finally:
            for obj, closer in ((face, "destroy"), (body_nv, "destroy"),
                                (body_mp, "close"), (ifm, "close")):
                if obj is not None:
                    try:
                        getattr(obj, closer)()
                    except Exception:
                        pass
            if camera is not None:
                camera.stop()
            self._set_status(running=False)

    # -------------------------------------------------------------------
    def _nv_pose_points(self, kp3d: np.ndarray) -> dict[str, np.ndarray]:
        from .nvar import KP
        scale = 0.001 if np.abs(kp3d).max() > 10 else 1.0
        pts = kp3d * scale * _NV_CONV
        names = ["left_shoulder", "right_shoulder", "left_elbow",
                 "right_elbow", "left_wrist", "right_wrist",
                 "left_hip", "right_hip", "nose", "left_ear", "right_ear"]
        return {n: pts[KP[n]] for n in names}

    def _nv_grip(self, kp3d: np.ndarray, mirror: bool = False) -> None:
        """Estimate hand open/close from sparse NVIDIA hand points."""
        from .nvar import KP
        for side in ("left", "right"):
            out_side = ({"left": "Right", "right": "Left"}[side]
                        if mirror else side.capitalize())
            wrist = kp3d[KP[f"{side}_wrist"]]
            tip = kp3d[KP[f"{side}_middle_tip"]]
            elbow = kp3d[KP[f"{side}_elbow"]]
            forearm = np.linalg.norm(wrist - elbow)
            if forearm < 1e-6:
                continue
            ratio = np.linalg.norm(tip - wrist) / forearm
            # open hand: tip far (~0.75 forearm), fist: close (~0.35)
            curl = float(np.clip((0.72 - ratio) / 0.35, 0.0, 1.0))
            self.body_retargeter.set_grip(out_side, curl)

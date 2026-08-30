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
from .ifm_sender import IfmSender, ifm_to_arkit
from .interp import OutputInterpolator
from .vmc_sender import VmcSender
from .retarget import quat_from_axis_angle, quat_mul


def _eye_quat(pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Unity eye-bone local rotation: X = look down (+), Y = look right (+),
    composed in Unity's Euler order (q = Qy * Qx)."""
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), np.deg2rad(pitch_deg))
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.deg2rad(yaw_deg))
    return quat_mul(qy, qx)

# NVIDIA keypoint -> Unity conversion (X-right/Y-up/Z-toward-camera ->
# avatar space): (-x, y, z), plus mm -> m if magnitudes suggest mm.
_NV_CONV = np.array([-1.0, 1.0, 1.0])

# Set to a path to log raw head-pose data for debugging; None disables.
import os as _os
_DEBUG_HEAD_LOG = _os.environ.get("MOCAP_HEAD_LOG") or None


class _AsyncBody:
    """Runs the (CPU-heavy) body backend on its own thread with
    latest-frame-wins semantics so the face pipeline keeps full FPS."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self._frame = None
        self._ts = 0
        self._result = None
        self._seq = 0
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame, ts_ms: int) -> None:
        with self._lock:
            self._frame = frame
            self._ts = ts_ms
        self._event.set()

    def latest(self):
        with self._lock:
            return self._result, self._seq

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._event.wait(0.2):
                continue
            self._event.clear()
            with self._lock:
                frame, ts = self._frame, self._ts
            if frame is None:
                continue
            try:
                res = self._fn(frame, ts)
            except Exception as e:  # surfaced by the main loop
                self.error = e
                break
            with self._lock:
                self._result = res
                self._seq += 1

    def stop(self) -> None:
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=3.0)


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
        self._show_camera = settings.show_camera
        self.body_retargeter.gate_enabled = settings.body_gate_enabled
        self.body_retargeter.gate_rad = np.deg2rad(settings.body_gate_deg)

    # -- GUI-facing controls (thread-safe by value assignment) ----------
    def set_preview_enabled(self, on: bool) -> None:
        self._preview_enabled = on

    def set_show_camera(self, on: bool) -> None:
        self._show_camera = on

    def set_gate_enabled(self, on: bool) -> None:
        self.body_retargeter.gate_enabled = on

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
        body_async = None
        ifm = None
        vmc = None
        interp = None
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
                if s.face_output in ("ifm", "both"):
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
            if vmc is None and face is not None and s.face_output != "ifm":
                # Perfect Sync over VMC without body tracking
                vmc = VmcSender(s.vmc_host, s.vmc_port)

            if s.output_interp:
                # Output-stage resampler: senders are called from its own
                # thread at output_fps; the tracking loop only pushes samples.
                sinks = {}
                if ifm is not None:
                    sinks["ifm"] = ifm.send
                if vmc is not None:
                    sinks["bones"] = vmc.send_frame
                    sinks["blend"] = vmc.send_blendshapes
                interp = OutputInterpolator(s.output_fps, sinks)

            self._set_status(info="トラッキング中")
            body_async = None
            if body_mp is not None:
                body_async = _AsyncBody(body_mp.process)
            last_fid = -1
            t_start = time.perf_counter()
            fps_n, fps_t = 0, time.perf_counter()
            last_face_send = 0.0
            last_body_send = 0.0
            last_body_seq = -1
            last_debug2d = None
            body_found = False
            l_found = r_found = False

            while not self._stop.is_set():
                frame, fid = camera.latest()
                if frame is None or fid == last_fid:
                    time.sleep(0.002)
                    continue
                last_fid = fid
                now = time.perf_counter()
                t = now - t_start
                overlay = None
                if self._preview_enabled:
                    overlay = frame.copy() if self._show_camera \
                        else np.zeros_like(frame)

                face_found = False
                if face is not None:
                    if ifm is not None:
                        ifm.set_destination(s.face_host, s.face_port)
                    self.face_pipeline.mirror = s.mirror_tracking
                    found, expr, pose_q, trans, lm = face.process(frame)
                    face_found = bool(found)
                    if found and now - last_face_send >= 1.0 / max(
                            1, s.face_send_rate):
                        last_face_send = now
                        bs, head_e, head_p, reye, leye = \
                            self.face_pipeline.process(expr, pose_q, trans, t)
                        if ifm is not None:
                            if interp is not None:
                                interp.push("ifm", now,
                                            (bs, head_e, head_p, reye, leye))
                            else:
                                ifm.send(bs, head_e, head_p, reye, leye)
                        if vmc is not None and s.face_output != "ifm":
                            vmc.set_destination(s.vmc_host, s.vmc_port)
                            vals = {ifm_to_arkit(k): v for k, v in bs.items()}
                            if s.eye_mode == "bone":
                                for k in vals:
                                    if k.startswith("eyeLook"):
                                        vals[k] = 0.0
                            if interp is not None:
                                interp.push("blend", now, vals)
                            else:
                                vmc.send_blendshapes(vals)
                            if s.eye_mode in ("bone", "both"):
                                self.body_retargeter.set_eye_rotations(
                                    _eye_quat(leye[0], leye[1]),
                                    _eye_quat(reye[0], reye[1]))
                            else:
                                self.body_retargeter.set_eye_rotations(
                                    None, None)
                            if body_async is None and body_nv is None:
                                # no body frames: send the head chain alone
                                fb = self.body_retargeter.face_bones_frame()
                                if interp is not None:
                                    interp.push("bones", now, fb)
                                else:
                                    vmc.send_frame(fb)
                        # Also drive Neck/Head over VMC (receivers whose
                        # body tracking owns the skeleton ignore iFM head).
                        self.body_retargeter.set_head_pose(
                            self.face_pipeline.latest_head_quat)
                        if _DEBUG_HEAD_LOG and int(t * 10) % 5 == 0:
                            with open(_DEBUG_HEAD_LOG, "a",
                                      encoding="utf-8") as f:
                                f.write(
                                    f"t={t:6.1f} raw_q="
                                    f"({pose_q[0]:+.3f},{pose_q[1]:+.3f},"
                                    f"{pose_q[2]:+.3f},{pose_q[3]:+.3f}) "
                                    f"trans=({trans[0]:+.3f},{trans[1]:+.3f},"
                                    f"{trans[2]:+.3f}) euler_deg="
                                    f"({head_e[0]:+6.1f},{head_e[1]:+6.1f},"
                                    f"{head_e[2]:+6.1f})\n")
                    if overlay is not None and found:
                        for x, y in lm:
                            cv2.circle(overlay, (int(x), int(y)), 1,
                                       (0, 255, 128), -1)

                if vmc is not None:
                    vmc.set_destination(s.vmc_host, s.vmc_port)
                    new_body = None
                    if body_async is not None:
                        if body_async.error is not None:
                            raise body_async.error
                        body_async.submit(frame, int(t * 1000))
                        res, seq = body_async.latest()
                        if res is not None and seq != last_body_seq:
                            last_body_seq = seq
                            pose_pts, lhand, rhand, last_debug2d = res
                            new_body = (pose_pts, lhand, rhand)
                        if overlay is not None and last_debug2d:
                            oh, ow = overlay.shape[:2]
                            for x, y, kind in last_debug2d:
                                color = (0, 128, 255) if kind == 0 \
                                    else (255, 200, 0)
                                cv2.circle(overlay,
                                           (int(x * ow), int(y * oh)),
                                           2, color, -1)
                    elif body_nv is not None:
                        kp3d, kp2d, conf = body_nv.process(frame)
                        pose_pts = None
                        if float(np.mean(conf)) > 0.2:
                            pose_pts = self._nv_pose_points(kp3d)
                            self._nv_grip(kp3d, s.mirror_tracking)
                        new_body = (pose_pts, None, None)
                        if overlay is not None:
                            for (x, y), c in zip(kp2d, conf):
                                if c > 0.3:
                                    cv2.circle(overlay, (int(x), int(y)), 3,
                                               (0, 128, 255), -1)
                    if new_body is not None:
                        pose_pts, lhand, rhand = new_body
                        if s.mirror_tracking:
                            pose_pts, lhand, rhand = _mirror_body(
                                pose_pts, lhand, rhand)
                        body_found = pose_pts is not None
                        l_found = lhand is not None
                        r_found = rhand is not None
                        if body_found and now - last_body_send >= 1.0 / max(
                                1, s.body_send_rate):
                            last_body_send = now
                            bones = self.body_retargeter.process(
                                pose_pts, lhand, rhand, t)
                            if interp is not None:
                                interp.push("bones", now, bones)
                            else:
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
            if interp is not None:
                interp.stop()
            if body_async is not None:
                body_async.stop()
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

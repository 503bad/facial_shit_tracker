"""Stereo engine: camera-B capture, dual inference, pairing, fusion,
and the result adapter back to the legacy body-pipeline contract.

Ownership / isolation rules (per the design doc):

  * Camera A frames are *submitted* by the tracker loop (read-only
    shares); the engine never opens or closes camera A.
  * Camera B, both HolisticLandmarker instances, worker threads, queues
    and all fusion state are owned by the engine and released on
    ``stop()``.
  * ``latest()`` returns results in exactly the tuple shape the legacy
    ``_AsyncBody`` produces - ``(pose_pts, lhand, rhand, debug2d)`` - so
    the downstream retargeter/smoothing/interpolation are reused without
    modification.  Stereo refinements are *blended into* the legacy-style
    mono observation; the worst case (no stereo pairs, tracks lost)
    degrades to exactly what the mono path would output.
  * Failures set ``self.error``; the tracker reverts to the legacy path.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque

import cv2
import numpy as np

from ..mp_body import MP_POSE, _VIS_NAMES
from .capture import TimedCamera
from .config import LOG_DIR, StereoSettings
from .fusion import FusionParams, StereoFusion, DISABLED, LOST


def _hand_to_unity(norm_lm: np.ndarray, w: int, h: int) -> np.ndarray:
    """Normalized (21,3) hand landmarks -> legacy Unity-oriented points
    (same formula as mp_body._hand_to_unity)."""
    out = np.empty_like(norm_lm)
    out[:, 0] = -norm_lm[:, 0] * w
    out[:, 1] = -norm_lm[:, 1] * h
    out[:, 2] = -norm_lm[:, 2] * w
    return out


class StereoEngine:
    def __init__(self, s2: StereoSettings, calib, camera_a_index: int,
                 camera_a_size: tuple[int, int]) -> None:
        self.s2 = s2
        self._cam_a_index = camera_a_index
        self._cam_a_size = tuple(camera_a_size)
        self.calib = calib.scaled_to(camera_a_size,
                                     (s2.camera_b_width, s2.camera_b_height))
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()          # result / stats / pip
        self._in_lock = threading.Lock()       # submitted A frame
        self._in_event = threading.Event()
        self._frame_a = None
        self._ts_a = 0.0
        self._result = None
        self._seq = 0
        self._pip: np.ndarray | None = None
        self._pip_t = 0.0
        self._b_lock = threading.Lock()
        self._b_results: deque = deque(maxlen=8)   # (ts, seq, Observation)
        self._b_used_seq = -1
        self._cam_b: TimedCamera | None = None
        self._obs_a = None
        self._obs_b = None
        self._th_a: threading.Thread | None = None
        self._th_b: threading.Thread | None = None
        self._fusion: StereoFusion | None = None
        self._legs_enabled = False
        self._log_fh = None
        self._log_lines = 0
        # stats
        self._stats = {"fps_a": 0.0, "fps_b": 0.0, "pair_pct": 0.0,
                       "skew_ms": 0.0, "n_stereo": 0, "n_mono": 0,
                       "hips_z": None, "states": {}}
        self._cnt_a = 0
        self._cnt_b = 0
        self._cnt_pair = 0
        self._skews: deque = deque(maxlen=60)
        self._stat_t = time.perf_counter()

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Open camera B and both landmarkers.  Raises on failure (the
        caller falls back to the legacy path; the app keeps running)."""
        s2 = self.s2
        if s2.camera_b_index == self._cam_a_index:
            raise RuntimeError(
                "カメラBにカメラAと同じデバイスが選択されています。"
                "2カメラ追跡には別のカメラを選択してください。")
        cam = TimedCamera(s2.camera_b_index, s2.camera_b_width,
                          s2.camera_b_height, s2.camera_b_fps)
        if not cam.start():
            raise RuntimeError(
                f"カメラB（デバイス {s2.camera_b_index}）を開けませんでした。")
        self._cam_b = cam
        # Wait for the first frame to learn the real capture size.
        frame_b = None
        for _ in range(100):
            frame_b, _, _ = cam.latest()
            if frame_b is not None:
                break
            time.sleep(0.05)
        if frame_b is None:
            cam.stop()
            self._cam_b = None
            raise RuntimeError("カメラBからフレームが届きません。")
        bh, bw = frame_b.shape[:2]
        self.calib = self.calib.scaled_to(self._cam_a_size, (bw, bh))

        try:
            from .mp2d import CameraObserver
            self._obs_a = CameraObserver()
            self._obs_b = CameraObserver()
        except Exception:
            cam.stop()
            self._cam_b = None
            if self._obs_a is not None:
                try:
                    self._obs_a.close()
                except Exception:
                    pass
                self._obs_a = None
            raise

        params = FusionParams.from_settings(s2)
        self._fusion = StereoFusion(self.calib, params, dict(MP_POSE))
        self._fusion.legs_enabled = self._legs_enabled

        if s2.debug_log:
            self._open_log()

        self._stop.clear()
        self._th_b = threading.Thread(target=self._loop_b, daemon=True)
        self._th_a = threading.Thread(target=self._loop_a, daemon=True)
        self._th_b.start()
        self._th_a.start()

    def stop(self) -> None:
        self._stop.set()
        self._in_event.set()
        for th in (self._th_a, self._th_b):
            if th is not None:
                th.join(timeout=3.0)
        self._th_a = self._th_b = None
        for obs in (self._obs_a, self._obs_b):
            if obs is not None:
                try:
                    obs.close()
                except Exception:
                    pass
        self._obs_a = self._obs_b = None
        if self._cam_b is not None:
            self._cam_b.stop()
            self._cam_b = None
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    # ------------------------------------------------------------------
    def set_legs_enabled(self, on: bool) -> None:
        self._legs_enabled = bool(on)
        if self._fusion is not None:
            self._fusion.legs_enabled = bool(on)

    def submit_a(self, frame_bgr: np.ndarray, ts: float) -> None:
        """Latest-wins submission from the tracker loop (read-only)."""
        with self._in_lock:
            self._frame_a = frame_bgr
            self._ts_a = ts
        self._in_event.set()

    def latest(self):
        """((pose_pts, lhand, rhand, debug2d), seq) - legacy shape."""
        with self._lock:
            return self._result, self._seq

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_pip(self) -> np.ndarray | None:
        with self._lock:
            return self._pip

    # -- camera-B worker ----------------------------------------------
    def _loop_b(self) -> None:
        last_fid = -1
        seq = 0
        try:
            while not self._stop.is_set():
                frame, fid, ts = self._cam_b.latest()
                if frame is None or fid == last_fid:
                    time.sleep(0.002)
                    continue
                last_fid = fid
                obs = self._obs_b.process(frame, ts)
                seq += 1
                self._cnt_b += 1
                with self._b_lock:
                    self._b_results.append((ts, seq, obs))
                if (self.s2.show_pip
                        and ts - self._pip_t > 0.15):
                    self._make_pip(frame, obs)
                    self._pip_t = ts
        except Exception as e:
            self.error = e

    def _make_pip(self, frame: np.ndarray, obs) -> None:
        h, w = frame.shape[:2]
        pw = 320
        ph = int(h * pw / w)
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        for x, y, kind in obs.debug2d:
            color = {0: (0, 128, 255), 1: (255, 200, 0),
                     2: (0, 0, 255)}.get(kind, (0, 0, 255))
            cv2.circle(small, (int(x * pw), int(y * ph)), 1, color, -1)
        cv2.putText(small, "CAM B", (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        with self._lock:
            self._pip = small

    # -- camera-A worker + fusion -------------------------------------
    def _loop_a(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._in_event.wait(0.2):
                    continue
                self._in_event.clear()
                with self._in_lock:
                    frame, ts = self._frame_a, self._ts_a
                if frame is None:
                    continue
                obs_a = self._obs_a.process(frame, ts)
                self._cnt_a += 1
                tol = self.s2.pair_tolerance_ms / 1000.0  # live-adjustable
                obs_b, skew = self._match_b(ts, tol)
                self._fuse_and_publish(obs_a, obs_b, skew)
                self._update_stats()
        except Exception as e:
            self.error = e

    def _match_b(self, ts_a: float, tol: float):
        """Find the camera-B observation closest in source time to ts_a.

        Waits briefly for B inference to catch up (results arrive after
        their frame's timestamp); never blocks past ~55 ms; never reuses
        a B observation for two A frames; never pairs beyond tolerance.
        """
        deadline = time.perf_counter() + 0.055
        best = None
        while True:
            with self._b_lock:
                for bts, bseq, bobs in self._b_results:
                    if bseq <= self._b_used_seq:
                        continue
                    d = abs(bts - ts_a)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, bseq, bobs)
                newest_ts = self._b_results[-1][0] if self._b_results \
                    else -1e9
            if best is None:
                if newest_ts > ts_a + tol:
                    return None, None      # B already past A: no match
                if newest_ts < ts_a - 0.3:
                    # camera B stalled / far behind: do not block the A
                    # pipeline waiting for it - go mono immediately.
                    return None, None
            if best is not None and (newest_ts > ts_a or
                                     time.perf_counter() >= deadline):
                # A later B frame exists (no closer one is coming) or we
                # are out of time: commit the best match.
                with self._b_lock:
                    self._b_used_seq = best[1]
                self._cnt_pair += 1
                self._skews.append(abs(best[0]) * 1000.0)
                return best[2], best[0] * 1000.0
            if time.perf_counter() >= deadline:
                return None, None
            if self._stop.is_set():
                return None, None
            time.sleep(0.003)

    # -- fusion + adaptation ------------------------------------------
    def _fuse_and_publish(self, obs_a, obs_b, skew_ms) -> None:
        pose_pts = None
        lhand = rhand = None
        hips_z = None
        fused = None
        if obs_a.pose_world is not None and obs_a.pose_norm is not None:
            if obs_a.pose_vis is not None:
                fused = self._fusion.step(obs_a.ts, obs_a,
                                          obs_b if obs_b is not None
                                          and obs_b.pose_norm is not None
                                          else None, skew_ms)
            pose_pts, hips_z = self._adapt_pose(obs_a, fused)
        w, h = obs_a.img_size
        if obs_a.lhand_norm is not None:
            lhand = _hand_to_unity(obs_a.lhand_norm, w, h)
        if obs_a.rhand_norm is not None:
            rhand = _hand_to_unity(obs_a.rhand_norm, w, h)
        result = (pose_pts, lhand, rhand, list(obs_a.debug2d))
        self._log_step(obs_a, obs_b, fused, skew_ms)
        with self._lock:
            self._result = result
            self._seq += 1
            if fused is not None:
                self._stats["n_stereo"] = fused.n_stereo
                self._stats["n_mono"] = fused.n_mono
                self._stats["states"] = dict(fused.states)
            self._stats["hips_z"] = hips_z

    def _adapt_pose(self, obs_a, fused):
        """Build the legacy pose dict (mono base) and blend in the fused
        stereo positions where their tracks are valid."""
        world = obs_a.pose_world
        unity = world * -1.0                       # mp_body._CONV
        pose = {name: unity[i].copy() for name, i in MP_POSE.items()}
        vis = obs_a.pose_vis
        pose["_vis"] = {name: float(vis[MP_POSE[name]])
                        for name in _VIS_NAMES} if vis is not None else {}
        w, h = obs_a.img_size
        norm = obs_a.pose_norm
        lhip, rhip = MP_POSE["left_hip"], MP_POSE["right_hip"]
        lsh, rsh = MP_POSE["left_shoulder"], MP_POSE["right_shoulder"]
        pose["_img"] = {
            "hip_center": (float((norm[lhip, 0] + norm[rhip, 0]) / 2),
                           float((norm[lhip, 1] + norm[rhip, 1]) / 2)),
            "shoulder_center": (float((norm[lsh, 0] + norm[rsh, 0]) / 2),
                                float((norm[lsh, 1] + norm[rsh, 1]) / 2)),
            "aspect": w / h,
        }

        hips_z = None
        if fused is not None:
            wl = fused.weights.get("left_hip", 0.0)
            wr = fused.weights.get("right_hip", 0.0)
            if wl > 0.0 and wr > 0.0:
                hipc = (fused.positions["left_hip"]
                        + fused.positions["right_hip"]) / 2.0
                base_w = float(np.clip(self.s2.stereo_weight, 0.0, 1.0))
                if base_w > 0.0:
                    for name in MP_POSE:
                        wj = fused.weights.get(name, 0.0) * base_w
                        if wj <= 0.0:
                            continue
                        st_unity = -(fused.positions[name] - hipc)
                        pose[name] = (pose[name] * (1.0 - wj)
                                      + st_unity * wj)
                if (self.s2.send_depth_translation
                        and min(wl, wr) >= 0.5):
                    hips_z = float(hipc[2])
                    pose["_stereo_hips_z"] = hips_z
        return pose, hips_z

    # -- stats / logging ----------------------------------------------
    def _update_stats(self) -> None:
        now = time.perf_counter()
        dt = now - self._stat_t
        if dt < 1.0:
            return
        with self._lock:
            self._stats["fps_a"] = self._cnt_a / dt
            self._stats["fps_b"] = self._cnt_b / dt
            self._stats["pair_pct"] = (100.0 * self._cnt_pair
                                       / max(self._cnt_a, 1))
            self._stats["skew_ms"] = (float(np.median(self._skews))
                                      if self._skews else 0.0)
        self._cnt_a = self._cnt_b = self._cnt_pair = 0
        self._stat_t = now

    def _open_log(self) -> None:
        try:
            LOG_DIR.mkdir(exist_ok=True)
            path = LOG_DIR / time.strftime("stereo_%Y%m%d_%H%M%S.jsonl")
            self._log_fh = open(path, "w", encoding="utf-8")
            header = {"type": "header",
                      "calib": self.calib.to_dict(),
                      "settings": {
                          "pair_tolerance_ms": self.s2.pair_tolerance_ms,
                          "min_visibility": self.s2.min_visibility,
                          "max_epipolar_px": self.s2.max_epipolar_px,
                          "recovery_frames": self.s2.recovery_frames,
                          "predict_timeout_ms": self.s2.predict_timeout_ms,
                          "recover_blend_ms": self.s2.recover_blend_ms}}
            self._log_fh.write(json.dumps(header) + "\n")
        except Exception:
            self._log_fh = None

    def _log_step(self, obs_a, obs_b, fused, skew_ms) -> None:
        if self._log_fh is None or self._log_lines >= 400_000:
            return
        try:
            uv_a = obs_a.pose_uv()
            uv_b = obs_b.pose_uv() if obs_b is not None else None
            joints = {}
            for name, idx in MP_POSE.items():
                j = {}
                if uv_a is not None and obs_a.pose_vis is not None:
                    j["a"] = [round(float(uv_a[idx, 0]), 2),
                              round(float(uv_a[idx, 1]), 2),
                              round(float(obs_a.pose_vis[idx]), 3)]
                if uv_b is not None and obs_b.pose_vis is not None:
                    j["b"] = [round(float(uv_b[idx, 0]), 2),
                              round(float(uv_b[idx, 1]), 2),
                              round(float(obs_b.pose_vis[idx]), 3)]
                if fused is not None:
                    st = fused.states.get(name)
                    if st is not None and st not in (DISABLED,):
                        j["s"] = st
                    if name in fused.positions:
                        j["p"] = [round(float(v), 4)
                                  for v in fused.positions[name]]
                        j["w"] = round(fused.weights[name], 3)
                if j:
                    joints[name] = j
            rec = {"t": round(obs_a.ts, 4),
                   "skew_ms": (round(skew_ms, 2)
                               if skew_ms is not None else None),
                   "joints": joints}
            self._log_fh.write(json.dumps(rec) + "\n")
            self._log_lines += 1
        except Exception:
            self._log_fh = None

    # ------------------------------------------------------------------
    def info_line(self) -> str:
        st = self.get_stats()
        z = st.get("hips_z")
        ztxt = f" Z={z:.2f}m" if z is not None else ""
        states = st.get("states", {})
        n_lost = sum(1 for v in states.values() if v == LOST)
        return (f"2カメラ: ペア{st['pair_pct']:.0f}%"
                f" ズレ{st['skew_ms']:.0f}ms"
                f" 立体{st['n_stereo']}点/単眼{st['n_mono']}点"
                f" 喪失{n_lost}{ztxt}"
                f" (A {st['fps_a']:.0f}fps / B {st['fps_b']:.0f}fps)")

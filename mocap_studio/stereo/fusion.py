"""Per-joint stereo/mono fusion with re-detection ("復帰") control.

Responsibilities (kept separate from the user-facing smoothing that the
legacy pipeline applies downstream):

  * a small constant-velocity filter per joint (camera-A frame, metres)
    with distance-dependent depth uncertainty;
  * "re-detected" != "safe to output": observations arriving after a gap
    or far from the prediction become *candidates* and are only accepted
    after ``recovery_frames`` mutually-consistent observations;
  * re-acceptance never turns the correction offset into velocity - the
    new velocity is measured from the candidate history itself;
  * mono (camera-A-only) observations steer the track along the viewing
    ray while depth is held/predicted (the doc's 簡易+ approach);
  * missing observations are predicted with decaying velocity for a
    bounded time, then the joint reports LOST (weight 0) and the caller
    falls back to the legacy mono value.

All state lives inside this module; nothing here touches legacy state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Joint states (also used for the GUI/status display).
STEREO = "STEREO"
MONO = "MONO"
PREDICTED = "PREDICTED"
RECOVERING = "RECOVERING"
LOST = "LOST"
DISABLED = "DISABLED"


@dataclass
class FusionParams:
    min_visibility: float = 0.5
    max_epipolar_px: float = 8.0
    recovery_frames: int = 4
    predict_timeout_s: float = 0.22
    recover_blend_s: float = 0.18
    obs_grace_s: float = 0.08        # single missed frames stay in-state
    jump_gate_m: float = 0.35        # innovation gate (plus vmax * gap)
    vmax_body_ms: float = 4.0        # plausible joint speeds
    vmax_hand_ms: float = 6.0
    tau_velocity_s: float = 0.15     # velocity decay while predicting
    q_pos: float = 0.4               # process noise (m^2/s)
    sigma_px: float = 2.0            # assumed landmark pixel noise
    mono_gain: float = 0.55          # per-observation pull toward the ray
    candidate_max_age_s: float = 0.8
    min_var: float = 1e-6
    max_var: float = 1.0

    @classmethod
    def from_settings(cls, s2) -> "FusionParams":
        return cls(
            min_visibility=float(s2.min_visibility),
            max_epipolar_px=float(s2.max_epipolar_px),
            recovery_frames=int(s2.recovery_frames),
            predict_timeout_s=float(s2.predict_timeout_ms) / 1000.0,
            recover_blend_s=float(s2.recover_blend_ms) / 1000.0,
        )


_FAST_JOINTS = ("wrist", "pinky", "index", "thumb", "elbow")


class JointTrack:
    """State machine + per-axis filter for a single joint."""

    def __init__(self, name: str, params: FusionParams) -> None:
        self.name = name
        self.p = params
        self.vmax = (params.vmax_hand_ms
                     if any(k in name for k in _FAST_JOINTS)
                     else params.vmax_body_ms)
        self.state = LOST
        self.x = np.zeros(3)
        self.v = np.zeros(3)
        self.var = np.full(3, params.max_var)
        self._init = False
        self._t = None            # last step time
        self._last_obs_t = -1e9   # last accepted obs (stereo or mono)
        self.last_stereo_t = -1e9
        self._prev_pos: np.ndarray | None = None
        self._cand: list[tuple[float, np.ndarray]] = []
        self._blend_t0 = -1e9
        self.rejects = 0          # consecutive gated-out stereo obs

    # -- helpers ------------------------------------------------------
    def _predict(self, t: float) -> None:
        if self._t is None:
            self._t = t
            return
        dt = min(max(t - self._t, 0.0), 0.25)
        self._t = t
        if not self._init:
            return
        self.x = self.x + self.v * dt
        self.v = self.v * math.exp(-dt / max(self.p.tau_velocity_s, 1e-3))
        self.var = np.minimum(self.var + self.p.q_pos * dt, self.p.max_var)

    def _candidate(self, t: float, pos: np.ndarray, meas_var: np.ndarray
                   ) -> None:
        """Quarantine an observation; re-accept only after a consistent
        streak, with velocity measured from the candidates themselves."""
        self._cand = [(ct, cp) for ct, cp in self._cand
                      if t - ct <= self.p.candidate_max_age_s]
        self._cand.append((t, pos.copy()))
        if len(self._cand) < max(2, self.p.recovery_frames):
            return
        cand = self._cand[-self.p.recovery_frames:]
        for (t0, p0), (t1, p1) in zip(cand, cand[1:]):
            dt = t1 - t0
            if dt <= 0.0 or dt > 0.25:
                return
            if np.linalg.norm(p1 - p0) / dt > self.vmax:
                return
        # Consistent: re-initialise at the *observed* trajectory.
        span = cand[-1][0] - cand[0][0]
        self.x = cand[-1][1].copy()
        self.v = ((cand[-1][1] - cand[0][1]) / span if span > 1e-3
                  else np.zeros(3))
        # Never inherit the old track's velocity nor derive it from the
        # correction offset; covariance restarts wide-ish, not zero.
        self.var = np.maximum(meas_var * 2.0, self.p.min_var)
        self._init = True
        self.state = RECOVERING
        self._blend_t0 = t
        self._last_obs_t = t
        self.last_stereo_t = t
        self._prev_pos = self.x.copy()
        self._cand.clear()
        self.rejects = 0

    def _expire_if_stale(self, t: float) -> None:
        """The fusion may not have been stepped at all during a long
        detection outage (e.g. the person left camera A entirely); make
        sure such a gap still forces the candidate/confirmation path
        instead of slipping through a loose velocity gate."""
        if self._init and t - self._last_obs_t > self.p.predict_timeout_s:
            self.state = LOST
            self._init = False

    # -- observation entry points -------------------------------------
    def step_stereo(self, t: float, pos: np.ndarray,
                    meas_var: np.ndarray) -> None:
        self._predict(t)
        self._expire_if_stale(t)
        if not self._init or self.state == LOST:
            self._candidate(t, pos, meas_var)
            return
        gap = min(t - self._last_obs_t, 0.5)
        gate = self.p.jump_gate_m + self.vmax * max(gap, 1.0 / 60.0)
        d = pos - self.x
        if float(np.linalg.norm(d)) > gate:
            # Outlier or a genuinely moved target: quarantine, keep
            # predicting (the doc's "初回誤検出の飛び" defence).
            self.rejects += 1
            self._candidate(t, pos, meas_var)
            if self.rejects > 12:
                # persistent disagreement: stop trusting the old track
                self.state = LOST
                self._init = False
            return
        self.rejects = 0
        K = self.var / (self.var + np.maximum(meas_var, self.p.min_var))
        self.x = self.x + K * d
        self.var = np.maximum(self.var * (1.0 - K), self.p.min_var)
        if self._prev_pos is not None and gap > 1e-3:
            v_meas = (self.x - self._prev_pos) / max(gap, 1.0 / 120.0)
            n = float(np.linalg.norm(v_meas))
            if n > self.vmax:            # cap: corrections are not speed
                v_meas = v_meas * (self.vmax / n)
            self.v = 0.4 * self.v + 0.6 * v_meas
        self._prev_pos = self.x.copy()
        self._last_obs_t = t
        self.last_stereo_t = t
        if self.state != RECOVERING:
            self.state = STEREO
        self._cand.clear()

    def step_mono(self, t: float, ray: np.ndarray) -> None:
        """Camera-A-only observation: unit ray in the camera-A frame.
        Constrains direction; depth stays predicted/held."""
        self._predict(t)
        self._expire_if_stale(t)
        if not self._init or self.state == LOST:
            # A mono point alone cannot initialise an absolute 3D track.
            return
        depth = float(np.dot(self.x, ray))
        if depth < 0.15:
            return
        target = ray * depth              # same along-ray distance
        d = target - self.x
        gap = min(t - self._last_obs_t, 0.5)
        gate = self.p.jump_gate_m + self.vmax * max(gap, 1.0 / 60.0)
        if float(np.linalg.norm(d)) > gate:
            return
        self.x = self.x + self.p.mono_gain * d
        self.var = np.minimum(self.var + 0.02 * abs(gap), self.p.max_var)
        self._prev_pos = self.x.copy()
        self._last_obs_t = t
        if self.state not in (RECOVERING,):
            self.state = MONO if t - self.last_stereo_t > 0.2 else self.state

    def step_none(self, t: float) -> None:
        self._predict(t)

    def finish(self, t: float) -> None:
        """Update timing-based state transitions after the observations
        (or lack thereof) of this frame have been applied."""
        if not self._init:
            self.state = LOST
            return
        silent = t - self._last_obs_t
        if silent > self.p.predict_timeout_s:
            self.state = LOST
            self._init = False
        elif silent > self.p.obs_grace_s:
            if self.state != RECOVERING:
                self.state = PREDICTED
        if self.state == RECOVERING \
                and t - self._blend_t0 >= self.p.recover_blend_s \
                and t - self._last_obs_t <= self.p.obs_grace_s:
            self.state = STEREO if t - self.last_stereo_t <= 0.2 else MONO

    # -- output -------------------------------------------------------
    def output(self, t: float):
        """Returns (pos_camA_frame, weight 0..1) or None when invalid."""
        if not self._init or self.state in (LOST, DISABLED):
            return None
        w = 1.0
        silent = t - self._last_obs_t
        if silent > self.p.obs_grace_s:
            # fade out over the prediction window
            span = max(self.p.predict_timeout_s - self.p.obs_grace_s, 1e-3)
            w = 1.0 - min((silent - self.p.obs_grace_s) / span, 1.0)
        if self.state == RECOVERING or t - self._blend_t0 \
                < self.p.recover_blend_s:
            ramp = (t - self._blend_t0) / max(self.p.recover_blend_s, 1e-3)
            w *= float(np.clip(ramp, 0.0, 1.0))
        if w <= 0.0:
            return None
        return self.x.copy(), float(w)


@dataclass
class FusedFrame:
    """Result of one fusion step (camera-A frame, metres)."""
    t: float
    positions: dict = field(default_factory=dict)   # name -> (3,) ndarray
    weights: dict = field(default_factory=dict)     # name -> 0..1
    states: dict = field(default_factory=dict)      # name -> state str
    n_stereo: int = 0
    n_mono: int = 0
    pair_skew_ms: float | None = None               # None = mono frame


_LEG_JOINT_KEYS = ("knee", "ankle", "heel", "foot_index")


class StereoFusion:
    """Owns all joint tracks; consumes paired/mono observations."""

    def __init__(self, calib, params: FusionParams,
                 joint_names: dict[str, int]) -> None:
        """joint_names: pose joint name -> MediaPipe landmark index."""
        self.calib = calib
        self.p = params
        self.joints = dict(joint_names)
        self.tracks = {n: JointTrack(n, params) for n in self.joints}
        self._inv_Ka = np.linalg.inv(calib.K_a)
        self.legs_enabled = True

    def _ray_a(self, uv: np.ndarray) -> np.ndarray:
        v = self._inv_Ka @ np.array([uv[0], uv[1], 1.0])
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def step(self, t: float, obs_a, obs_b, pair_skew_ms: float | None
             ) -> FusedFrame:
        """obs_a/obs_b: mp2d.Observation (obs_b may be None = mono frame).

        obs_a must have pose data (caller guarantees); joints below the
        visibility gate in a camera are treated as unobserved there.
        """
        out = FusedFrame(t=t, pair_skew_ms=pair_skew_ms)
        uv_a = obs_a.pose_uv()
        vis_a = obs_a.pose_vis
        uv_b = obs_b.pose_uv() if obs_b is not None else None
        vis_b = obs_b.pose_vis if obs_b is not None else None

        active = {}
        for name, idx in self.joints.items():
            if not self.legs_enabled and any(
                    k in name for k in _LEG_JOINT_KEYS):
                self.tracks[name].state = DISABLED
                out.states[name] = DISABLED
                continue
            active[name] = idx

        # Batch-triangulate every joint visible in both cameras.
        stereo_names = []
        if uv_a is not None and uv_b is not None and vis_b is not None:
            for name, idx in active.items():
                if vis_a is not None and vis_a[idx] >= self.p.min_visibility \
                        and vis_b[idx] >= self.p.min_visibility:
                    stereo_names.append(name)
        tri = {}
        if stereo_names:
            ia = np.array([self.joints[n] for n in stereo_names])
            X, err, valid = self.calib.triangulate(
                uv_a[ia], uv_b[ia],
                max_err_px=self.p.max_epipolar_px)
            for k, name in enumerate(stereo_names):
                if valid[k]:
                    tri[name] = X[k]

        for name, idx in active.items():
            track = self.tracks[name]
            if name in tri:
                pos = tri[name]
                z = max(float(pos[2]), 0.2)
                s_xy = z * self.p.sigma_px / float(self.calib.K_a[0, 0])
                s_z = self.calib.depth_sigma(z, self.p.sigma_px)
                meas_var = np.array([s_xy ** 2, s_xy ** 2,
                                     max(s_z, s_xy) ** 2])
                track.step_stereo(t, pos, meas_var)
            elif (uv_a is not None and vis_a is not None
                    and vis_a[idx] >= self.p.min_visibility):
                track.step_mono(t, self._ray_a(uv_a[idx]))
            else:
                track.step_none(t)
            track.finish(t)
            res = track.output(t)
            out.states[name] = track.state
            if res is not None:
                out.positions[name], out.weights[name] = res
            if track.state == STEREO:
                out.n_stereo += 1
            elif track.state == MONO:
                out.n_mono += 1
        return out

    def reset(self) -> None:
        for name in self.tracks:
            self.tracks[name] = JointTrack(name, self.p)

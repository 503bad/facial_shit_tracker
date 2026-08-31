# -*- coding: utf-8 -*-
"""Synthetic verification of the 2-camera depth extension (no hardware).

Run:  .venv\\Scripts\\python.exe tests\\test_stereo_synthetic.py

Covers, with ground-truth geometry:
  * approximate calibration from clicked correspondences (R/t recovery)
  * triangulation accuracy under pixel noise
  * fusion state machine: spike rejection, loss -> prediction -> LOST,
    recovery confirmation, no velocity from corrections, mono depth hold
  * legacy invariance: BodyRetargeter output is unchanged without the
    stereo key; the mirror helper passes stereo metadata through
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mocap_studio.stereo.calibration import (StereoCalibration,
                                             estimate_relative_pose,
                                             intrinsics_from_fov)
from mocap_studio.stereo.fusion import (FusionParams, JointTrack,
                                        StereoFusion, LOST, MONO, STEREO)
from mocap_studio.stereo.mp2d import Observation

CHECKS = []


def check(name: str, cond: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(cond), detail))
    mark = "OK " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- setup
W, H = 1280, 720
FOV = 120.0
BASELINE = 0.30
K = intrinsics_from_fov(W, H, FOV)


def make_truth(ry_deg: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth R, t (camera B is BASELINE to the right of A, with a
    small yaw so the rig is not perfectly parallel)."""
    a = math.radians(ry_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)],
                  [0, 1, 0],
                  [-math.sin(a), 0, math.cos(a)]])
    c_b = np.array([BASELINE, 0.0, 0.0])
    t = -R @ c_b
    return R, t


R_TRUE, T_TRUE = make_truth()


def project(X: np.ndarray, R=None, t=None) -> np.ndarray:
    """3D points (N,3) in camera-A frame -> pixels in the given camera."""
    if R is not None:
        X = X @ R.T + t
    uv = np.empty((len(X), 2))
    uv[:, 0] = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]
    uv[:, 1] = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
    return uv


def truth_calib() -> StereoCalibration:
    return StereoCalibration(K_a=K.copy(), K_b=K.copy(),
                             R=R_TRUE.copy(), t=T_TRUE.copy(),
                             size_a=(W, H), size_b=(W, H),
                             baseline_m=BASELINE)


def scene_points(n: int, rng) -> np.ndarray:
    """Random points visible in both cameras, varied depth."""
    pts = []
    while len(pts) < n:
        p = np.array([rng.uniform(-1.2, 1.2), rng.uniform(-0.7, 0.7),
                      rng.uniform(0.6, 3.2)])
        ua = project(p[None])[0]
        ub = project(p[None], R_TRUE, T_TRUE)[0]
        if (20 < ua[0] < W - 20 and 20 < ua[1] < H - 20
                and 20 < ub[0] < W - 20 and 20 < ub[1] < H - 20):
            pts.append(p)
    return np.array(pts)


# ------------------------------------------------- 1. calibration
def test_calibration() -> StereoCalibration:
    rng = np.random.default_rng(7)
    X = scene_points(25, rng)
    uva = project(X) + rng.normal(0, 0.4, (len(X), 2))
    uvb = project(X, R_TRUE, T_TRUE) + rng.normal(0, 0.4, (len(X), 2))
    calib = estimate_relative_pose(uva, uvb, (W, H), (W, H),
                                   FOV, False, BASELINE)
    # rotation error
    dR = calib.R @ R_TRUE.T
    ang = math.degrees(math.acos(min(1.0, (np.trace(dR) - 1) / 2)))
    tdir = calib.t / np.linalg.norm(calib.t)
    tdir_true = T_TRUE / np.linalg.norm(T_TRUE)
    terr = math.degrees(math.acos(
        min(1.0, abs(float(np.dot(tdir, tdir_true))))))
    check("calibration: rotation recovered", ang < 0.5,
          f"err={ang:.3f} deg")
    check("calibration: baseline direction recovered", terr < 1.0,
          f"err={terr:.3f} deg")
    check("calibration: residual sane", calib.residual_med_px < 2.0,
          f"med={calib.residual_med_px:.2f}px")
    # degenerate input must raise
    try:
        estimate_relative_pose(uva[:, :] * 0 + [[640, 360]] * len(uva),
                               uvb, (W, H), (W, H), FOV, False, BASELINE)
        check("calibration: degenerate points rejected", False)
    except Exception:
        check("calibration: degenerate points rejected", True)
    try:
        estimate_relative_pose(uva[:5], uvb[:5], (W, H), (W, H),
                               FOV, False, BASELINE)
        check("calibration: too-few points rejected", False)
    except Exception:
        check("calibration: too-few points rejected", True)
    return calib


# ------------------------------------------------- 2. triangulation
def test_triangulation(calib: StereoCalibration) -> None:
    rng = np.random.default_rng(11)
    X = scene_points(300, rng)
    noise = 0.5
    uva = project(X) + rng.normal(0, noise, (len(X), 2))
    uvb = project(X, R_TRUE, T_TRUE) + rng.normal(0, noise, (len(X), 2))
    Xr, err, valid = calib.triangulate(uva, uvb)
    check("triangulate: most points valid",
          valid.mean() > 0.95, f"{valid.mean() * 100:.0f}%")
    near = valid & (X[:, 2] < 2.0)
    dz = np.abs(Xr[near, 2] - X[near, 2])
    # The approximate (clicked) calibration keeps a small yaw residual
    # that acts as a smooth depth *gain* error; absolute depth is only
    # approximate, which the design doc accepts.  What the app consumes
    # is depth relative to a neutral, so verify (a) bounded absolute
    # error and (b) a near-unit depth gain.
    check("triangulate(est-calib): depth p95 < 15cm for Z<2m",
          np.percentile(dz, 95) < 0.15,
          f"p95={np.percentile(dz, 95) * 100:.1f}cm")
    gain = float(np.sum(Xr[near, 2] * X[near, 2])
                 / np.sum(X[near, 2] ** 2))
    check("triangulate(est-calib): depth gain within 8%",
          abs(gain - 1.0) < 0.08, f"gain={gain:.3f}")
    # With exact calibration the error is noise-limited.
    Xr_t, _, valid_t = truth_calib().triangulate(uva, uvb)
    near_t = valid_t & (X[:, 2] < 2.0)
    dz_t = np.abs(Xr_t[near_t, 2] - X[near_t, 2])
    check("triangulate(true-calib): depth p95 < 5cm for Z<2m",
          np.percentile(dz_t, 95) < 0.05,
          f"p95={np.percentile(dz_t, 95) * 100:.1f}cm")
    # behind-camera / mismatched pairs must be flagged invalid
    Xr2, err2, valid2 = calib.triangulate(uva[:10], uvb[10:20])
    check("triangulate: mismatched pairs mostly rejected",
          valid2.mean() < 0.5, f"{valid2.mean() * 100:.0f}% pass")


# ------------------------------------------------- 3. fusion behaviour
JOINT = "left_wrist"
JIDX = 15


def obs_for(t, pts3d: dict[str, np.ndarray], cam: str,
            vis: float = 1.0) -> Observation:
    pose_norm = np.zeros((33, 2))
    pvis = np.zeros(33)
    for name, p in pts3d.items():
        uv = (project(p[None])[0] if cam == "a"
              else project(p[None], R_TRUE, T_TRUE)[0])
        idx = {"left_wrist": 15, "left_hip": 23, "right_hip": 24}[name]
        pose_norm[idx] = (uv[0] / W, uv[1] / H)
        pvis[idx] = vis
    return Observation(ts=t, seq=0, img_size=(W, H),
                       pose_norm=pose_norm, pose_vis=pvis,
                       pose_world=np.zeros((33, 3)))


def test_fusion(calib: StereoCalibration) -> None:
    params = FusionParams(recovery_frames=4)
    joints = {"left_wrist": 15, "left_hip": 23, "right_hip": 24}
    fusion = StereoFusion(calib, params, joints)
    dt = 1.0 / 30.0
    rng = np.random.default_rng(3)

    def wrist(t):
        return np.array([0.3 * math.sin(t * 2.0), 0.0, 1.5])

    hips = {"left_hip": np.array([-0.1, 0.3, 1.6]),
            "right_hip": np.array([0.1, 0.3, 1.6])}

    # steady tracking
    t = 0.0
    last = None
    for i in range(45):
        t += dt
        pts = dict(hips, left_wrist=wrist(t))
        f = fusion.step(t, obs_for(t, pts, "a"), obs_for(t, pts, "b"), 2.0)
        last = f
    check("fusion: steady state is STEREO",
          last.states[JOINT] == STEREO)
    err = np.linalg.norm(last.positions[JOINT] - wrist(t))
    check("fusion: steady position accurate", err < 0.05,
          f"err={err * 100:.1f}cm")

    # single-frame spike (bogus far observation) must not move the output
    before = last.positions[JOINT].copy()
    t += dt
    pts = dict(hips, left_wrist=np.array([0.5, -0.5, 3.2]))  # fake jump
    f = fusion.step(t, obs_for(t, pts, "a"), obs_for(t, pts, "b"), 2.0)
    moved = np.linalg.norm(f.positions[JOINT] - before) \
        if JOINT in f.positions else 0.0
    check("fusion: 1-frame spike rejected", moved < 0.10,
          f"moved={moved * 100:.1f}cm")
    # spike must not corrupt velocity once real obs resume
    for i in range(3):
        t += dt
        pts = dict(hips, left_wrist=wrist(t))
        f = fusion.step(t, obs_for(t, pts, "a"), obs_for(t, pts, "b"), 2.0)
    err = np.linalg.norm(f.positions[JOINT] - wrist(t))
    check("fusion: recovers from spike", err < 0.08,
          f"err={err * 100:.1f}cm")

    # mono fallback: camera B lost -> XY follows, depth held
    z_before = f.positions[JOINT][2]
    for i in range(8):
        t += dt
        pts = dict(hips, left_wrist=wrist(t))
        f = fusion.step(t, obs_for(t, pts, "a"), None, None)
    check("fusion: mono state after B loss",
          f.states[JOINT] in (MONO, STEREO),
          f.states[JOINT])
    xy_err = np.linalg.norm(f.positions[JOINT][:2] - wrist(t)[:2])
    check("fusion: mono keeps following XY", xy_err < 0.08,
          f"err={xy_err * 100:.1f}cm")
    check("fusion: mono holds depth",
          abs(f.positions[JOINT][2] - z_before) < 0.15,
          f"dz={abs(f.positions[JOINT][2] - z_before) * 100:.1f}cm")

    # full loss -> LOST after timeout (no infinite prediction)
    for i in range(20):
        t += dt
        f = fusion.step(t, obs_for(t, hips, "a"), obs_for(t, hips, "b"),
                        2.0)
    check("fusion: LOST after timeout", f.states[JOINT] == LOST)
    check("fusion: LOST joint has no output", JOINT not in f.positions)

    # reappear at a genuinely different position: accepted only after
    # recovery_frames, ramped in, velocity not derived from the gap
    new_pos = np.array([-0.4, 0.2, 2.2])
    outs = []
    for i in range(20):
        t += dt
        p = new_pos + np.array([0.01 * i, 0.0, 0.0])   # slow real motion
        pts = dict(hips, left_wrist=p)
        f = fusion.step(t, obs_for(t, pts, "a"), obs_for(t, pts, "b"), 2.0)
        outs.append((f.states.get(JOINT),
                     f.positions.get(JOINT),
                     f.weights.get(JOINT, 0.0)))
    n_none = sum(1 for s, p, w in outs if p is None)
    check("fusion: recovery needs confirmation frames",
          3 <= n_none <= 8, f"quarantined {n_none} frames")
    final_state, final_pos, final_w = outs[-1]
    check("fusion: re-acquired at new position",
          final_pos is not None
          and np.linalg.norm(final_pos - (new_pos + [0.19, 0, 0])) < 0.10,
          f"state={final_state}")
    check("fusion: recovered weight ramps to 1", final_w > 0.95,
          f"w={final_w:.2f}")
    # long outage where fusion is never stepped at all (person absent
    # from camera A): the first observation after the gap must still be
    # quarantined, not accepted through a loose velocity gate
    t += 1.0
    far = np.array([0.5, -0.3, 0.9])
    pts = dict(hips, left_wrist=far)
    f = fusion.step(t, obs_for(t, pts, "a"), obs_for(t, pts, "b"), 2.0)
    ok_gap = (JOINT not in f.positions
              or np.linalg.norm(f.positions[JOINT] - far) > 0.3)
    check("fusion: unstepped 1s gap still quarantines first obs", ok_gap,
          f.states.get(JOINT))

    # output never crosses space at implausible speed during recovery
    speeds = []
    prev = None
    for s, p, w in outs:
        if p is not None and prev is not None:
            speeds.append(np.linalg.norm(p - prev) / dt)
        prev = p
    check("fusion: no teleport velocity after recovery",
          max(speeds) < params.vmax_hand_ms if speeds else True,
          f"max={max(speeds):.2f}m/s" if speeds else "no speeds")


# ------------------------------------------------- 4. legacy invariance
def test_legacy_invariance() -> None:
    from mocap_studio.body_pipeline import (BodyRetargeter,
                                            DEFAULT_BONE_OFFSETS)

    def base_pose(z_key: float | None = None) -> dict:
        pose = {
            "left_shoulder": np.array([-0.2, 0.5, 0.0]),
            "right_shoulder": np.array([0.2, 0.5, 0.0]),
            "left_elbow": np.array([-0.45, 0.5, 0.0]),
            "right_elbow": np.array([0.45, 0.5, 0.0]),
            "left_wrist": np.array([-0.7, 0.5, 0.0]),
            "right_wrist": np.array([0.7, 0.5, 0.0]),
            "left_hip": np.array([-0.1, 0.0, 0.0]),
            "right_hip": np.array([0.1, 0.0, 0.0]),
        }
        if z_key is not None:
            pose["_stereo_hips_z"] = z_key
        return pose

    # without the stereo key: root must stay at the bind offset (legacy)
    r = BodyRetargeter(0.0, 0.0)
    bones = None
    for i in range(60):
        bones = r.process(base_pose(), None, None, i / 30.0)
    check("legacy: root fixed without stereo key",
          tuple(bones["Hips"][0]) == DEFAULT_BONE_OFFSETS["Hips"])

    # with the stereo key: depth moves the root along +Z after stepping in
    r = BodyRetargeter(0.0, 0.0)
    for i in range(60):
        r.process(base_pose(1.8), None, None, i / 30.0)     # neutral
    for i in range(60, 150):
        bones = r.process(base_pose(1.4), None, None, i / 30.0)
    dz = bones["Hips"][0][2] - DEFAULT_BONE_OFFSETS["Hips"][2]
    check("stereo hook: root depth follows (+Z toward viewer)",
          0.3 < dz < 0.5, f"dz={dz:.3f}m")

    # key removed again: root glides back to the bind offset
    for i in range(150, 400):
        bones = r.process(base_pose(), None, None, i / 30.0)
    dz = abs(bones["Hips"][0][2] - DEFAULT_BONE_OFFSETS["Hips"][2])
    check("stereo hook: root returns to neutral after OFF",
          dz < 0.01, f"resid={dz:.4f}m")

    # mirror helper passes stereo metadata through untouched
    from mocap_studio.tracker import _mirror_body
    pose = base_pose(1.23)
    pose["_vis"] = {"left_hip": 1.0, "right_hip": 0.5}
    m, _, _ = _mirror_body(pose, None, None)
    check("mirror: stereo key passes through",
          m.get("_stereo_hips_z") == 1.23)
    check("mirror: vis swap still works",
          m["_vis"]["right_hip"] == 1.0 and m["_vis"]["left_hip"] == 0.5)


# ------------------------------------------------- 5. settings isolation
def test_settings() -> None:
    import tempfile
    from mocap_studio.stereo.config import StereoSettings
    check("settings: default OFF", StereoSettings().enabled is False)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.json"
        s = StereoSettings(enabled=True, camera_b_index=2, fov_deg=110.0)
        s.save(p)
        s2 = StereoSettings.load(p)
        check("settings: roundtrip",
              s2.enabled and s2.camera_b_index == 2
              and s2.fov_deg == 110.0)
        p.write_text("{broken", encoding="utf-8")
        check("settings: corrupt file -> defaults (no crash)",
              StereoSettings.load(p).enabled is False)


if __name__ == "__main__":
    calib = test_calibration()
    test_triangulation(calib)
    test_fusion(truth_calib())
    test_legacy_invariance()
    test_settings()
    failed = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    sys.exit(1 if failed else 0)

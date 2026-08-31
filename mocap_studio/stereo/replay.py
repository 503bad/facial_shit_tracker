"""Offline replay / verification of recorded stereo observations.

Usage:
    python -m mocap_studio.stereo.replay stereo_logs/stereo_XXXX.jsonl

Re-runs pairing-independent fusion on the recorded per-frame joint
observations (written when 「観測ログを記録」 is enabled) and prints
quality metrics: stereo success rate, pair skew, depth statistics, and
jump sizes (p95 / max) split into normal motion vs. recovery frames.
The same input therefore reproduces the same fusion decisions, which is
the reproducibility requirement of the design doc.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..mp_body import MP_POSE
from .calibration import StereoCalibration
from .fusion import FusionParams, StereoFusion, RECOVERING
from .mp2d import Observation


def _obs_from_rec(joints: dict, key: str, ts: float, size) -> Observation:
    pose_norm = np.full((33, 2), np.nan)
    vis = np.zeros(33)
    found = False
    w, h = size
    for name, idx in MP_POSE.items():
        j = joints.get(name)
        if j and key in j:
            u, v, vv = j[key]
            pose_norm[idx] = (u / w, v / h)
            vis[idx] = vv
            found = True
    if not found:
        return Observation(ts=ts, seq=0, img_size=size)
    pose_norm = np.nan_to_num(pose_norm, nan=0.0)
    return Observation(ts=ts, seq=0, img_size=size,
                       pose_norm=pose_norm, pose_vis=vis,
                       pose_world=np.zeros((33, 3)))


def replay(path: str) -> int:
    header = None
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("type") == "header":
                header = rec
            else:
                records.append(rec)
    if header is None:
        print("ヘッダー行がありません（校正情報が必要です）")
        return 1
    calib = StereoCalibration.from_dict(header["calib"])
    s = header.get("settings", {})
    params = FusionParams(
        min_visibility=s.get("min_visibility", 0.5),
        max_epipolar_px=s.get("max_epipolar_px", 8.0),
        recovery_frames=int(s.get("recovery_frames", 4)),
        predict_timeout_s=s.get("predict_timeout_ms", 220.0) / 1000.0,
        recover_blend_s=s.get("recover_blend_ms", 180.0) / 1000.0)
    fusion = StereoFusion(calib, params, dict(MP_POSE))

    n = 0
    n_pair = 0
    skews = []
    stereo_counts = []
    prev_pos: dict[str, tuple[float, np.ndarray]] = {}
    jumps_normal = []
    jumps_recover = []
    hip_z = []
    for rec in records:
        t = rec["t"]
        joints = rec["joints"]
        obs_a = _obs_from_rec(joints, "a", t, calib.size_a)
        obs_b = _obs_from_rec(joints, "b", t, calib.size_b)
        if obs_a.pose_norm is None:
            continue
        skew = rec.get("skew_ms")
        has_b = obs_b.pose_norm is not None and skew is not None
        fused = fusion.step(t, obs_a, obs_b if has_b else None, skew)
        n += 1
        if has_b:
            n_pair += 1
            skews.append(abs(skew))
        stereo_counts.append(fused.n_stereo)
        for name, pos in fused.positions.items():
            if name in prev_pos:
                pt, pp = prev_pos[name]
                dt = t - pt
                if 0.0 < dt < 0.2:
                    jump = float(np.linalg.norm(pos - pp))
                    if fused.states.get(name) == RECOVERING:
                        jumps_recover.append(jump / dt)
                    else:
                        jumps_normal.append(jump / dt)
            prev_pos[name] = (t, pos)
        if ("left_hip" in fused.positions
                and "right_hip" in fused.positions):
            hip_z.append(float((fused.positions["left_hip"][2]
                                + fused.positions["right_hip"][2]) / 2))

    def p(v, q):
        return float(np.percentile(v, q)) if v else float("nan")

    print(f"フレーム数            : {n}")
    print(f"ステレオペア率        : {100.0 * n_pair / max(n, 1):.1f}%")
    print(f"ペア時刻差 中央値/p95 : {p(skews, 50):.1f} / {p(skews, 95):.1f} ms")
    print(f"立体化関節数 平均     : {np.mean(stereo_counts):.1f}")
    if hip_z:
        print(f"腰の距離 Z 中央値     : {np.median(hip_z):.3f} m "
              f"(σ={np.std(hip_z):.3f})")
    print("関節速度 [m/s]（通常）: "
          f"p95={p(jumps_normal, 95):.2f} max="
          f"{max(jumps_normal) if jumps_normal else float('nan'):.2f}")
    print("関節速度 [m/s]（復帰）: "
          f"p95={p(jumps_recover, 95):.2f} max="
          f"{max(jumps_recover) if jumps_recover else float('nan'):.2f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(replay(sys.argv[1]))

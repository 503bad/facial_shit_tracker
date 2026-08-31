"""Approximate stereo calibration from hand-clicked natural correspondences.

Pipeline ("近似校正" as defined in the design doc):

  1. Intrinsics are seeded from the nominal field of view (no printed
     markers, no chessboard).
  2. The user clicks the same physical corners (furniture, shelves, ...)
     in both still images; the relative pose is estimated robustly with
     ``findEssentialMat`` + ``recoverPose`` on normalized points.
  3. The unit translation is scaled by the measured baseline length.

The profile records everything needed to reproduce / invalidate the
calibration (camera ids, capture sizes, FOV, points, residuals).

Coordinate conventions: camera frames are OpenCV-style (x right, y down,
z forward).  ``triangulate`` returns points in the camera-A frame.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import CALIBRATION_PATH


class CalibrationError(RuntimeError):
    """Raised with a user-readable (Japanese) message."""


def intrinsics_from_fov(width: int, height: int, fov_deg: float,
                        diagonal: bool = False) -> np.ndarray:
    """Pinhole K from a nominal FOV (principal point at image centre)."""
    fov = math.radians(max(1.0, min(179.0, float(fov_deg))))
    if diagonal:
        half_span = 0.5 * math.hypot(width, height)
    else:
        half_span = 0.5 * width
    f = half_span / math.tan(fov / 2.0)
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _normalize_pts(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel -> normalized image coordinates (no distortion model)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - K[0, 2]) / K[0, 0]
    out[:, 1] = (pts[:, 1] - K[1, 2]) / K[1, 1]
    return out


@dataclass
class StereoCalibration:
    """Calibration profile: intrinsics + relative pose (B w.r.t. A)."""

    K_a: np.ndarray                    # 3x3
    K_b: np.ndarray                    # 3x3
    R: np.ndarray                      # 3x3, x_b = R @ x_a + t
    t: np.ndarray                      # (3,), metres (|t| = baseline)
    size_a: tuple[int, int]            # (w, h) capture size used
    size_b: tuple[int, int]
    camera_a_index: int = 0
    camera_b_index: int = 1
    fov_deg: float = 120.0
    fov_is_diagonal: bool = False
    baseline_m: float = 0.30
    status: str = "approx"             # "approx" | "verified"
    residual_med_px: float = 0.0
    residual_p95_px: float = 0.0
    n_points: int = 0
    n_inliers: int = 0
    created: str = ""
    points_a: list = field(default_factory=list)   # clicked pixels (audit)
    points_b: list = field(default_factory=list)

    # -- derived ------------------------------------------------------
    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.t))

    def scaled_to(self, size_a: tuple[int, int],
                  size_b: tuple[int, int]) -> "StereoCalibration":
        """Adapt intrinsics to a different capture resolution (same
        aspect / sensor crop assumed; a pure resize scales K linearly)."""
        def scale_K(K, old, new):
            sx, sy = new[0] / old[0], new[1] / old[1]
            K2 = K.copy()
            K2[0, 0] *= sx
            K2[0, 2] *= sx
            K2[1, 1] *= sy
            K2[1, 2] *= sy
            return K2
        c = dataclasses.replace(
            self,
            K_a=scale_K(self.K_a, self.size_a, size_a),
            K_b=scale_K(self.K_b, self.size_b, size_b),
            size_a=tuple(size_a), size_b=tuple(size_b))
        return c

    # -- geometry -----------------------------------------------------
    def triangulate(self, uv_a: np.ndarray, uv_b: np.ndarray,
                    max_err_px: float = 8.0, min_z: float = 0.15,
                    max_z: float = 8.0):
        """Triangulate matched pixel points.

        Returns ``(X, err_px, valid)``: X (N,3) in the camera-A frame
        [m]; err_px worst reprojection error per point; valid bool mask
        (positive finite depth in both views and small residual).
        """
        na = _normalize_pts(uv_a, self.K_a)
        nb = _normalize_pts(uv_b, self.K_b)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([self.R, self.t.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P1, P2, na.T.copy(), nb.T.copy())
        w = Xh[3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        X = (Xh[:3] / w).T                       # (N,3) camera-A frame
        za = X[:, 2]
        Xb = X @ self.R.T + self.t
        zb = Xb[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            pa = X[:, :2] / za[:, None]
            pb = Xb[:, :2] / zb[:, None]
        f_a = float(self.K_a[0, 0])
        f_b = float(self.K_b[0, 0])
        err_a = np.linalg.norm(pa - na, axis=1) * f_a
        err_b = np.linalg.norm(pb - nb, axis=1) * f_b
        err = np.maximum(err_a, err_b)
        finite = np.isfinite(X).all(axis=1) & np.isfinite(err)
        valid = (finite & (za > min_z) & (zb > min_z)
                 & (za < max_z) & (zb < max_z) & (err <= max_err_px))
        return X, err, valid

    def depth_sigma(self, z: float, sigma_px: float = 2.0) -> float:
        """First-order depth uncertainty at distance z for a disparity
        error of sigma_px (sigma_Z = Z^2 / (f B) * sigma_d)."""
        f = float(self.K_a[0, 0])
        b = max(self.baseline, 1e-6)
        return (z * z) / (f * b) * sigma_px

    # -- persistence --------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        for k in ("K_a", "K_b", "R", "t"):
            d[k] = np.asarray(getattr(self, k)).tolist()
        d["size_a"] = list(self.size_a)
        d["size_b"] = list(self.size_b)
        return d

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2),
                        encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "StereoCalibration":
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("K_a", "K_b", "R", "t"):
            d[k] = np.asarray(d[k], dtype=np.float64)
        d["size_a"] = tuple(d["size_a"])
        d["size_b"] = tuple(d["size_b"])
        return cls(**d)

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "StereoCalibration | None":
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


# ---------------------------------------------------------------------
def _refine_relative_pose(na: np.ndarray, nb: np.ndarray,
                          R: np.ndarray, t: np.ndarray,
                          iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt refinement of (R, t-direction) minimizing the
    reprojection error of the triangulated points in both views.

    With a near-parallel narrow-baseline rig, ``recoverPose`` leaves a
    residual yaw error that biases depth linearly (dZ ~ Z^2 * dyaw / B);
    this refinement removes most of it.  Pure numpy/OpenCV, ~6 params,
    numeric Jacobian - cheap for 10-40 correspondence points.
    """
    t = t / max(np.linalg.norm(t), 1e-12)

    def residuals(rvec: np.ndarray, tv: np.ndarray) -> np.ndarray:
        Rm, _ = cv2.Rodrigues(rvec)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([Rm, tv.reshape(3, 1)])
        Xh = cv2.triangulatePoints(P1, P2, na.T.copy(), nb.T.copy())
        w = np.where(np.abs(Xh[3]) < 1e-12, 1e-12, Xh[3])
        X = (Xh[:3] / w).T
        za = np.where(np.abs(X[:, 2]) < 1e-9, 1e-9, X[:, 2])
        Xb = X @ Rm.T + tv
        zb = np.where(np.abs(Xb[:, 2]) < 1e-9, 1e-9, Xb[:, 2])
        ra = X[:, :2] / za[:, None] - na
        rb = Xb[:, :2] / zb[:, None] - nb
        return np.concatenate([ra.ravel(), rb.ravel()])

    rvec, _ = cv2.Rodrigues(R)
    params = np.concatenate([rvec.ravel(), t])
    lam = 1e-4
    best = residuals(params[:3], params[3:])
    best_cost = float(best @ best)
    for _ in range(iters):
        r0 = residuals(params[:3], params[3:])
        J = np.empty((len(r0), 6))
        eps = 1e-6
        for k in range(6):
            dp = params.copy()
            dp[k] += eps
            tv = dp[3:] / max(np.linalg.norm(dp[3:]), 1e-12)
            J[:, k] = (residuals(dp[:3], tv) - r0) / eps
        JtJ = J.T @ J
        g = J.T @ r0
        try:
            step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ))
                                   + 1e-12 * np.eye(6), -g)
        except np.linalg.LinAlgError:
            break
        cand = params + step
        cand[3:] /= max(np.linalg.norm(cand[3:]), 1e-12)
        r1 = residuals(cand[:3], cand[3:])
        cost = float(r1 @ r1)
        if cost < best_cost:
            params, best_cost = cand, cost
            lam = max(lam * 0.5, 1e-8)
            if float(np.linalg.norm(step)) < 1e-10:
                break
        else:
            lam = min(lam * 4.0, 1e2)
    Rr, _ = cv2.Rodrigues(params[:3])
    tr = params[3:] / max(np.linalg.norm(params[3:]), 1e-12)
    return Rr, tr


def estimate_relative_pose(
        pts_a: np.ndarray, pts_b: np.ndarray,
        size_a: tuple[int, int], size_b: tuple[int, int],
        fov_deg: float, fov_is_diagonal: bool, baseline_m: float,
        camera_a_index: int = 0, camera_b_index: int = 1,
) -> StereoCalibration:
    """Estimate R,t from clicked correspondences (approximate calibration).

    Raises CalibrationError with an actionable Japanese message when the
    point set is insufficient or degenerate.
    """
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    pts_b = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    n = len(pts_a)
    if n != len(pts_b):
        raise CalibrationError("左右の対応点の数が一致していません。")
    if n < 10:
        raise CalibrationError(
            f"対応点が不足しています（{n}組）。10組以上、推奨20〜30組を"
            "画面の上下左右・手前・奥に分散させてクリックしてください。")

    # Spread check: points concentrated in a small region or on a line
    # give a degenerate essential matrix.
    for name, pts, size in (("カメラA", pts_a, size_a),
                            ("カメラB", pts_b, size_b)):
        std = pts.std(axis=0)
        if std[0] < size[0] * 0.08 or std[1] < size[1] * 0.08:
            raise CalibrationError(
                f"{name}の対応点が狭い範囲に集中しています。"
                "画像の広い範囲（四隅の方向にも）に分散させてください。")

    K_a = intrinsics_from_fov(size_a[0], size_a[1], fov_deg, fov_is_diagonal)
    K_b = intrinsics_from_fov(size_b[0], size_b[1], fov_deg, fov_is_diagonal)
    na = _normalize_pts(pts_a, K_a)
    nb = _normalize_pts(pts_b, K_b)

    thr = 2.5 / float(K_a[0, 0])   # ~2.5 px in normalized units
    E, mask = cv2.findEssentialMat(
        na, nb, focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=thr)
    if E is None or E.shape[0] < 3:
        raise CalibrationError(
            "基本行列を推定できませんでした。対応点の取り違えがないか、"
            "同一平面だけに点が集中していないか確認してください。")
    E = E[:3, :3]
    n_pose, R, t, pose_mask = cv2.recoverPose(E, na, nb, mask=mask)
    inliers = int(np.count_nonzero(pose_mask))
    if inliers < 8 or inliers < 0.5 * n:
        raise CalibrationError(
            f"整合する対応点が少なすぎます（{inliers}/{n}組）。"
            "番号を確認し、間違った組を削除して再計算してください。")

    t = t.reshape(3)
    # Non-linear refinement on the inliers (kills the residual yaw error
    # that would otherwise bias depth on a near-parallel rig).
    sel0 = np.asarray(pose_mask).reshape(-1) != 0
    R, t = _refine_relative_pose(na[sel0], nb[sel0],
                                 np.asarray(R, dtype=np.float64),
                                 t.astype(np.float64))
    t = t / max(np.linalg.norm(t), 1e-12) * float(baseline_m)

    calib = StereoCalibration(
        K_a=K_a, K_b=K_b, R=np.asarray(R, dtype=np.float64), t=t,
        size_a=tuple(size_a), size_b=tuple(size_b),
        camera_a_index=camera_a_index, camera_b_index=camera_b_index,
        fov_deg=float(fov_deg), fov_is_diagonal=bool(fov_is_diagonal),
        baseline_m=float(baseline_m), status="approx",
        n_points=n, n_inliers=inliers,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        points_a=pts_a.tolist(), points_b=pts_b.tolist())

    # Validate by triangulating the inliers: demand mostly-positive
    # depths and record residual statistics.
    sel = np.asarray(pose_mask).reshape(-1) != 0
    X, err, valid = calib.triangulate(pts_a[sel], pts_b[sel],
                                      max_err_px=6.0, min_z=0.05, max_z=30.0)
    pos = np.count_nonzero(valid)
    if pos < max(6, 0.6 * inliers):
        raise CalibrationError(
            "三角測量の検証に失敗しました（正の距離になる点が少なすぎます）。"
            "カメラの左右順（AとB）や対応点の組を確認してください。")
    good_err = err[valid]
    calib.residual_med_px = float(np.median(good_err))
    calib.residual_p95_px = float(np.percentile(good_err, 95))
    if calib.residual_med_px > 5.0:
        raise CalibrationError(
            f"再投影誤差が大きすぎます（中央値 {calib.residual_med_px:.1f}px）。"
            "クリック位置の精度を上げるか、画角・解像度の設定を確認してください。")
    return calib

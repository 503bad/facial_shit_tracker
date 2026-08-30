"""Ground contact / foot-lock solver ("接地IK").

Runs on the sender side after the FK rotations are final.  Using the
avatar's real bone offsets it computes the world position of both feet,
then:

- grounds the lower foot (hips Y so the lowest toe sits on the floor),
- locks the planted foot's XZ (hips XZ solved so it does not slide;
  switching feet uses the FK position at the switch, so no jumps),
- flattens the planted foot's sole (yaw-only foot rotation),
- never snaps to the image-based hips translation: while airborne or
  standing still on both feet it only bleeds toward it slowly (bounds
  drift without pops); stance detection has hysteresis and the per-frame
  hips XZ change is capped as a last safety.

Only rotations of the two Foot bones and the Hips translation are
modified; everything else passes through untouched.
"""

from __future__ import annotations

import math

import numpy as np

from .retarget import (IDENTITY, normalize, quat_from_axis_angle, quat_inv,
                       quat_mul)
from .smoothing import quat_slerp

_UP = np.array([0.0, 1.0, 0.0])
_CHAIN = ("UpperLeg", "LowerLeg", "Foot", "Toes")


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_mul(quat_mul(q, qv), quat_inv(q))[:3]


def _q(rotations: dict, name: str) -> np.ndarray:
    q = rotations.get(name)
    return np.asarray(q, dtype=np.float64) if q is not None else IDENTITY.copy()


class GroundSolver:
    def __init__(self) -> None:
        self.plant_height_m = 0.06     # enter stance: ankle height diff below
        self.plant_speed_mps = 0.5     # enter stance: ankle speed below
        self.unplant_height_m = 0.10   # leave stance: height diff above
        self.unplant_speed_mps = 0.9   # leave stance: speed above
        self.recenter_rate = 0.15      # 1/s pull toward the image estimate
        self.max_step_m = 0.05         # hips XZ change per frame (safety)
        self.flatten_blend_sec = 0.15
        self._lock: dict[str, np.ndarray | None] = {"Left": None, "Right": None}
        self._planted: dict[str, bool] = {"Left": False, "Right": False}
        self._dominant: str | None = None
        self._plant_w: dict[str, float] = {"Left": 0.0, "Right": 0.0}
        self._prev_ankle: dict[str, tuple[float, np.ndarray] | None] = {
            "Left": None, "Right": None}
        self._hips_xz = None   # solved hips XZ delta carried between frames
        self._t_prev: float | None = None

    def reset(self) -> None:
        self.__init__()

    # ------------------------------------------------------------------
    def _fk_leg(self, side: str, rotations: dict, offsets: dict,
                hips_pos: np.ndarray, hips_q: np.ndarray):
        """World positions/rotations of UpperLeg, LowerLeg, Foot, Toes."""
        pos, q = hips_pos, hips_q
        out = {}
        for bone in _CHAIN:
            name = f"{side}{bone}"
            off = np.asarray(offsets.get(name, (0.0, 0.0, 0.0)), dtype=np.float64)
            pos = pos + _rot(q, off)
            q = quat_mul(q, _q(rotations, name))
            out[name] = (pos, q)
        return out

    def _stance(self, side: str, pose: dict, t: float) -> bool:
        a = pose.get(f"{side.lower()}_ankle")
        o = pose.get(f"{'right' if side == 'Left' else 'left'}_ankle")
        vis = pose.get("_vis", {})
        if a is None or o is None or vis.get(f"{side.lower()}_ankle", 1.0) < 0.5:
            return False
        speed = 0.0
        prev = self._prev_ankle[side]
        if prev is not None and t - prev[0] > 1e-3:
            speed = float(np.linalg.norm(a - prev[1]) / (t - prev[0]))
        self._prev_ankle[side] = (t, np.asarray(a, dtype=np.float64))
        height = float(a[1] - o[1])
        # hysteresis: harder to enter than to stay, so a brief speed or
        # height blip does not toggle the stance
        if self._planted[side]:
            stay = (height <= self.unplant_height_m
                    and speed <= self.unplant_speed_mps)
            self._planted[side] = stay
        else:
            self._planted[side] = (height <= self.plant_height_m
                                   and speed <= self.plant_speed_mps)
        return self._planted[side]

    # ------------------------------------------------------------------
    def solve(self, rotations: dict, hips_delta: np.ndarray, offsets: dict,
              pose: dict | None, t: float):
        """Returns (hips_delta, rotations) adjusted for ground contact."""
        if pose is None:
            return hips_delta, rotations
        vis = pose.get("_vis", {})
        if (vis.get("left_ankle", 0.0) < 0.5 and
                vis.get("right_ankle", 0.0) < 0.5):
            return hips_delta, rotations
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 1e-3)
        self._t_prev = t

        hips_bind = np.asarray(offsets.get("Hips", (0.0, 1.0, 0.0)), dtype=np.float64)
        hips_q = _q(rotations, "Hips")
        delta = np.array(hips_delta, dtype=np.float64)
        if self._hips_xz is not None:
            delta[0], delta[2] = self._hips_xz  # carry the locked solution

        # --- stance detection from tracking -------------------------
        planted = {s: self._stance(s, pose, t) for s in ("Left", "Right")}

        # --- FK with the current hips estimate -----------------------
        hips_pos = hips_bind + delta
        fk = {s: self._fk_leg(s, rotations, offsets, hips_pos, hips_q)
              for s in ("Left", "Right")}
        toes = {s: fk[s][f"{s}Toes"][0] for s in ("Left", "Right")}

        # --- ground the lower foot (no floating / sinking) -----------
        lowest = min(toes["Left"][1], toes["Right"][1])
        delta[1] -= lowest
        for s in toes:
            toes[s] = toes[s] + np.array([0.0, -lowest, 0.0])

        # --- foot lock: solve hips XZ so the planted foot stays put --
        if self._dominant is not None and not planted[self._dominant]:
            self._dominant = None
        if self._dominant is None:
            for s in ("Left", "Right"):
                if planted[s]:
                    self._dominant = s
                    self._lock[s] = toes[s][[0, 2]].copy()  # continuity
                    break
        for s in ("Left", "Right"):
            if not planted[s]:
                self._lock[s] = None
            elif self._lock[s] is None:
                self._lock[s] = toes[s][[0, 2]].copy()

        prev_xz = (np.array(self._hips_xz) if self._hips_xz is not None
                   else np.array([delta[0], delta[2]]))
        if self._dominant is not None:
            lock = self._lock[self._dominant]
            cur = toes[self._dominant][[0, 2]]
            shift = lock - cur
            delta[0] += shift[0]
            delta[2] += shift[1]
        # Bleed toward the image estimate while standing still on both
        # feet or while airborne - never snap to it (that was the pop).
        if (all(planted.values()) or self._dominant is None) and dt > 0:
            k = min(1.0, self.recenter_rate * dt)
            target = np.array([hips_delta[0], hips_delta[2]])
            cur_xz = np.array([delta[0], delta[2]])
            move = (target - cur_xz) * k
            delta[0] += move[0]
            delta[2] += move[1]
            for s in ("Left", "Right"):
                if self._lock[s] is not None:
                    self._lock[s] = self._lock[s] + move
        # safety: cap the per-frame hips XZ change so any residual
        # discontinuity becomes a short slide instead of a pop
        step = np.array([delta[0], delta[2]]) - prev_xz
        n = float(np.linalg.norm(step))
        if n > self.max_step_m:
            step = step * (self.max_step_m / n)
            delta[0] = prev_xz[0] + step[0]
            delta[2] = prev_xz[1] + step[1]
            # locks are left as they are: the planted foot slides for this
            # frame and the hips finish converging on the next ones
        self._hips_xz = (delta[0], delta[2])

        # --- flatten planted feet (yaw-only sole) ---------------------
        out = dict(rotations)
        for s in ("Left", "Right"):
            target_w = 1.0 if planted[s] else 0.0
            if dt > 0:
                k = min(1.0, dt / max(self.flatten_blend_sec, 1e-3))
                self._plant_w[s] += (target_w - self._plant_w[s]) * k
            w = self._plant_w[s]
            if w <= 1e-3:
                continue
            lower_g = fk[s][f"{s}LowerLeg"][1]
            foot_g = fk[s][f"{s}Foot"][1]
            fwd = _rot(foot_g, np.array([0.0, 0.0, 1.0]))
            yaw = math.atan2(float(fwd[0]), float(fwd[2]))
            flat_g = quat_from_axis_angle(_UP, yaw)
            flat_local = quat_mul(quat_inv(lower_g), flat_g)
            cur_local = _q(rotations, f"{s}Foot")
            out[f"{s}Foot"] = quat_slerp(cur_local, flat_local, w)
        return delta, out

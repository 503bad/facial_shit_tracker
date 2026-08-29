"""Smoothing filters for tracking data.

The One Euro filter is used for positional / scalar data because it adapts its
cutoff to signal speed: strong smoothing at rest (no jitter) with low lag on
fast motion.  Quaternions are smoothed with a slerp-based exponential filter.

A single user-facing "strength" value in [0, 1] is mapped onto the filter
parameters so the GUI can expose one slider per data stream.
"""

from __future__ import annotations

import math
import time

import numpy as np


# ---------------------------------------------------------------------------
# One Euro filter (Casiez et al. 2012), vectorised over numpy arrays.
# ---------------------------------------------------------------------------

class _LowPass:
    def __init__(self) -> None:
        self.y: np.ndarray | None = None

    def apply(self, x: np.ndarray, alpha: np.ndarray | float) -> np.ndarray:
        if self.y is None:
            self.y = np.asarray(x, dtype=np.float64).copy()
        else:
            self.y = self.y + alpha * (x - self.y)
        return self.y

    def reset(self) -> None:
        self.y = None


def _smoothing_alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Vectorised One Euro filter for an array of scalar channels."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()
        self._t_prev = None

    def apply(self, x: np.ndarray, t: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if t is None:
            t = time.perf_counter()
        if self._t_prev is None:
            dt = 1.0 / 60.0
        else:
            dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        prev = self._x.y
        dx = np.zeros_like(x) if prev is None else (x - prev) / dt
        edx = self._dx.apply(dx, _smoothing_alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        return self._x.apply(x, _smoothing_alpha(cutoff, dt))


class SmoothedChannels:
    """One Euro filter bank whose aggressiveness is set by strength in [0,1].

    strength 0.0 -> pass-through (no smoothing)
    strength 1.0 -> very strong smoothing (low min_cutoff, small beta)
    """

    # min_cutoff sweeps log-linearly from 10 Hz (barely any smoothing) down
    # to 0.15 Hz (heavy); beta shrinks so fast motion stays responsive at
    # moderate strengths but is also damped at maximum strength.
    _CUTOFF_HI = 10.0
    _CUTOFF_LO = 0.15
    _BETA_HI = 1.0
    _BETA_LO = 0.005

    def __init__(self, strength: float = 0.5) -> None:
        self._filter = OneEuroFilter()
        self._strength = -1.0
        self.set_strength(strength)

    def set_strength(self, strength: float) -> None:
        strength = float(np.clip(strength, 0.0, 1.0))
        if strength == self._strength:
            return
        self._strength = strength
        log_hi, log_lo = math.log(self._CUTOFF_HI), math.log(self._CUTOFF_LO)
        self._filter.min_cutoff = math.exp(log_hi + (log_lo - log_hi) * strength)
        blog_hi, blog_lo = math.log(self._BETA_HI), math.log(self._BETA_LO)
        self._filter.beta = math.exp(blog_hi + (blog_lo - blog_hi) * strength)

    @property
    def strength(self) -> float:
        return self._strength

    def apply(self, values: np.ndarray, t: float | None = None) -> np.ndarray:
        if self._strength <= 0.0:
            self._filter.reset()
            return np.asarray(values, dtype=np.float64)
        return self._filter.apply(values, t)

    def reset(self) -> None:
        self._filter.reset()


# ---------------------------------------------------------------------------
# Quaternion smoothing (slerp toward the new sample).
# ---------------------------------------------------------------------------

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Slerp between quaternions given as (x, y, z, w)."""
    q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
    q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + t * (q1 - q0))
    theta0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta0 * t
    q2 = _quat_normalize(q1 - q0 * dot)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


class QuaternionSmoother:
    """Adaptive slerp filter: blend factor derives from strength and dt.

    Uses a time-constant formulation so behaviour is frame-rate independent:
    blend = 1 - exp(-dt / tau), tau mapped from strength.
    """

    _TAU_LO = 0.0     # strength 0 -> instant
    _TAU_HI = 0.45    # strength 1 -> ~0.45 s time constant

    def __init__(self, strength: float = 0.5) -> None:
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self._q: np.ndarray | None = None
        self._t_prev: float | None = None

    def set_strength(self, strength: float) -> None:
        self.strength = float(np.clip(strength, 0.0, 1.0))

    def reset(self) -> None:
        self._q = None
        self._t_prev = None

    def apply(self, q: np.ndarray, t: float | None = None) -> np.ndarray:
        q = _quat_normalize(np.asarray(q, dtype=np.float64))
        if t is None:
            t = time.perf_counter()
        if self._q is None or self.strength <= 0.0:
            self._q = q
            self._t_prev = t
            return q
        dt = max(t - (self._t_prev or t), 1e-6)
        self._t_prev = t
        tau = self._TAU_LO + (self._TAU_HI - self._TAU_LO) * (self.strength ** 1.5)
        if tau < 1e-6:
            self._q = q
            return q
        blend = 1.0 - math.exp(-dt / tau)
        self._q = quat_slerp(self._q, q, blend)
        return self._q

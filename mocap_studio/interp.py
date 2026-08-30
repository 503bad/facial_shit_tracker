"""Output-stage snapshot interpolation.

Tracking produces samples at the camera / body-backend rate; when enabled,
an output thread resamples those streams at a fixed rate (e.g. 60 Hz) by
interpolating between the two samples that bracket ``now - delay``.
The delay adapts to each stream's measured interval so a bracketing pair
almost always exists; nothing is extrapolated, so there is no overshoot.

Sample kinds:
  "bones": {name: (pos, quat)}         -> quats slerped, pos taken from newer
  "blend": {name: float}               -> lerped
  "ifm":   (bs, head_e, head_p, reye, leye) -> all lerped
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from .smoothing import quat_slerp

_MAX_SAMPLES = 8
_STALE_SEC = 1.0
_MIN_DELAY = 0.020
_MAX_DELAY = 0.250


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _lerp_dict(a: dict, b: dict, u: float) -> dict:
    out = dict(b)
    for k, va in a.items():
        if k in b:
            out[k] = _lerp(va, b[k], u)
    return out


def _lerp_tuple(a, b, u: float):
    return tuple(_lerp(x, y, u) for x, y in zip(a, b))


def _interp_bones(a: dict, b: dict, u: float) -> dict:
    out = dict(b)
    for name, (pos_a, q_a) in a.items():
        if name in b:
            pos_b, q_b = b[name]
            q = quat_slerp(np.asarray(q_a), np.asarray(q_b), u)
            out[name] = (pos_b, (float(q[0]), float(q[1]),
                                 float(q[2]), float(q[3])))
    return out


def _interp_ifm(a, b, u: float):
    bs_a, he_a, hp_a, re_a, le_a = a
    bs_b, he_b, hp_b, re_b, le_b = b
    return (_lerp_dict(bs_a, bs_b, u), _lerp_tuple(he_a, he_b, u),
            _lerp_tuple(hp_a, hp_b, u), _lerp_tuple(re_a, re_b, u),
            _lerp_tuple(le_a, le_b, u))


_INTERP = {"bones": _interp_bones, "blend": _lerp_dict, "ifm": _interp_ifm}


class _Stream:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.samples: deque = deque(maxlen=_MAX_SAMPLES)  # (t, sample)
        self.interval = 1.0 / 30.0  # EMA of push spacing

    def push(self, t: float, sample) -> None:
        if self.samples:
            dt = t - self.samples[-1][0]
            if 0.0 < dt < 1.0:
                self.interval = 0.8 * self.interval + 0.2 * dt
        self.samples.append((t, sample))

    @property
    def delay(self) -> float:
        return float(np.clip(self.interval * 1.25 + 0.005,
                             _MIN_DELAY, _MAX_DELAY))

    def sample_at(self, now: float):
        """Interpolated sample for render time now - delay, or None."""
        if not self.samples:
            return None
        t_latest = self.samples[-1][0]
        if now - t_latest > _STALE_SEC:
            return None
        t_r = now - self.delay
        prev = None
        for t, s in self.samples:
            if t >= t_r:
                if prev is None:
                    return s  # render time is before our oldest sample
                t0, s0 = prev
                u = (t_r - t0) / max(t - t0, 1e-6)
                return _INTERP[self.kind](s0, s, float(np.clip(u, 0.0, 1.0)))
            prev = (t, s)
        return self.samples[-1][1]  # render time beyond latest: hold


class OutputInterpolator:
    """Collects timestamped samples and re-emits them at a fixed rate."""

    def __init__(self, fps: int, sinks: dict) -> None:
        """sinks: {"bones": fn(bones), "blend": fn(vals), "ifm": fn(*ifm)}"""
        self.fps = max(1, int(fps))
        self._sinks = sinks
        self._streams = {k: _Stream(k) for k in _INTERP}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def push(self, kind: str, t: float, sample) -> None:
        with self._lock:
            self._streams[kind].push(t, sample)

    def _loop(self) -> None:
        period = 1.0 / self.fps
        next_t = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            with self._lock:
                outs = {k: st.sample_at(now)
                        for k, st in self._streams.items()}
            for kind, sample in outs.items():
                sink = self._sinks.get(kind)
                if sample is None or sink is None:
                    continue
                try:
                    if kind == "ifm":
                        sink(*sample)
                    else:
                        sink(sample)
                except Exception:
                    pass
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

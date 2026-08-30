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
_REFINE_MAX_SAMPLES = 64
_REFINE_MIN_SAMPLES = 3


def _robust_kernel_mean(dts: np.ndarray, X: np.ndarray, sigma: float
                        ) -> np.ndarray:
    """Outlier-resistant kernel regression at dt = 0.

    dts: (n,) sample times relative to the render time; X: (n, d) values
    (NaN = channel absent in that sample).  Weights are a Gaussian in time
    times a robustness factor that nearly zeroes samples deviating from the
    per-channel median by more than 3 MAD - a one-or-two-frame spike is
    thereby ignored while genuine motion (consistent across neighbours)
    passes.
    """
    wt = np.exp(-0.5 * (dts / max(sigma, 1e-3)) ** 2)[:, None]  # (n,1)
    valid = ~np.isnan(X)
    med = np.nanmedian(X, axis=0)                                # (d,)
    dev = np.abs(X - med)
    mad = np.nanmedian(dev, axis=0) + 1e-6
    robust = np.where(dev <= 3.0 * mad + 1e-3, 1.0, 0.05)
    w = np.where(valid, wt * robust, 0.0)
    num = np.nansum(np.where(valid, X, 0.0) * w, axis=0)
    den = w.sum(axis=0)
    out = np.where(den > 1e-9, num / np.maximum(den, 1e-9), med)
    return out


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
    def __init__(self, kind: str, lookahead: float = 0.0) -> None:
        self.kind = kind
        self.lookahead = float(lookahead)
        self.samples: deque = deque(
            maxlen=_REFINE_MAX_SAMPLES if lookahead > 0 else _MAX_SAMPLES)
        self.interval = 1.0 / 30.0  # EMA of push spacing

    def push(self, t: float, sample) -> None:
        if self.samples:
            dt = t - self.samples[-1][0]
            if 0.0 < dt < 1.0:
                self.interval = 0.8 * self.interval + 0.2 * dt
        self.samples.append((t, sample))

    @property
    def delay(self) -> float:
        d = float(np.clip(self.interval * 1.25 + 0.005,
                          _MIN_DELAY, _MAX_DELAY))
        return max(d, self.lookahead)

    # ---- look-ahead refinement -------------------------------------
    def _window(self, t_r: float):
        return [(t - t_r, s) for t, s in self.samples
                if abs(t - t_r) <= self.lookahead]

    def _refine_bones(self, win):
        names = sorted({n for _, s in win for n in s})
        idx = {n: i for i, n in enumerate(names)}
        n, k = len(win), len(names)
        X = np.full((n, k, 4), np.nan)
        dts = np.array([dt for dt, _ in win])
        nearest = int(np.argmin(np.abs(dts)))
        ref = win[nearest][1]
        for si, (_, s) in enumerate(win):
            for name, (pos, q) in s.items():
                qv = np.asarray(q, dtype=np.float64)
                if name in ref and float(np.dot(qv, ref[name][1])) < 0:
                    qv = -qv  # hemisphere-align to the reference sample
                X[si, idx[name]] = qv
        out = _robust_kernel_mean(dts, X.reshape(n, k * 4),
                                  self.lookahead / 2.0).reshape(k, 4)
        result = {}
        for name in names:
            q = out[idx[name]]
            nrm = np.linalg.norm(q)
            q = q / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0])
            pos = ref[name][0] if name in ref else next(
                s[name][0] for _, s in win if name in s)
            result[name] = (pos, (float(q[0]), float(q[1]),
                                  float(q[2]), float(q[3])))
        return result

    def _refine_dict(self, win, dicts):
        keys = sorted({kk for d in dicts for kk in d})
        idx = {kk: i for i, kk in enumerate(keys)}
        X = np.full((len(win), len(keys)), np.nan)
        for si, d in enumerate(dicts):
            for kk, v in d.items():
                X[si, idx[kk]] = v
        dts = np.array([dt for dt, _ in win])
        out = _robust_kernel_mean(dts, X, self.lookahead / 2.0)
        return {kk: float(out[idx[kk]]) for kk in keys}

    def _refine_ifm(self, win):
        bs = self._refine_dict(win, [s[0] for _, s in win])
        vec = np.array([[*s[1], *s[2], *s[3], *s[4]] for _, s in win],
                       dtype=np.float64)
        dts = np.array([dt for dt, _ in win])
        v = _robust_kernel_mean(dts, vec, self.lookahead / 2.0)
        return (bs, tuple(v[0:3]), tuple(v[3:6]), tuple(v[6:9]),
                tuple(v[9:12]))

    def _refined_at(self, t_r: float):
        win = self._window(t_r)
        if len(win) < _REFINE_MIN_SAMPLES:
            return None
        if self.kind == "bones":
            return self._refine_bones(win)
        if self.kind == "blend":
            return self._refine_dict(win, [s for _, s in win])
        return self._refine_ifm(win)

    def sample_at(self, now: float):
        """Interpolated sample for render time now - delay, or None."""
        if not self.samples:
            return None
        t_latest = self.samples[-1][0]
        if now - t_latest > _STALE_SEC:
            return None
        t_r = now - self.delay
        if self.lookahead > 0:
            refined = self._refined_at(t_r)
            if refined is not None:
                return refined
            # too few samples in the window: fall back to interpolation
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

    def __init__(self, fps: int, sinks: dict, lookahead: float = 0.0) -> None:
        """sinks: {"bones": fn(bones), "blend": fn(vals), "ifm": fn(*ifm)}
        lookahead > 0 enables look-ahead refinement: output is delayed by
        that much and each frame is a robust kernel average of the samples
        within +-lookahead (spikes removed, jitter smoothed)."""
        self.fps = max(1, int(fps))
        self._sinks = sinks
        self._streams = {k: _Stream(k, lookahead) for k in _INTERP}
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

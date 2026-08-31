"""Timestamped webcam capture for the stereo extension.

Same latest-frame-wins pattern as the app's ``Camera``, plus a receive
timestamp (``time.perf_counter()``) recorded the moment each frame is
read.  The timestamp is an *estimate of capture time by receive time*
(exposure/USB/decode latency remains); pairing tolerances must budget
for that.

This class is owned entirely by the stereo extension: it is only ever
constructed for camera B (and temporarily inside the calibration
dialog), never for resources the legacy path is using.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class TimedCamera:
    def __init__(self, index: int, width: int = 1280, height: int = 720,
                 fps: int = 30) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._ts = 0.0
        self._running = False
        self.last_error: str | None = None

    def start(self) -> bool:
        self.stop()
        cap = cv2.VideoCapture(self.index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            self.last_error = f"カメラ {self.index} を開けませんでした"
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    @property
    def actual_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (self.width, self.height)
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def _loop(self) -> None:
        while self._running and self._cap is not None:
            ok, frame = self._cap.read()
            ts = time.perf_counter()
            if not ok:
                self.last_error = "カメラからのフレーム取得に失敗しました"
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._ts = ts

    def latest(self) -> tuple[np.ndarray | None, int, float]:
        """(frame BGR, frame_id, receive timestamp [perf_counter s])."""
        with self._lock:
            return self._frame, self._frame_id, self._ts

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None

"""Webcam capture on a background thread with latest-frame semantics."""

from __future__ import annotations

import threading

import cv2
import numpy as np


def enumerate_cameras(max_index: int = 8) -> list[int]:
    """Probe camera indices that can be opened (MSMF backend on Windows)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


class Camera:
    """Continuously grabs frames; consumers take the most recent one."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720,
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
            if not ok:
                self.last_error = "カメラからのフレーム取得に失敗しました"
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def latest(self) -> tuple[np.ndarray | None, int]:
        """Return (frame BGR, frame_id). frame_id increments per new frame."""
        with self._lock:
            return self._frame, self._frame_id

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame = None

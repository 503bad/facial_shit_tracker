"""iFacialMocap v1 UDP sender.

Sends face tracking packets in the legacy "v1" text format understood by
VSeeFace / VMagicMirror / Warudo etc.  See docs/ifacialmocap_v1.md.

We stream directly to the configured host:port (proven to work without a
handshake) and additionally answer handshakes arriving on our local 49983.
"""

from __future__ import annotations

import socket
import threading

HANDSHAKE = b"iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719"

IFM_BLENDSHAPES = [
    "browInnerUp", "browDown_L", "browDown_R",
    "browOuterUp_L", "browOuterUp_R",
    "eyeLookUp_L", "eyeLookUp_R", "eyeLookDown_L", "eyeLookDown_R",
    "eyeLookIn_L", "eyeLookIn_R", "eyeLookOut_L", "eyeLookOut_R",
    "eyeBlink_L", "eyeBlink_R", "eyeSquint_L", "eyeSquint_R",
    "eyeWide_L", "eyeWide_R",
    "cheekPuff", "cheekSquint_L", "cheekSquint_R",
    "noseSneer_L", "noseSneer_R",
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthRollUpper", "mouthRollLower", "mouthShrugUpper", "mouthShrugLower",
    "mouthClose", "mouthSmile_L", "mouthSmile_R",
    "mouthFrown_L", "mouthFrown_R", "mouthDimple_L", "mouthDimple_R",
    "mouthUpperUp_L", "mouthUpperUp_R",
    "mouthLowerDown_L", "mouthLowerDown_R",
    "mouthPress_L", "mouthPress_R",
    "mouthStretch_L", "mouthStretch_R", "tongueOut",
]


def build_packet(blendshapes: dict[str, float],
                 head_euler_deg: tuple[float, float, float],
                 head_pos_m: tuple[float, float, float],
                 right_eye_deg: tuple[float, float, float],
                 left_eye_deg: tuple[float, float, float]) -> bytes:
    """Build one v1 datagram. blendshape values are 0.0-1.0 floats."""
    parts = []
    for name in IFM_BLENDSHAPES:
        v = int(round(max(0.0, min(1.0, blendshapes.get(name, 0.0))) * 100))
        parts.append(f"{name}-{v}")
    head = ",".join(f"{v:.6f}" for v in (*head_euler_deg, *head_pos_m))
    reye = ",".join(f"{v:.6f}" for v in right_eye_deg)
    leye = ",".join(f"{v:.6f}" for v in left_eye_deg)
    packet = "|".join(parts) + f"|=head#{head}|rightEye#{reye}|leftEye#{leye}|"
    return packet.encode("ascii")


class IfmSender:
    """UDP sender with optional handshake listener on local port 49983."""

    def __init__(self, host: str = "127.0.0.1", port: int = 49983) -> None:
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._handshake_targets: set[tuple[str, int]] = set()
        self._listener: socket.socket | None = None
        self._listen_thread: threading.Thread | None = None
        self._running = False
        self._start_listener()

    def _start_listener(self) -> None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("0.0.0.0", 49983))
            s.settimeout(0.5)
        except OSError:
            return  # port busy (receiver on same machine) - direct send only
        self._listener = s
        self._running = True
        self._listen_thread = threading.Thread(target=self._listen_loop,
                                               daemon=True)
        self._listen_thread.start()

    def _listen_loop(self) -> None:
        while self._running and self._listener is not None:
            try:
                data, addr = self._listener.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.startswith(HANDSHAKE):
                with self._lock:
                    self._handshake_targets.add((addr[0], 49983))

    def set_destination(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def send(self, blendshapes: dict[str, float],
             head_euler_deg, head_pos_m,
             right_eye_deg, left_eye_deg) -> None:
        packet = build_packet(blendshapes, head_euler_deg, head_pos_m,
                              right_eye_deg, left_eye_deg)
        targets = {(self.host, self.port)}
        with self._lock:
            targets |= self._handshake_targets
        for t in targets:
            try:
                self._sock.sendto(packet, t)
            except OSError:
                pass

    def close(self) -> None:
        self._running = False
        if self._listener is not None:
            self._listener.close()
            self._listener = None

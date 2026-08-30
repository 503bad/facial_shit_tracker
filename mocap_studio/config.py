"""Application settings with JSON persistence next to the package."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "settings.json"


@dataclass
class Settings:
    # Camera
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30
    mirror_preview: bool = True
    mirror_tracking: bool = True

    # Face -> iFacialMocap v1 (UDP sender)
    face_enabled: bool = True
    face_host: str = "127.0.0.1"
    face_port: int = 49983
    face_output: str = "ifm"             # "ifm" | "vmc" | "both"
    eye_mode: str = "bone"               # VMC eyes: "bone" | "morph" | "both"
    face_smoothing: float = 0.3          # expression blendshapes
    head_smoothing: float = 0.35         # head pose

    # Body -> VMC protocol (OSC sender)
    body_enabled: bool = True
    vmc_host: str = "127.0.0.1"
    vmc_port: int = 39539
    body_smoothing: float = 0.5          # arm / spine bones
    finger_smoothing: float = 0.6        # finger curl estimation
    body_gate_enabled: bool = True       # micro-tremor suppression gate
    body_gate_deg: float = 2.0           # gate threshold in degrees
    show_camera: bool = True             # camera image in the preview
    send_legs: bool = False              # lower-body (legs) over VMC
    ground_mode: bool = False            # foot lock / grounding (legs on)
    body_mode_high_quality: bool = True  # NVAR body pose mode 0 (HQ) vs 1

    # Send rate limits (Hz)
    face_send_rate: int = 60
    body_send_rate: int = 60
    # Output-stage interpolation: resample all streams at output_fps with
    # a small adaptive delay (no change to tracking when off).
    output_interp: bool = False
    output_fps: int = 60
    # Look-ahead refinement (needs output_interp): delay output by
    # output_lookahead_sec and robustly average the samples around each
    # frame - removes spikes/jitter at the cost of that latency.
    output_refine: bool = False
    output_lookahead_sec: float = 0.3

    # Optional VRM model for accurate bone bind offsets in VMC output
    vrm_model_path: str = ""

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(
            json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Settings":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

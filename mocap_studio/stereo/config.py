"""Settings for the 2-camera depth extension.

Persisted in their own files next to ``settings.json`` so the main
application settings are never touched by this feature:

  * ``stereo_settings.json``     - user options (default: feature OFF)
  * ``stereo_calibration.json``  - calibration profile (see calibration.py)
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
STEREO_CONFIG_PATH = _ROOT / "stereo_settings.json"
CALIBRATION_PATH = _ROOT / "stereo_calibration.json"
LOG_DIR = _ROOT / "stereo_logs"


@dataclass
class StereoSettings:
    # Master switch. Default OFF: with this False the stereo package is
    # never initialised and the app behaves exactly as before.
    enabled: bool = False

    # Second camera (camera A is the main app's camera).
    camera_b_index: int = 1
    camera_b_width: int = 1280
    camera_b_height: int = 720
    camera_b_fps: int = 30

    # Nominal optics used to build approximate intrinsics.  Applied to
    # both cameras (same model assumed); refined only by calibration.
    fov_deg: float = 120.0
    fov_is_diagonal: bool = False
    baseline_m: float = 0.30

    # Pairing / fusion tuning.
    pair_tolerance_ms: float = 15.0   # max |ts_A - ts_B| for a stereo pair
    min_visibility: float = 0.5       # per-joint MediaPipe visibility gate
    max_epipolar_px: float = 8.0      # triangulation residual gate
    recovery_frames: int = 4          # consistent obs needed to re-accept
    predict_timeout_ms: float = 220.0  # predict this long, then LOST
    recover_blend_ms: float = 180.0   # output ramp after re-acceptance
    stereo_weight: float = 1.0        # 0..1 blend stereo vs mono positions

    # Output options.
    send_depth_translation: bool = True  # hips Z from stereo depth
    show_pip: bool = True                # camera-B picture-in-picture
    debug_log: bool = False              # JSONL observation log for replay

    def save(self, path: Path = STEREO_CONFIG_PATH) -> None:
        path.write_text(
            json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = STEREO_CONFIG_PATH) -> "StereoSettings":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

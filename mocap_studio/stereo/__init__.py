"""2-camera markerless depth extension (optional, default OFF).

Everything stereo lives in this package.  The host app touches it only
through small, failure-isolated hooks:

  * ``tracker.py`` swaps its body-observation source to ``StereoEngine``
    while the feature is enabled (legacy path untouched when OFF).
  * ``body_pipeline.py`` consumes the optional ``"_stereo_hips_z"`` pose
    key for depth translation (inert when the key is absent).
  * ``gui.py`` adds the settings panel from ``stereo.ui``.

No heavy imports at package level - MediaPipe/OpenCV structures are only
constructed when the feature is actually started.
"""

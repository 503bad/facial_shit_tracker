"""VMC protocol (OSC/UDP) sender for upper-body bone data.

Sends to a Marionette (VSeeFace, VMagicMirror, etc.) listening on UDP 39539
by default.  Bone rotations are LOCAL (parent-relative), quaternion order
(x, y, z, w), Unity left-handed Y-up coordinates in meters.

Spec: https://protocol.vmc.info/specification  (see docs/vmc_protocol.md)
"""

from __future__ import annotations

import time

from pythonosc.osc_bundle_builder import IMMEDIATELY, OscBundleBuilder
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.udp_client import SimpleUDPClient

# Unity HumanBodyBones names (subset order doesn't matter; one msg per bone).
HUMAN_BONES = [
    "Hips", "Spine", "Chest", "UpperChest", "Neck", "Head",
    "LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
    "RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",
]
FINGER_BONES = [
    f"{side}{finger}{seg}"
    for side in ("Left", "Right")
    for finger in ("Thumb", "Index", "Middle", "Ring", "Little")
    for seg in ("Proximal", "Intermediate", "Distal")
]

_MAX_BONES_PER_BUNDLE = 20  # keep each UDP packet under ~1500 bytes
_MAX_BLENDS_PER_BUNDLE = 30


def _bone_msg(name: str, pos, quat) -> OscMessageBuilder:
    b = OscMessageBuilder("/VMC/Ext/Bone/Pos")
    b.add_arg(name, "s")
    for v in pos:
        b.add_arg(float(v), "f")
    for v in quat:  # (x, y, z, w)
        b.add_arg(float(v), "f")
    return b


class VmcSender:
    """Assembles per-frame OSC bundles and sends them over UDP."""

    def __init__(self, host: str = "127.0.0.1", port: int = 39539) -> None:
        self._client = SimpleUDPClient(host, port)
        self._t0 = time.perf_counter()
        self.host = host
        self.port = port

    def set_destination(self, host: str, port: int) -> None:
        if host != self.host or port != self.port:
            self._client = SimpleUDPClient(host, port)
            self.host = host
            self.port = port

    def send_blendshapes(self, values: dict[str, float]) -> None:
        """Send /VMC/Ext/Blend/Val for every entry, then /VMC/Ext/Blend/Apply.

        Names are sent verbatim (e.g. ARKit "eyeBlinkLeft" for Perfect Sync
        receivers that match morph-target names, such as VRM4U).
        """
        items = list(values.items())
        for i in range(0, len(items), _MAX_BLENDS_PER_BUNDLE):
            bundle = OscBundleBuilder(IMMEDIATELY)
            for name, v in items[i:i + _MAX_BLENDS_PER_BUNDLE]:
                m = OscMessageBuilder("/VMC/Ext/Blend/Val")
                m.add_arg(name, "s")
                m.add_arg(float(max(0.0, min(1.0, v))), "f")
                bundle.add_content(m.build())
            self._client.send(bundle.build())
        tail = OscBundleBuilder(IMMEDIATELY)
        tail.add_content(OscMessageBuilder("/VMC/Ext/Blend/Apply").build())
        self._client.send(tail.build())

    def send_frame(self, bones: dict[str, tuple[tuple, tuple]],
                   root_pos=(0.0, 0.0, 0.0),
                   root_quat=(0.0, 0.0, 0.0, 1.0),
                   tracking_ok: bool = True) -> None:
        """Send one frame.

        bones: {bone_name: ((px, py, pz), (qx, qy, qz, qw))} local transforms.
        """
        head = OscBundleBuilder(IMMEDIATELY)

        ok = OscMessageBuilder("/VMC/Ext/OK")
        for v in (1, 3, 0, 1 if tracking_ok else 0):
            ok.add_arg(v, "i")
        head.add_content(ok.build())

        t = OscMessageBuilder("/VMC/Ext/T")
        t.add_arg(time.perf_counter() - self._t0, "f")
        head.add_content(t.build())

        root = OscMessageBuilder("/VMC/Ext/Root/Pos")
        root.add_arg("root", "s")
        for v in (*root_pos, *root_quat, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0):
            root.add_arg(float(v), "f")
        head.add_content(root.build())
        self._client.send(head.build())

        items = list(bones.items())
        for i in range(0, len(items), _MAX_BONES_PER_BUNDLE):
            bundle = OscBundleBuilder(IMMEDIATELY)
            for name, (pos, quat) in items[i:i + _MAX_BONES_PER_BUNDLE]:
                bundle.add_content(_bone_msg(name, pos, quat).build())
            self._client.send(bundle.build())

        tail = OscBundleBuilder(IMMEDIATELY)
        apply_ = OscMessageBuilder("/VMC/Ext/Blend/Apply")
        tail.add_content(apply_.build())
        self._client.send(tail.build())

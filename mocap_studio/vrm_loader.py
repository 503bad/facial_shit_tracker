"""Minimal VRM (glTF binary) reader.

Extracts, for VMC transmission:
- humanoid bone -> bind-pose local translation (meters, glTF right-handed;
  converted to Unity left-handed by negating X is NOT needed for VRM because
  VRM models are stored with a 180-degree Y rotation convention — normalized
  VRM 1.0 exports already match Unity local offsets closely enough that VMC
  receivers, which re-target by bone name, only need plausible offsets).
- which optional bones exist (UpperChest, Jaw, fingers...).

Also detects Perfect Sync capability (52 ARKit morph targets on any mesh).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# VRM humanoid bone name (VRM1/camelCase) -> Unity HumanBodyBones name.
_VRM_TO_UNITY = {
    "hips": "Hips", "spine": "Spine", "chest": "Chest",
    "upperChest": "UpperChest", "neck": "Neck", "head": "Head",
    "jaw": "Jaw", "leftEye": "LeftEye", "rightEye": "RightEye",
    "leftShoulder": "LeftShoulder", "leftUpperArm": "LeftUpperArm",
    "leftLowerArm": "LeftLowerArm", "leftHand": "LeftHand",
    "rightShoulder": "RightShoulder", "rightUpperArm": "RightUpperArm",
    "rightLowerArm": "RightLowerArm", "rightHand": "RightHand",
    "leftUpperLeg": "LeftUpperLeg", "leftLowerLeg": "LeftLowerLeg",
    "leftFoot": "LeftFoot", "leftToes": "LeftToes",
    "rightUpperLeg": "RightUpperLeg", "rightLowerLeg": "RightLowerLeg",
    "rightFoot": "RightFoot", "rightToes": "RightToes",
    # VRM1 thumb naming -> Unity naming
    "leftThumbMetacarpal": "LeftThumbProximal",
    "leftThumbProximal": "LeftThumbIntermediate",
    "leftThumbDistal": "LeftThumbDistal",
    "rightThumbMetacarpal": "RightThumbProximal",
    "rightThumbProximal": "RightThumbIntermediate",
    "rightThumbDistal": "RightThumbDistal",
    # VRM0 thumb naming (same Unity result)
    "leftThumbIntermediate": "LeftThumbIntermediate",
    "rightThumbIntermediate": "RightThumbIntermediate",
}
for _side in ("left", "right"):
    for _finger in ("Index", "Middle", "Ring", "Little"):
        for _seg in ("Proximal", "Intermediate", "Distal"):
            _VRM_TO_UNITY[f"{_side}{_finger}{_seg}"] = (
                _side.capitalize() + _finger + _seg)

ARKIT_BLENDSHAPES = [
    "browInnerUp", "browDownLeft", "browDownRight",
    "browOuterUpLeft", "browOuterUpRight",
    "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "noseSneerLeft", "noseSneerRight",
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthFunnel", "mouthPucker", "mouthLeft", "mouthRight",
    "mouthRollUpper", "mouthRollLower", "mouthShrugUpper", "mouthShrugLower",
    "mouthClose", "mouthSmileLeft", "mouthSmileRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthStretchLeft", "mouthStretchRight", "tongueOut",
]


class VrmInfo:
    def __init__(self) -> None:
        self.spec_version: str = ""
        # Unity bone name -> (x, y, z) bind local translation
        self.bone_offsets: dict[str, tuple[float, float, float]] = {}
        self.perfect_sync: bool = False
        self.model_name: str = ""

    @property
    def bones(self) -> set[str]:
        return set(self.bone_offsets)


def load_vrm(path: str | Path) -> VrmInfo:
    path = Path(path)
    with open(path, "rb") as f:
        magic, _ver, _length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:  # 'glTF'
            raise ValueError("glTFバイナリではありません")
        clen, ctype = struct.unpack("<II", f.read(8))
        if ctype != 0x4E4F534A:  # 'JSON'
            raise ValueError("JSONチャンクが見つかりません")
        gltf = json.loads(f.read(clen))

    info = VrmInfo()
    exts = gltf.get("extensions", {})
    nodes = gltf.get("nodes", [])

    if "VRMC_vrm" in exts:
        vrm = exts["VRMC_vrm"]
        info.spec_version = vrm.get("specVersion", "1.0")
        info.model_name = vrm.get("meta", {}).get("name", path.stem)
        human_bones = vrm.get("humanoid", {}).get("humanBones", {})
        items = [(k, v.get("node")) for k, v in human_bones.items()]
    elif "VRM" in exts:
        vrm = exts["VRM"]
        info.spec_version = "0.x"
        info.model_name = vrm.get("meta", {}).get("title", path.stem)
        items = [(b.get("bone"), b.get("node"))
                 for b in vrm.get("humanoid", {}).get("humanBones", [])]
    else:
        raise ValueError("VRM拡張が見つかりません")

    for vrm_name, node_idx in items:
        unity = _VRM_TO_UNITY.get(vrm_name)
        if unity is None or node_idx is None or node_idx >= len(nodes):
            continue
        t = nodes[node_idx].get("translation", [0.0, 0.0, 0.0])
        info.bone_offsets[unity] = (float(t[0]), float(t[1]), float(t[2]))

    # Perfect Sync: does any mesh carry (nearly) all ARKit morph targets?
    arkit = set(ARKIT_BLENDSHAPES)
    for mesh in gltf.get("meshes", []):
        names = mesh.get("extras", {}).get("targetNames")
        if not names:
            prim = (mesh.get("primitives") or [{}])[0]
            names = prim.get("extras", {}).get("targetNames")
        if names and len(arkit & set(names)) >= 40:
            info.perfect_sync = True
            break

    return info

"""ctypes bindings for the NVIDIA AR SDK (Maxine, v0.8.x).

Wraps nvARPose.dll / NVCVImage.dll with the two features this app needs:
FaceExpressions (53 expression coefficients + head pose) and
BodyPoseEstimation (34 3D keypoints).

All Set*/Get* calls register POINTERS with the SDK, so every buffer passed in
is kept referenced on the Python object for the feature's lifetime.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_ubyte, c_uint, c_uint8, c_ulong, c_void_p)

import numpy as np

SDK_DIR = os.environ.get(
    "NV_AR_SDK_PATH",
    r"C:\Program Files\NVIDIA Corporation\NVIDIA AR SDK")
MODEL_DIR = os.environ.get("NVAR_MODEL_DIR", os.path.join(SDK_DIR, "models"))

NVCV_SUCCESS = 0

# NvCVImage enums
NVCV_BGR = 5
NVCV_U8 = 1
NVCV_CHUNKY = 0
NVCV_CPU = 0
NVCV_GPU = 1

# Temporal filter bits (FaceExpressions)
NVAR_TEMPORAL_FILTER_FACE_BOX = 1 << 0
NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS = 1 << 1
NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE = 1 << 2
NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS = 1 << 4
NVAR_TEMPORAL_FILTER_FACIAL_GAZE = 1 << 5
NVAR_TEMPORAL_FILTER_ENHANCE_EXPRESSIONS = 1 << 8

EXPRESSION_COUNT = 53
NUM_LANDMARKS = 126
NUM_BODY_KEYPOINTS = 34

# FaceExpressions coefficient order (ExpressionApp _exprAbbr table).
EXPRESSION_NAMES = [
    "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R",
    "browOuterUp_L", "browOuterUp_R", "cheekPuff_L", "cheekPuff_R",
    "cheekSquint_L", "cheekSquint_R", "eyeBlink_L", "eyeBlink_R",
    "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R",
    "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R",
    "eyeSquint_L", "eyeSquint_R", "eyeWide_L", "eyeWide_R",
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R",
    "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R",
    "mouthPress_L", "mouthPress_R", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R",
    "mouthUpperUp_L", "mouthUpperUp_R", "noseSneer_L", "noseSneer_R",
]

# BodyPoseEstimation keypoint order (BodyTrack.cpp).
BODY_KEYPOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "torso", "left_knee", "right_knee",
    "neck", "left_ankle", "right_ankle", "left_big_toe", "right_big_toe",
    "left_small_toe", "right_small_toe", "left_heel", "right_heel", "nose",
    "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky_knuckle", "right_pinky_knuckle",
    "left_middle_tip", "right_middle_tip", "left_index_knuckle",
    "right_index_knuckle", "left_thumb_tip", "right_thumb_tip",
]
KP = {name: i for i, name in enumerate(BODY_KEYPOINT_NAMES)}


class NvCVImage(Structure):
    _fields_ = [
        ("width", c_uint), ("height", c_uint), ("pitch", c_int),
        ("pixelFormat", c_int), ("componentType", c_int),
        ("pixelBytes", c_ubyte), ("componentBytes", c_ubyte),
        ("numComponents", c_ubyte), ("planar", c_ubyte),
        ("gpuMem", c_ubyte), ("colorspace", c_ubyte),
        ("reserved", c_ubyte * 2),
        ("pixels", c_void_p), ("deletePtr", c_void_p),
        ("deleteProc", c_void_p), ("bufferBytes", ctypes.c_ulonglong),
    ]


class NvAR_Quaternion(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("w", c_float)]


class NvAR_Point2f(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]


class NvAR_Point3f(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]


class NvAR_Vector3f(Structure):
    _fields_ = [("vec", c_float * 3)]


class NvAR_Rect(Structure):
    _fields_ = [("x", c_float), ("y", c_float),
                ("width", c_float), ("height", c_float)]


class NvAR_BBoxes(Structure):
    _fields_ = [("boxes", POINTER(NvAR_Rect)),
                ("num_boxes", c_uint8), ("max_boxes", c_uint8)]


class NvArError(RuntimeError):
    pass


_nvcv = None
_nvar = None


def _load_libs():
    global _nvcv, _nvar
    if _nvar is not None:
        return
    if not os.path.isdir(SDK_DIR):
        raise NvArError(
            f"NVIDIA AR SDK が見つかりません: {SDK_DIR}\n"
            "SDKをインストールするか NV_AR_SDK_PATH を設定してください。")
    os.add_dll_directory(SDK_DIR)
    _nvcv = ctypes.CDLL(os.path.join(SDK_DIR, "NVCVImage.dll"))
    _nvar = ctypes.CDLL(os.path.join(SDK_DIR, "nvARPose.dll"))

    _nvcv.NvCV_GetErrorStringFromCode.restype = c_char_p
    _nvcv.NvCV_GetErrorStringFromCode.argtypes = [c_int]
    _nvcv.NvCVImage_Init.argtypes = [
        POINTER(NvCVImage), c_uint, c_uint, c_int, c_void_p,
        c_int, c_int, c_uint, c_uint]
    _nvcv.NvCVImage_Alloc.argtypes = [
        POINTER(NvCVImage), c_uint, c_uint, c_int, c_int,
        c_uint, c_uint, c_uint]
    _nvcv.NvCVImage_Dealloc.argtypes = [POINTER(NvCVImage)]
    _nvcv.NvCVImage_Transfer.argtypes = [
        POINTER(NvCVImage), POINTER(NvCVImage), c_float, c_void_p,
        POINTER(NvCVImage)]

    _nvar.NvAR_Create.argtypes = [c_char_p, POINTER(c_void_p)]
    _nvar.NvAR_Load.argtypes = [c_void_p]
    _nvar.NvAR_Run.argtypes = [c_void_p]
    _nvar.NvAR_Destroy.argtypes = [c_void_p]
    _nvar.NvAR_CudaStreamCreate.argtypes = [POINTER(c_void_p)]
    _nvar.NvAR_CudaStreamDestroy.argtypes = [c_void_p]
    _nvar.NvAR_SetU32.argtypes = [c_void_p, c_char_p, c_uint]
    _nvar.NvAR_SetF32.argtypes = [c_void_p, c_char_p, c_float]
    _nvar.NvAR_SetString.argtypes = [c_void_p, c_char_p, c_char_p]
    _nvar.NvAR_SetCudaStream.argtypes = [c_void_p, c_char_p, c_void_p]
    _nvar.NvAR_SetObject.argtypes = [c_void_p, c_char_p, c_void_p, c_ulong]
    _nvar.NvAR_SetF32Array.argtypes = [
        c_void_p, c_char_p, POINTER(c_float), c_int]
    _nvar.NvAR_GetU32.argtypes = [c_void_p, c_char_p, POINTER(c_uint)]
    _nvar.NvAR_GetObject.argtypes = [
        c_void_p, c_char_p, POINTER(c_void_p), c_ulong]


def _check(status: int, what: str) -> None:
    if status != NVCV_SUCCESS:
        msg = _nvcv.NvCV_GetErrorStringFromCode(status).decode(
            "utf-8", "replace") if _nvcv else "?"
        raise NvArError(f"{what} failed: {status} ({msg})")


class _CpuImageBridge:
    """Wraps numpy BGR frames as NvCVImage and transfers them to a GPU image."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.gpu = NvCVImage()
        _check(_nvcv.NvCVImage_Alloc(
            byref(self.gpu), width, height, NVCV_BGR, NVCV_U8,
            NVCV_CHUNKY, NVCV_GPU, 1), "NvCVImage_Alloc(GPU)")
        self._cpu = NvCVImage()
        self._frame_ref: np.ndarray | None = None

    def upload(self, frame_bgr: np.ndarray, stream: c_void_p) -> None:
        if (frame_bgr.shape[0] != self.height
                or frame_bgr.shape[1] != self.width):
            raise NvArError("フレームサイズが初期化時と異なります")
        frame_bgr = np.ascontiguousarray(frame_bgr)
        self._frame_ref = frame_bgr  # keep alive during Transfer
        _check(_nvcv.NvCVImage_Init(
            byref(self._cpu), self.width, self.height,
            frame_bgr.strides[0], frame_bgr.ctypes.data,
            NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU), "NvCVImage_Init")
        _check(_nvcv.NvCVImage_Transfer(
            byref(self._cpu), byref(self.gpu), 1.0, stream, None),
            "NvCVImage_Transfer")

    def destroy(self) -> None:
        _nvcv.NvCVImage_Dealloc(byref(self.gpu))


class FaceExpressionEstimator:
    """NvAR FaceExpressions: 53 coefficients + head pose from a BGR frame."""

    def __init__(self, width: int, height: int,
                 temporal: bool = True, pose_mode_6dof: bool = True,
                 enable_cheek_puff: bool = False) -> None:
        _load_libs()
        self._handle = c_void_p()
        self._stream = c_void_p()
        _check(_nvar.NvAR_Create(b"FaceExpressions", byref(self._handle)),
               "NvAR_Create(FaceExpressions)")
        _check(_nvar.NvAR_CudaStreamCreate(byref(self._stream)),
               "NvAR_CudaStreamCreate")
        h = self._handle
        _nvar.NvAR_SetString(h, b"NvAR_Parameter_Config_ModelDir",
                             MODEL_DIR.encode("utf-8"))
        _nvar.NvAR_SetCudaStream(h, b"NvAR_Parameter_Config_CUDAStream",
                                 self._stream)
        temporal_bits = 0
        if temporal:
            temporal_bits = (NVAR_TEMPORAL_FILTER_FACE_BOX
                             | NVAR_TEMPORAL_FILTER_FACIAL_LANDMARKS
                             | NVAR_TEMPORAL_FILTER_FACE_ROTATIONAL_POSE
                             | NVAR_TEMPORAL_FILTER_FACIAL_EXPRESSIONS
                             | NVAR_TEMPORAL_FILTER_FACIAL_GAZE)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_Temporal", temporal_bits)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_PoseMode",
                          1 if pose_mode_6dof else 0)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_EnableCheekPuff",
                          1 if enable_cheek_puff else 0)
        _check(_nvar.NvAR_Load(h), "NvAR_Load(FaceExpressions)")

        # I/O buffers — must stay referenced for the feature's lifetime.
        self._bridge = _CpuImageBridge(width, height)
        self._boxes = (NvAR_Rect * 25)()
        self._bboxes = NvAR_BBoxes(ctypes.cast(self._boxes,
                                               POINTER(NvAR_Rect)), 0, 25)
        self._landmarks = (NvAR_Point2f * NUM_LANDMARKS)()
        self._lm_conf = (c_float * NUM_LANDMARKS)()
        self._expr = (c_float * EXPRESSION_COUNT)()
        self._pose = NvAR_Quaternion(0, 0, 0, 1)
        self._trans = NvAR_Vector3f()
        self._intrinsics = (c_float * 3)(float(height), width / 2.0,
                                         height / 2.0)

        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Input_Image", byref(self._bridge.gpu),
            ctypes.sizeof(NvCVImage)), "SetObject(Input_Image)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_BoundingBoxes", byref(self._bboxes),
            ctypes.sizeof(NvAR_BBoxes)), "SetObject(BoundingBoxes)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_Landmarks", self._landmarks,
            ctypes.sizeof(NvAR_Point2f)), "SetObject(Landmarks)")
        _check(_nvar.NvAR_SetF32Array(
            h, b"NvAR_Parameter_Output_LandmarksConfidence", self._lm_conf,
            NUM_LANDMARKS), "SetF32Array(LandmarksConfidence)")
        _check(_nvar.NvAR_SetF32Array(
            h, b"NvAR_Parameter_Output_ExpressionCoefficients", self._expr,
            EXPRESSION_COUNT), "SetF32Array(ExpressionCoefficients)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_Pose", byref(self._pose),
            ctypes.sizeof(NvAR_Quaternion)), "SetObject(Pose)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_PoseTranslation", byref(self._trans),
            ctypes.sizeof(NvAR_Vector3f)), "SetObject(PoseTranslation)")
        _check(_nvar.NvAR_SetF32Array(
            h, b"NvAR_Parameter_Input_CameraIntrinsicParams",
            self._intrinsics, 3), "SetF32Array(CameraIntrinsicParams)")

    def process(self, frame_bgr: np.ndarray):
        """Run on one frame.

        Returns (found, expressions[53], pose_quat(x,y,z,w), translation(x,y,z),
        landmarks[126,2]) — expressions is a copy in EXPRESSION_NAMES order.
        """
        self._bridge.upload(frame_bgr, self._stream)
        _check(_nvar.NvAR_Run(self._handle), "NvAR_Run(FaceExpressions)")
        found = self._bboxes.num_boxes > 0
        expr = np.ctypeslib.as_array(self._expr).copy()
        pose = np.array([self._pose.x, self._pose.y, self._pose.z,
                         self._pose.w])
        trans = np.array(self._trans.vec[:])
        lm = np.array([(p.x, p.y) for p in self._landmarks])
        return found, expr, pose, trans, lm

    def destroy(self) -> None:
        if self._handle:
            _nvar.NvAR_Destroy(self._handle)
            self._handle = c_void_p()
        if self._stream:
            _nvar.NvAR_CudaStreamDestroy(self._stream)
            self._stream = c_void_p()
        self._bridge.destroy()


class BodyPoseEstimator:
    """NvAR BodyPoseEstimation: 34 keypoints (2D + 3D) from a BGR frame."""

    def __init__(self, width: int, height: int, high_quality: bool = True,
                 temporal: bool = True) -> None:
        _load_libs()
        self._handle = c_void_p()
        self._stream = c_void_p()
        _check(_nvar.NvAR_Create(b"BodyPoseEstimation", byref(self._handle)),
               "NvAR_Create(BodyPoseEstimation)")
        _check(_nvar.NvAR_CudaStreamCreate(byref(self._stream)),
               "NvAR_CudaStreamCreate")
        h = self._handle
        _nvar.NvAR_SetString(h, b"NvAR_Parameter_Config_ModelDir",
                             MODEL_DIR.encode("utf-8"))
        _nvar.NvAR_SetCudaStream(h, b"NvAR_Parameter_Config_CUDAStream",
                                 self._stream)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_BatchSize", 1)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_Mode",
                          0 if high_quality else 1)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_Temporal",
                          1 if temporal else 0)
        _nvar.NvAR_SetU32(h, b"NvAR_Parameter_Config_UseCudaGraph", 1)
        _check(_nvar.NvAR_Load(h), "NvAR_Load(BodyPoseEstimation)")

        self._bridge = _CpuImageBridge(width, height)
        n = NUM_BODY_KEYPOINTS
        self._kp2d = (NvAR_Point2f * n)()
        self._kp3d = (NvAR_Point3f * n)()
        self._joint_angles = (NvAR_Quaternion * n)()
        self._kp_conf = (c_float * n)()

        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Input_Image", byref(self._bridge.gpu),
            ctypes.sizeof(NvCVImage)), "SetObject(Input_Image)")
        _nvar.NvAR_SetF32(h, b"NvAR_Parameter_Input_FocalLength", 800.79041)
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_KeyPoints", self._kp2d,
            ctypes.sizeof(self._kp2d)), "SetObject(KeyPoints)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_KeyPoints3D", self._kp3d,
            ctypes.sizeof(self._kp3d)), "SetObject(KeyPoints3D)")
        _check(_nvar.NvAR_SetObject(
            h, b"NvAR_Parameter_Output_JointAngles", self._joint_angles,
            ctypes.sizeof(self._joint_angles)), "SetObject(JointAngles)")
        _check(_nvar.NvAR_SetF32Array(
            h, b"NvAR_Parameter_Output_KeyPointsConfidence", self._kp_conf,
            n), "SetF32Array(KeyPointsConfidence)")

        # Neutral reference pose (T-pose) for retargeting.
        ref_ptr = c_void_p()
        _check(_nvar.NvAR_GetObject(
            h, b"NvAR_Parameter_Config_ReferencePose", byref(ref_ptr),
            ctypes.sizeof(NvAR_Point3f)), "GetObject(ReferencePose)")
        ref = ctypes.cast(ref_ptr, POINTER(NvAR_Point3f * n)).contents
        self.reference_pose = np.array([(p.x, p.y, p.z) for p in ref])

    def process(self, frame_bgr: np.ndarray):
        """Run on one frame.

        Returns (keypoints3d[34,3], keypoints2d[34,2], confidence[34]).
        """
        self._bridge.upload(frame_bgr, self._stream)
        _check(_nvar.NvAR_Run(self._handle), "NvAR_Run(BodyPoseEstimation)")
        kp3d = np.array([(p.x, p.y, p.z) for p in self._kp3d])
        kp2d = np.array([(p.x, p.y) for p in self._kp2d])
        conf = np.ctypeslib.as_array(self._kp_conf).copy()
        return kp3d, kp2d, conf

    def destroy(self) -> None:
        if self._handle:
            _nvar.NvAR_Destroy(self._handle)
            self._handle = c_void_p()
        if self._stream:
            _nvar.NvAR_CudaStreamDestroy(self._stream)
            self._stream = c_void_p()
        self._bridge.destroy()

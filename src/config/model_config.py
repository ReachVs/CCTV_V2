import os
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class HardwareTarget(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    EDGE_TPU = "edge_tpu"
    JETSON_TENSORRT = "tensorrt"
    COREML = "coreml"
    ONNX = "onnx"


class DetectionModelFamily(str, Enum):
    YOLOV8_NANO = "yolov8n"
    YOLOV11_NANO = "yolo11n"
    YOLOV8_POSE = "yolov8n-pose"
    YOLOV11_POSE = "yolo11n-pose"


class ModelConfig(BaseModel):
    """
    Model & Acceleration Pipeline Configuration for Zero-Latency AI Surveillance Engine.
    Supports dynamic model switching between YOLOv8-Nano and YOLOv11-Nano,
    as well as Edge-TPU, Jetson TensorRT, CoreML, and CPU/CUDA backends.
    """
    detection_model: str = Field(
        default="yolo11n.pt",
        description="Path or Ultralytics specifier for detection model (e.g. yolo11n.pt, yolov8n.pt, yolo11n.onnx, yolo11n.engine)"
    )
    pose_model: str = Field(
        default="yolo11n-pose.pt",
        description="Path or Ultralytics specifier for pose estimation model (e.g. yolo11n-pose.pt)"
    )
    yunet_model_path: str = Field(
        default="face_detection_yunet_2023mar.onnx",
        description="Path to YuNet 5-landmark face alignment ONNX file"
    )
    hardware_target: HardwareTarget = Field(
        default=HardwareTarget.CPU,
        description="Hardware acceleration runtime target"
    )
    img_size: int = Field(default=640, description="Inference canvas resolution dimension")
    conf_threshold: float = Field(default=0.40, description="Spatial detection confidence threshold")
    iou_threshold: float = Field(default=0.45, description="NMS IoU threshold")
    subsample_interval: int = Field(default=3, description="Temporal sub-sampling interval for heavy biometrics (every Nth frame per track ID)")

    def get_effective_detection_model(self) -> str:
        """Returns valid model path or falls back gracefully to default weights if specified weight is unavailable."""
        if os.path.exists(self.detection_model):
            return self.detection_model
        # Check standard root/data paths
        for fallback in [self.detection_model, "yolo11n.pt", "yolov8n.pt", "yolo26n.pt"]:
            if os.path.exists(fallback):
                return fallback
        return self.detection_model  # Let Ultralytics auto-download if string specifier

    def get_effective_pose_model(self) -> str:
        if os.path.exists(self.pose_model):
            return self.pose_model
        for fallback in [self.pose_model, "yolo11n-pose.pt", "yolov8n-pose.pt"]:
            if os.path.exists(fallback):
                return fallback
        return self.pose_model


# Global default configuration instance
DEFAULT_MODEL_CONFIG = ModelConfig()

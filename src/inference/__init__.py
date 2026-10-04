# Inference Package
from src.inference.person_detector import (
    PersonDetector,
    UltralyticsPersonDetector,
    ScriptedPersonDetector,
    DetectedBox,
)
from src.inference.face_embedder import (
    FaceEmbedder,
    ArcFaceEmbedder,
    ScriptedFaceEmbedder,
)
from src.inference.multi_camera import (
    CameraIngestionPool,
    MultiCameraIngestionManager,
    BufferlessVideoCapture,
    EventAggregator,
)

__all__ = [
    "PersonDetector",
    "UltralyticsPersonDetector",
    "ScriptedPersonDetector",
    "DetectedBox",
    "FaceEmbedder",
    "ArcFaceEmbedder",
    "ScriptedFaceEmbedder",
    "CameraIngestionPool",
    "MultiCameraIngestionManager",
    "BufferlessVideoCapture",
    "EventAggregator",
]

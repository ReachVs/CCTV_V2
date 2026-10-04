"""
Person Detector Seam & Adapters.
Encapsulates person detection, motion tracking, and fallback heuristics behind a deep, explicit seam.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Callable, Union, Dict
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.config.model_config import DEFAULT_MODEL_CONFIG


@dataclass(frozen=True)
class DetectedBox:
    """Represents a detected, tracked person bounding box in normalized/downsampled coordinates."""
    box: List[float]  # [x1, y1, x2, y2]
    conf: float
    track_id: int


class PersonDetector(ABC):
    """
    Abstract seam for person detection & multi-object tracking.
    Decouples BiometricTrackingEngine from specific inference engines (YOLO, Haar, CoreML).
    """

    @abstractmethod
    def detect_and_track(self, frame: np.ndarray) -> List[DetectedBox]:
        """
        Executes person detection and motion tracking on downsampled frame (e.g. 640x360).
        Returns a list of DetectedBox instances.
        """
        pass


class UltralyticsPersonDetector(PersonDetector):
    """
    Production adapter using Ultralytics YOLO with ByteTrack and Haar cascade fallback.
    Encapsulates both neural network inference and traditional vision fallback.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        yolo_model: Optional[Any] = None,
        tracker_cfg: Optional[str] = None,
        enable_haar: bool = True,
        haar_cascade: Optional[Any] = None
    ):
        self.tracker_cfg = tracker_cfg or "bytetrack.yaml"
        self.yolo_model = yolo_model

        if self.yolo_model is None and YOLO is not None:
            effective_path = model_path or DEFAULT_MODEL_CONFIG.get_effective_detection_model()
            try:
                self.yolo_model = YOLO(effective_path)
                dummy = np.zeros((360, 640, 3), dtype=np.uint8)
                try:
                    self.yolo_model.track(dummy, persist=True, classes=[0], tracker=self.tracker_cfg, verbose=False)
                except Exception:
                    pass
            except Exception as e:
                print(f"[UltralyticsPersonDetector Warning] YOLO initialization: {e}")

        self.enable_haar = enable_haar
        self.haar_cascade = haar_cascade
        if self.enable_haar and self.haar_cascade is None:
            try:
                self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception as e:
                self.haar_cascade = None
                print(f"[UltralyticsPersonDetector Warning] Haar cascade initialization: {e}")

    def detect_and_track(self, frame: np.ndarray) -> List[DetectedBox]:
        detections: List[DetectedBox] = []

        # 1. Primary YOLO + ByteTrack
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.track(
                    frame,
                    conf=0.25,
                    persist=True,
                    classes=[0],
                    tracker=self.tracker_cfg,
                    verbose=False
                )
                for res in results:
                    if res.boxes is not None and len(res.boxes) > 0:
                        b = res.boxes.xyxy.cpu().numpy()
                        c = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else [1.0] * len(b)
                        if res.boxes.id is not None:
                            t = res.boxes.id.cpu().numpy().astype(int)
                        else:
                            t = list(range(1, len(b) + 1))
                        for i in range(len(b)):
                            detections.append(DetectedBox(
                                box=[float(x) for x in b[i]],
                                conf=float(c[i]),
                                track_id=int(t[i])
                            ))
            except Exception:
                pass

        # 2. Haar Cascade Fallback when no bodies are tracked
        if len(detections) == 0 and self.enable_haar and self.haar_cascade is not None:
            try:
                target_h, target_w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(35, 35))
                if len(faces) > 0:
                    for idx_f, (hx, hy, hw, hh) in enumerate(faces):
                        pad_h = int(hh * 0.4)
                        pad_w = int(hw * 0.3)
                        bx1 = max(0.0, float(hx - pad_w))
                        by1 = max(0.0, float(hy - pad_h))
                        bx2 = min(float(target_w), float(hx + hw + pad_w))
                        by2 = min(float(target_h), float(hy + hh + int(pad_h * 2)))
                        detections.append(DetectedBox(
                            box=[bx1, by1, bx2, by2],
                            conf=0.85,
                            track_id=idx_f + 1
                        ))
            except Exception:
                pass

        return detections


class ScriptedPersonDetector(PersonDetector):
    """
    Deterministic test adapter for person detection and tracking.
    Enables hermetic, ultra-fast unit testing without loading weights or PyTorch.
    """

    def __init__(self, detections: Optional[Any] = None):
        self._detections = detections
        self.call_count = 0

    def set_detections(self, detections: Any) -> None:
        self._detections = detections

    def detect_and_track(self, frame: np.ndarray) -> List[DetectedBox]:
        self.call_count += 1
        raw = self._detections
        if callable(raw):
            raw = raw()
        if raw is None:
            return []

        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], DetectedBox):
            return raw

        out: List[DetectedBox] = []
        if hasattr(raw, "boxes"):
            raw = raw.boxes

        if hasattr(raw, "xyxy"):
            b = raw.xyxy.cpu().numpy()
            c = raw.conf.cpu().numpy() if hasattr(raw, "conf") and raw.conf is not None else [1.0] * len(b)
            t = raw.id.cpu().numpy().astype(int) if hasattr(raw, "id") and raw.id is not None else list(range(1, len(b) + 1))
            for i in range(len(b)):
                out.append(DetectedBox(
                    box=[float(x) for x in b[i]],
                    conf=float(c[i]),
                    track_id=int(t[i])
                ))
            return out

        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, DetectedBox):
                    out.append(item)
                elif isinstance(item, (tuple, list)) and len(item) == 3:
                    out.append(DetectedBox(
                        box=[float(x) for x in item[0]],
                        conf=float(item[1]),
                        track_id=int(item[2])
                    ))
                elif isinstance(item, dict):
                    out.append(DetectedBox(
                        box=[float(x) for x in item["box"]],
                        conf=float(item.get("conf", 1.0)),
                        track_id=int(item.get("track_id", 1))
                    ))
        return out

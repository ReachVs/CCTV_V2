import gc
import os
import json
import sqlite3
import queue
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None


class GDPRMemorySanitizer:
    """
    GDPR Compliance Protocol: Explicit Zero-Trace Memory Disposal Pattern.
    Overwrites raw camera frames and face crops in place with zeroes before deleting references.
    """
    @staticmethod
    def sanitize_image_buffer(arr: Optional[np.ndarray]):
        if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
            try:
                arr.fill(0)
            except Exception:
                pass
            del arr
            gc.collect()

    @staticmethod
    def sanitize_crop_list(crop_list: List[np.ndarray]):
        for crop in crop_list:
            if crop is not None and isinstance(crop, np.ndarray):
                try:
                    crop.fill(0)
                except Exception:
                    pass
        crop_list.clear()
        gc.collect()


class PoseOnlyEstimator:
    """
    Lightweight 17 Keypoint Body Estimator for Privacy-Preserving PoseOnlyTracking Mode.
    Bypasses facial biometrics entirely when enabled.
    Keypoint indices follow standard COCO 17-keypoint skeleton format:
      0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear, 5: L-Shoulder, 6: R-Shoulder,
      7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip,
      13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle.
    """
    def __init__(self, pose_model_spec: str = "yolo11n-pose.pt"):
        self.pose_model_spec = pose_model_spec
        self.yolo_pose = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            model_path = self.pose_model_spec if os.path.exists(self.pose_model_spec) else "yolo11n-pose.pt"
            self.yolo_pose = YOLO(model_path)
            print(f"[PoseEstimator] Loaded Ultralytics Pose model: '{model_path}'")
        except Exception as e:
            print(f"[PoseEstimator Warning] Could not load YOLO Pose model ({e}). Using synthetic keypoints fallback.")
            self.yolo_pose = None

    def estimate_pose_keypoints(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Dict[str, float]]:
        """
        Extracts 17 body keypoints within bounding box (x1, y1, x2, y2).
        Returns list of 17 keypoint dicts: [{"id": 0..16, "x": float, "y": float, "conf": float}].
        Raw pixel crop is NOT saved or retained.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1 or self.yolo_pose is None:
            return self._generate_fallback_keypoints(bbox)

        crop = frame[y1:y2, x1:x2]
        try:
            results = self.yolo_pose(crop, verbose=False)
            keypoints_data = []
            if results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                kpts = results[0].keypoints.data.cpu().numpy()
                if kpts.shape[0] > 0:
                    person_kpts = kpts[0]  # shape (17, 3) -> (x, y, conf)
                    for idx in range(min(17, len(person_kpts))):
                        kx, ky, conf = person_kpts[idx]
                        keypoints_data.append({
                            "id": idx,
                            "x": float(x1 + kx),
                            "y": float(y1 + ky),
                            "conf": float(conf)
                        })
                    GDPRMemorySanitizer.sanitize_image_buffer(crop)
                    return keypoints_data

            GDPRMemorySanitizer.sanitize_image_buffer(crop)
            return self._generate_fallback_keypoints(bbox)
        except Exception as e:
            GDPRMemorySanitizer.sanitize_image_buffer(crop)
            return self._generate_fallback_keypoints(bbox)

    def _generate_fallback_keypoints(self, bbox: Tuple[int, int, int, int]) -> List[Dict[str, float]]:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        keypoints = []
        for i in range(17):
            # Simple proportional anatomical placement fallback
            offset_x = (i % 3 - 1) * (bw * 0.2)
            offset_y = (i / 17.0 - 0.5) * bh
            keypoints.append({
                "id": i,
                "x": float(center_x + offset_x),
                "y": float(center_y + offset_y),
                "conf": 0.0
            })
        return keypoints


# Re-export unified audit logging interfaces for backward compatibility
from src.database.audit_repository import (
    AuditLogger,
    EncryptedWALAuditLogger,
    AESEncryptedWALAuditLogger,
    ScriptedAuditLogger,
)

"""
OpenCV Frame Annotator for CCTV Surveillance & Biometric HUD.
Decouples visual presentation, color maps, and HUD overlays from core mathematical inference.
"""
from typing import List, Any
import numpy as np
import cv2

try:
    from src.domain.subject_session import TrackStatus
except ImportError:
    TrackStatus = None


class FrameAnnotator:
    """
    Renders bounding boxes, biometric identity labels, keypoints,
    and contention badges onto video frames.
    """

    # Color Palette (BGR)
    COLOR_CONFIRMED = (0, 255, 0)       # Green
    COLOR_SCANNING = (0, 165, 255)      # Amber / Orange
    COLOR_CONTENDED = (0, 140, 255)     # Deep Orange / Red-Orange
    COLOR_UNVERIFIED = (130, 130, 130)  # Muted Gray
    COLOR_UNKNOWN = (0, 0, 255)         # Bright Red
    COLOR_POSE = (255, 191, 0)          # Cyan / Blue
    COLOR_KEYPOINT = (0, 255, 255)      # Yellow

    def __init__(self, font_scale: float = 0.5, thickness: int = 2):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.box_thickness = thickness

    def annotate(self, frame: np.ndarray, subjects: List[Any]) -> np.ndarray:
        """
        Draws annotations for all active TrackedSubjects onto frame.
        Modifies frame in-place and returns it.
        """
        if frame is None or len(subjects) == 0:
            return frame

        for s in subjects:
            box = getattr(s, "box", None)
            if not box or len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            track_id = getattr(s, "track_id", 0)
            name = getattr(s, "name", "")
            status = getattr(s, "status", None)
            is_contended = getattr(s, "is_contended", False) or (status == TrackStatus.CONTENDED if TrackStatus else False)
            is_scanning = getattr(s, "is_scanning", False) or (status == TrackStatus.SCANNING if TrackStatus else False)
            is_unknown = getattr(s, "is_unknown", False) or (status == TrackStatus.UNKNOWN if TrackStatus else False)
            is_unverified = (status == TrackStatus.UNVERIFIED if TrackStatus else False) or name.startswith("Unverified")
            is_pose = (status == TrackStatus.POSE_TARGET if TrackStatus else False) or name.startswith("Pose Target")
            kpts = getattr(s, "keypoints", None)

            # 1. Pose Target Branch (17-KPT pose estimation)
            if is_pose:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_POSE, self.box_thickness)
                if kpts:
                    for kp in kpts:
                        cv2.circle(frame, (int(kp["x"]), int(kp["y"])), 3, self.COLOR_KEYPOINT, -1)
                pose_label = f"POSE ID {track_id} (17-KPT)"
                cv2.putText(
                    frame, pose_label, (x1 + 5, y1 - 8),
                    self.font, self.font_scale, self.COLOR_POSE, 1, cv2.LINE_AA
                )
                continue

            # 2. Standard Biometric & Subject Tracking Branch
            if is_contended:
                color = self.COLOR_CONTENDED
                label_str = f"ID {track_id}: Contended (Crossover)"
            elif is_scanning:
                color = self.COLOR_SCANNING
                label_str = f"ID {track_id}: Scanning..."
            elif is_unverified:
                color = self.COLOR_UNVERIFIED
                label_str = f"ID {track_id}: Unverified"
            elif not is_unknown:
                color = self.COLOR_CONFIRMED
                label_str = f"ID {track_id}: {name}"
            else:
                color = self.COLOR_UNKNOWN
                label_str = f"ID {track_id}: Unknown Person"

            # Draw target bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw label banner & text
            banner_w = len(label_str) * 11
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + banner_w, y1), color, -1)
            cv2.putText(
                frame, label_str, (x1 + 5, y1 - 8),
                self.font, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA
            )

        return frame

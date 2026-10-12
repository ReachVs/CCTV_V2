"""
HUD Annotator Seam & Visual Presentation Engine.
Decouples visual presentation, tactical color palettes, and HUD overlays from core mathematical inference.
Provides CyberpunkHUDAnnotator (production rich visuals), NoOpAnnotator (headless/tests),
and ScriptedAnnotator (test harness).
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Tuple
import numpy as np
import cv2

try:
    from src.domain.subject_session import TrackStatus
except ImportError:
    TrackStatus = None


class Annotator(ABC):
    """
    Abstract seam for rendering HUD bounding boxes, biometric identity labels,
    pose keypoints, and telemetry overlays onto video frames.
    """

    @abstractmethod
    def annotate(
        self,
        frame: np.ndarray,
        subjects: List[Any],
        fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Renders annotations onto frame and returns the annotated frame.
        """
        pass


class NoOpAnnotator(Annotator):
    """
    High-performance pass-through annotator for headless servers, unit tests,
    and CLI benchmarks. Eliminates OpenCV GUI drawing latency completely.
    """

    def annotate(
        self,
        frame: np.ndarray,
        subjects: List[Any],
        fps: Optional[float] = None
    ) -> np.ndarray:
        return frame


class ScriptedAnnotator(Annotator):
    """
    Deterministic test adapter for annotator verification.
    Records calls, frame shapes, and subject lists without running OpenCV operations.
    """

    def __init__(self):
        self.calls: List[Tuple[int, int, Optional[float]]] = []
        self.last_subjects: List[Any] = []

    def annotate(
        self,
        frame: np.ndarray,
        subjects: List[Any],
        fps: Optional[float] = None
    ) -> np.ndarray:
        self.calls.append((len(subjects), frame.shape[0] if frame is not None else 0, fps))
        self.last_subjects = list(subjects)
        return frame


class CyberpunkHUDAnnotator(Annotator):
    """
    Production-grade visual annotator featuring tactical reticle brackets,
    dynamic status color palettes, and high-visibility HUD overlays.
    """

    # Color Palette (BGR)
    COLOR_CONFIRMED = (0, 255, 0)       # Green
    COLOR_SCANNING = (0, 165, 255)      # Amber / Orange
    COLOR_CONTENDED = (0, 140, 255)     # Deep Orange / Red-Orange
    COLOR_UNVERIFIED = (130, 130, 130)  # Muted Gray
    COLOR_UNKNOWN = (0, 0, 255)         # Bright Red
    COLOR_POSE = (255, 191, 0)          # Cyan / Blue
    COLOR_KEYPOINT = (0, 255, 255)      # Yellow
    COLOR_HEADER_BG = (15, 23, 42)      # Dark Slate (RGB: 42, 23, 15)

    def __init__(
        self,
        font_scale: float = 0.5,
        thickness: int = 2,
        show_telemetry_bar: bool = False,
        draw_labels: bool = False,
        draw_effects: bool = False,
    ):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.box_thickness = thickness
        self.show_telemetry_bar = show_telemetry_bar
        self.draw_labels = draw_labels
        self.draw_effects = draw_effects

    def _draw_corner_brackets(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]):
        """Draws tactical corner accents around target bounding box."""
        box_w = x2 - x1
        box_h = y2 - y1
        line_len = max(8, min(int(box_w * 0.20), int(box_h * 0.20), 25))
        t = self.box_thickness + 1

        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + line_len, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1 + line_len), color, t)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - line_len, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1 + line_len), color, t)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + line_len, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2 - line_len), color, t)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - line_len, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2 - line_len), color, t)

    def annotate(
        self,
        frame: np.ndarray,
        subjects: List[Any],
        fps: Optional[float] = None
    ) -> np.ndarray:
        if frame is None:
            return frame

        h_frame, w_frame = frame.shape[:2]

        # 1. Top Tactical Telemetry HUD Bar
        if self.draw_effects and self.show_telemetry_bar and (fps is not None or len(subjects) > 0):
            header_text = f"SEC-NET ONLINE"
            if fps is not None:
                header_text += f" | FPS: {fps:.1f}"
            header_text += f" | ACTIVE TARGETS: {len(subjects)}"
            cv2.rectangle(frame, (10, 10), (10 + len(header_text) * 9 + 16, 34), self.COLOR_HEADER_BG, -1)
            cv2.rectangle(frame, (10, 10), (10 + len(header_text) * 9 + 16, 34), (70, 80, 95), 1)
            cv2.putText(frame, header_text, (18, 27), self.font, 0.42, (220, 230, 242), 1, cv2.LINE_AA)

        if len(subjects) == 0:
            return frame

        # 2. Render Each Subject Box & Status Badge
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

            # A. Pose Target Branch (17-KPT pose skeleton)
            if is_pose:
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.COLOR_POSE, self.box_thickness)
                if self.draw_effects:
                    self._draw_corner_brackets(frame, x1, y1, x2, y2, self.COLOR_POSE)
                if kpts:
                    for kp in kpts:
                        cv2.circle(frame, (int(kp["x"]), int(kp["y"])), 3, self.COLOR_KEYPOINT, -1)
                if self.draw_labels:
                    pose_label = f"POSE ID {track_id} (17-KPT)"
                    cv2.putText(
                        frame, pose_label, (x1 + 5, max(15, y1 - 8)),
                        self.font, self.font_scale, self.COLOR_POSE, 1, cv2.LINE_AA
                    )
                continue

            # B. Standard Biometric & Subject Tracking Branch
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

            # Draw clean target bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            # Optional corner brackets effect
            if self.draw_effects:
                self._draw_corner_brackets(frame, x1, y1, x2, y2, color)

            # Optional label banner & text
            if self.draw_labels:
                banner_w = len(label_str) * 11
                banner_y1 = max(0, y1 - 25)
                banner_y2 = max(25, y1)
                cv2.rectangle(frame, (x1, banner_y1), (x1 + banner_w, banner_y2), color, -1)
                cv2.putText(
                    frame, label_str, (x1 + 5, banner_y2 - 7),
                    self.font, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA
                )

        return frame


# Backward-compatible alias for existing callers
FrameAnnotator = CyberpunkHUDAnnotator

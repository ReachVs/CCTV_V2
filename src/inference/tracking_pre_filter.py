"""
Deep Module: TrackingPreFilter
==============================
Encapsulates candidate spatial validation, zero-jitter filtering, anatomical presence
(YuNet face and YOLO pose keypoint verification), sub-box containment, spatio-temporal
overlap suppression (NMS), and crossover contention arbitration behind a minimalist interface.
"""

from dataclasses import dataclass
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np


def compute_box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Union (IoU) between bounding boxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea) / float(boxAArea + boxBArea - interArea + 1e-6)


def compute_box_containment(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Computes Intersection-over-Smaller (IoS) / Containment ratio between two bounding boxes.
    Returns the fraction of the smaller box's area that overlaps with the other box.
    Range: [0.0, 1.0].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea <= 0:
        return 0.0
    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    smallerArea = min(areaA, areaB)
    return float(interArea) / float(smallerArea)


@dataclass
class FilteredCandidate:
    track_id: int
    box: Tuple[int, int, int, int]
    conf: float
    area: int
    is_confirmed: bool


@dataclass
class PreFilterResult:
    candidates: List[FilteredCandidate]
    current_frame_boxes: Dict[int, Tuple[int, int, int, int]]
    newly_contended_track_ids: List[int]
    newly_contended_pairs: List[Tuple[int, int, float]]
    resolved_contended_track_ids: List[int]
    suppressed_track_ids: List[int]


class TrackingPreFilter:
    """
    Deep Module: Pre-filters raw tracker detections and manages spatial contention state.
    Hides YuNet face alignment checks, YOLO pose keypoint fallbacks, temporal cache expirations,
    IoU overlap matrix evaluations, and 3-frame contention hysteresis.
    """

    def __init__(
        self,
        face_aligner: Optional[Any] = None,
        pose_estimator: Optional[Any] = None,
        jitter_engine: Optional[Any] = None,
    ):
        self.face_aligner = face_aligner
        self.pose_estimator = pose_estimator
        self.jitter_engine = jitter_engine
        self.lock = threading.RLock()

        # State tracked across frames
        self.head_presence_cache: Dict[int, int] = {}
        self.contended_tracks: Dict[int, int] = {}
        self.contention_blacklist: Dict[int, int] = {}
        self.recent_track_history: Dict[int, Dict[str, Any]] = {}

    def purge_track(self, track_id: int) -> None:
        """Purges cached spatial and contention state for a terminated track."""
        with self.lock:
            self.head_presence_cache.pop(track_id, None)
            self.contended_tracks.pop(track_id, None)
            self.contention_blacklist.pop(track_id, None)
            self.recent_track_history.pop(track_id, None)

    def has_head_or_shoulder_presence(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        track_id: int,
        current_frame_num: int
    ) -> bool:
        """
        Validates anatomical structure before admitting detection into identity scanning.
        Uses cached result if available (15 frame TTL) to avoid redundant inference.
        """
        with self.lock:
            cached_exp = self.head_presence_cache.get(track_id, 0)
            if current_frame_num <= cached_exp:
                return True

        fx1, fy1, fx2, fy2 = box
        box_h = fy2 - fy1
        box_w = fx2 - fx1

        # 1. Fast Face Detection Check on Upper Torso/Head Crop (YuNet)
        h_frame, w_frame = frame.shape[:2]
        crop_y1 = max(0, fy1 - int(box_h * 0.10))
        crop_y2 = min(h_frame, fy1 + int(box_h * 0.55))
        crop_x1 = max(0, fx1 - int(box_w * 0.15))
        crop_x2 = min(w_frame, fx2 + int(box_w * 0.15))

        if crop_y2 > crop_y1 and crop_x2 > crop_x1 and self.face_aligner is not None:
            head_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if head_crop.size > 0:
                try:
                    aligned = self.face_aligner.align_face(head_crop, strict=True)
                    if aligned is not None:
                        with self.lock:
                            self.head_presence_cache[track_id] = current_frame_num + 15
                        return True
                except Exception:
                    pass

        # 2. Pose Keypoints Fallback (0: Nose, 1..2: Eyes, 3..4: Ears, 5..6: Shoulders)
        if self.pose_estimator is not None and getattr(self.pose_estimator, "yolo_pose", None) is not None:
            try:
                kpts = self.pose_estimator.estimate_pose_keypoints(frame, box)
                if kpts and len(kpts) > 0:
                    for kp in kpts:
                        k_id = kp.get("id", -1)
                        k_conf = kp.get("conf", 0.0)
                        if 0 <= k_id <= 6 and k_conf >= 0.35:
                            with self.lock:
                                self.head_presence_cache[track_id] = current_frame_num + 15
                            return True
            except Exception:
                pass

        return False

    def filter_and_arbitrate(
        self,
        frame: np.ndarray,
        detections: List[Any],
        scale_x: float,
        scale_y: float,
        current_frame_num: int,
        active_track_ids: Set[int],
        confirmed_track_ids: Set[int],
    ) -> PreFilterResult:
        """
        Executes spatial jitter filtering, aspect ratio/size checks, head presence checks,
        spatial overlap suppression, and cross-subject contention arbitration.
        """
        raw_candidates: List[Dict[str, Any]] = []
        suppressed_track_ids: List[int] = []

        # 1. Geometric Validation & Jitter Smoothing
        for det in detections:
            box = det.box
            conf = det.conf
            track_id = det.track_id

            is_active = (track_id in active_track_ids) or (track_id in confirmed_track_ids)
            min_conf = 0.25 if is_active else 0.45
            if float(conf) < min_conf:
                continue

            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            if self.jitter_engine is not None:
                fx1, fy1, fx2, fy2 = self.jitter_engine.filter_box(track_id, (x1, y1, x2, y2), confidence=float(conf))
            else:
                fx1, fy1, fx2, fy2 = x1, y1, x2, y2

            box_h = fy2 - fy1
            box_w = fx2 - fx1
            aspect_ratio = float(box_w) / float(box_h) if box_h > 0 else 0.0

            if box_h < 80 or box_w < 80:
                continue
            if aspect_ratio < 0.40 or aspect_ratio > 1.80:
                continue

            area = box_h * box_w
            is_confirmed = track_id in confirmed_track_ids

            # Pre-Admission Head/Shoulder Filter
            if not is_confirmed:
                if not self.has_head_or_shoulder_presence(frame, (fx1, fy1, fx2, fy2), track_id, current_frame_num):
                    if track_id in active_track_ids:
                        suppressed_track_ids.append(track_id)
                    continue

            raw_candidates.append({
                "track_id": track_id,
                "box": (fx1, fy1, fx2, fy2),
                "conf": float(conf),
                "area": area,
                "is_confirmed": is_confirmed
            })

        # Sort: Confirmed first, then larger body area, then higher confidence
        raw_candidates.sort(key=lambda c: (1 if c["is_confirmed"] else 0, c["area"], c["conf"]), reverse=True)

        # 2. Spatio-Temporal Containment & Overlap NMS
        current_frame_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        accepted_candidates: List[FilteredCandidate] = []

        for cand in raw_candidates:
            cid = cand["track_id"]
            cbox = cand["box"]
            c_confirmed = cand["is_confirmed"]

            suppressed = False
            # Check against already accepted boxes in this frame
            for acc_id, acc_box in current_frame_boxes.items():
                iou = compute_box_iou(cbox, acc_box)
                containment = compute_box_containment(cbox, acc_box)
                if iou > 0.35 or containment > 0.40:
                    is_established_cid = c_confirmed or (cid in active_track_ids)
                    is_established_acc = (acc_id in active_track_ids) or (acc_id in confirmed_track_ids)
                    if is_established_cid and is_established_acc and iou <= 0.65:
                        pass
                    else:
                        suppressed = True
                        break

            # Check against recent confirmed tracks in history (within 15 frames)
            if not suppressed and not c_confirmed:
                for hist_id, hist_info in self.recent_track_history.items():
                    if hist_id != cid and hist_info.get("confirmed", False):
                        if (current_frame_num - hist_info.get("last_seen", 0)) <= 15:
                            iou = compute_box_iou(cbox, hist_info["box"])
                            containment = compute_box_containment(cbox, hist_info["box"])
                            if iou > 0.35 or containment > 0.40:
                                if (cid in active_track_ids) and iou <= 0.65:
                                    pass
                                else:
                                    suppressed = True
                                    break

            if suppressed:
                if not c_confirmed:
                    suppressed_track_ids.append(cid)
                    self.purge_track(cid)
                continue

            current_frame_boxes[cid] = cbox
            accepted_candidates.append(FilteredCandidate(
                track_id=cid,
                box=cbox,
                conf=cand["conf"],
                area=cand["area"],
                is_confirmed=c_confirmed
            ))

        # 3. Crossover Contention & Pairwise Overlap Evaluation
        newly_contended: List[int] = []
        newly_contended_pairs: List[Tuple[int, int, float]] = []
        resolved_contended: List[int] = []

        with self.lock:
            prev_contended = set(self.contended_tracks.keys())
            # Update Hysteresis for currently contended tracks
            for c_tid in list(self.contended_tracks.keys()):
                if c_tid in current_frame_boxes:
                    c_box = current_frame_boxes[c_tid]
                    max_iou = 0.0
                    max_cont = 0.0
                    for other_tid, other_box in current_frame_boxes.items():
                        if other_tid == c_tid:
                            continue
                        max_iou = max(max_iou, compute_box_iou(c_box, other_box))
                        max_cont = max(max_cont, compute_box_containment(c_box, other_box))

                    if max_iou < 0.15 and max_cont < 0.20:
                        self.contended_tracks[c_tid] += 1
                        if self.contended_tracks[c_tid] >= 3:
                            del self.contended_tracks[c_tid]
                            resolved_contended.append(c_tid)
                    else:
                        self.contended_tracks[c_tid] = 0
                else:
                    self.contended_tracks.pop(c_tid, None)

            # Detect New Pairwise Contention
            box_items = list(current_frame_boxes.items())
            for i in range(len(box_items)):
                tid1, b1 = box_items[i]
                for j in range(i + 1, len(box_items)):
                    tid2, b2 = box_items[j]
                    iou_pair = compute_box_iou(b1, b2)
                    cont_pair = compute_box_containment(b1, b2)
                    if iou_pair > 0.30 or cont_pair > 0.40:
                        for tid in (tid1, tid2):
                            self.contended_tracks[tid] = 0
                            self.contention_blacklist[tid] = current_frame_num + 45
                            if tid not in newly_contended:
                                newly_contended.append(tid)

                        if tid1 not in prev_contended or tid2 not in prev_contended:
                            newly_contended_pairs.append((tid1, tid2, float(iou_pair)))

        return PreFilterResult(
            candidates=accepted_candidates,
            current_frame_boxes=current_frame_boxes,
            newly_contended_track_ids=newly_contended,
            newly_contended_pairs=newly_contended_pairs,
            resolved_contended_track_ids=resolved_contended,
            suppressed_track_ids=suppressed_track_ids,
        )

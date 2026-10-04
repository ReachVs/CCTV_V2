# Force single-threaded execution across low-level C++ math backends
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
torch.set_num_threads(1)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.model_config import DEFAULT_MODEL_CONFIG, ModelConfig
from src.inference.temporal_subsampler import TemporalSubsampler
from src.utils.jitter_filter import JitterFilterEngine
from src.utils.face_align import YuNetFaceAligner
from src.utils.fsrcnn_upscaler import FSRCNNUpscaler
from src.rendering.annotator import FrameAnnotator
from src.domain.subject_session import SubjectSession, TrackStatus
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
from src.utils.privacy_compliance import (
    GDPRMemorySanitizer,
    PoseOnlyEstimator,
    AESEncryptedWALAuditLogger
)

try:
    import faiss
except ImportError:
    faiss = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None


# =====================================================================
# Domain Contracts (CONTEXT.md)
# =====================================================================

@dataclass
class TrackedSubject:
    """Immutable domain representation of a subject observed by the engine."""
    track_id: int
    name: str
    confidence: float
    box: Tuple[int, int, int, int]
    is_confirmed: bool
    is_scanning: bool
    is_unknown: bool
    is_contended: bool = False
    status: Optional[str] = None
    keypoints: Optional[List[Dict[str, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    """Immutable result contract returned by BiometricTrackingEngine.process_frame()."""
    annotated_frame: np.ndarray
    active_tracks: List[TrackedSubject]
    audit_events: List[Dict[str, Any]]
    frame_count: int
    fps: float


# =====================================================================
# Vector Index Seam & Adapters (Delegated to src.database.vector_store)
# =====================================================================
from src.database.vector_store import (
    VectorIndexAdapter,
    FAISSIndexAdapter,
    InMemoryIndexAdapter
)


# =====================================================================
# Internal Utilities
# =====================================================================

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


# =====================================================================
# Deep Biometric Tracking Engine Module
# =====================================================================

class BiometricTrackingEngine:
    """
    Deep Module: Consolidates Motion Detection, Zero-Jitter Spatial Filtering,
    Smart Tracker-Gating, Umeyama Landmark Alignment, AdaFace Vector Search,
    Consensus Verification, and Thread-Safe State Management behind a clean interface.
    """

    def __init__(
        self,
        index_adapter: Optional[VectorIndexAdapter] = None,
        detector: Optional[PersonDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        db_path: Optional[str] = None,
        l2_threshold: float = 0.68,
        margin_threshold: float = 0.15,
        consensus_votes: int = 2,
        history_window: int = 5,
        cooldown_frames: int = 150,
        model_name: str = "ArcFace",
        batch_size: int = 32,
        timeout_frames: int = 3,
    ):
        self.lock = threading.RLock()
        self.l2_threshold = l2_threshold
        self.margin_threshold = margin_threshold
        self.consensus_votes = consensus_votes
        self.history_window = history_window
        self.cooldown_frames = cooldown_frames
        self.model_name = model_name
        self.batch_size = batch_size
        self.timeout_frames = timeout_frames

        # Resolve paths
        data_dir = os.path.join(PROJECT_ROOT, "data")
        self.index_path = index_path or os.path.join(data_dir, "faces.index")
        self.metadata_path = metadata_path or os.path.join(data_dir, "metadata.json")
        self.db_path = db_path or os.path.join(data_dir, "faces.db")

        # Vector Index Setup
        self.metadata: Dict[str, Any] = {}
        if index_adapter is not None:
            self.index_adapter = index_adapter
        else:
            self.index_adapter = self._load_default_index()

        # Encapsulated Models & Pipelines
        self.bytetrack_cfg = os.path.join(PROJECT_ROOT, "bytetrack.yaml")
        if not os.path.exists(self.bytetrack_cfg):
            self.bytetrack_cfg = "bytetrack.yaml"

        # Person Detector Seam (Ultralytics + ByteTrack + Haar or Scripted test adapter)
        if detector is not None:
            self.detector = detector
        else:
            self.detector = UltralyticsPersonDetector(
                tracker_cfg=self.bytetrack_cfg,
                enable_haar=True
            )

        self.jitter_engine = JitterFilterEngine(erratic_threshold=80.0)
        self.face_aligner = YuNetFaceAligner()
        self.fsrcnn_upscaler = FSRCNNUpscaler()

        # Face Embedder Seam (ArcFace or Scripted test adapter)
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = ArcFaceEmbedder(
                model_name=self.model_name,
                face_aligner=self.face_aligner,
                upscaler=self.fsrcnn_upscaler
            )

        self.temporal_subsampler = TemporalSubsampler(subsample_interval=3)
        self.encrypted_audit_logger = AESEncryptedWALAuditLogger(db_path=self.db_path)
        self.pose_estimator = PoseOnlyEstimator(pose_model_spec=DEFAULT_MODEL_CONFIG.get_effective_pose_model())
        self.annotator = FrameAnnotator()

        # Privacy mode flag
        self._pose_only_mode = False

        # Internal Tracking State
        self.frame_count = 0
        self._sessions: Dict[int, SubjectSession] = {}
        self.active_tracks: Dict[int, str] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.track_first_seen: Dict[int, int] = {}
        self.track_last_submitted_frame: Dict[int, int] = {}
        self.recent_track_history: Dict[int, Dict[str, Any]] = {}
        self.track_keypoints: Dict[int, List[Dict[str, float]]] = {}
        self.limb_cooldowns: Dict[int, int] = {}
        self.identified_cooldowns: Dict[str, int] = {}
        self.active_identity_claims: Dict[str, int] = {}
        self.track_cache: Dict[int, Dict[str, Any]] = {}
        # Crossover Contention & Anti-ID-Swapping
        self.contended_tracks: Dict[int, int] = {}  # track_id -> consecutive non-overlap frame count
        self.contention_blacklist: Dict[int, int] = {}  # track_id -> blacklist expiration frame
        # Head/Shoulder Pre-Gating & Limb Noise Suppression
        self.head_presence_cache: Dict[int, int] = {}  # track_id -> presence validation expiration frame (15-frame TTL)

        # Batch Face Embedder Setup
        self._embedder_queue: List[Dict[str, Any]] = []
        self._embedder_lock = threading.RLock()
        self._keras_model = None
        self._embedder_running = True
        self._init_embedder_model()

        self._embedder_thread = threading.Thread(target=self._embedder_worker, daemon=True)
        self._embedder_thread.start()

    def _get_or_create_session(self, track_id: int, box: Tuple[int, int, int, int], frame_num: int) -> SubjectSession:
        if track_id not in self._sessions:
            session = SubjectSession(
                track_id=track_id,
                first_seen=frame_num,
                last_seen=frame_num,
                box=box
            )
            self._sessions[track_id] = session
        else:
            session = self._sessions[track_id]
            session.last_seen = frame_num
            session.box = box
        return session

    # -----------------------------------------------------------------
    # Initializers & Index Management
    # -----------------------------------------------------------------

    def _load_default_index(self) -> VectorIndexAdapter:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    self.metadata = json.load(f)
            except Exception:
                self.metadata = {}

        if faiss is not None and os.path.exists(self.index_path):
            try:
                raw_idx = faiss.read_index(self.index_path)
                return FAISSIndexAdapter(raw_idx)
            except Exception as e:
                print(f"[BiometricEngine] Failed to load FAISS index from '{self.index_path}': {e}")
        
        return FAISSIndexAdapter(dimension=512)

    @property
    def index(self) -> Any:
        if isinstance(self.index_adapter, FAISSIndexAdapter):
            return self.index_adapter.raw_index
        return self.index_adapter

    @property
    def ntotal(self) -> int:
        return self.index_adapter.ntotal

    @property
    def _yolo_model(self) -> Any:
        return getattr(self.detector, "yolo_model", None)

    @_yolo_model.setter
    def _yolo_model(self, model: Any) -> None:
        if isinstance(self.detector, UltralyticsPersonDetector):
            self.detector.yolo_model = model
        elif isinstance(self.detector, ScriptedPersonDetector):
            self.detector.set_detections(model)
        else:
            self.detector = UltralyticsPersonDetector(
                yolo_model=model,
                tracker_cfg=self.bytetrack_cfg,
                enable_haar=False
            )

    @property
    def _haar_cascade(self) -> Any:
        return getattr(self.detector, "haar_cascade", None)

    @_haar_cascade.setter
    def _haar_cascade(self, cascade: Any) -> None:
        if hasattr(self.detector, "haar_cascade"):
            self.detector.haar_cascade = cascade

    def _init_yolo(self):
        if not hasattr(self, "detector") or self.detector is None:
            self.detector = UltralyticsPersonDetector(tracker_cfg=self.bytetrack_cfg, enable_haar=True)

    @property
    def _keras_model(self) -> Any:
        return getattr(self.embedder, "keras_model", None)

    @_keras_model.setter
    def _keras_model(self, model: Any) -> None:
        if hasattr(self.embedder, "keras_model"):
            self.embedder.keras_model = model

    def _init_embedder_model(self):
        if hasattr(self.embedder, "_init_model"):
            self.embedder._init_model()

    def reload_index(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None) -> None:
        """Atomic hot-reload of FAISS vector index and metadata without dropping active tracks."""
        with self.lock:
            if index_path:
                self.index_path = index_path
            if metadata_path:
                self.metadata_path = metadata_path
            self.index_adapter = self._load_default_index()
            print(f"[BiometricEngine] Hot-reloaded vector index ({self.index_adapter.ntotal} profiles).")

    def update_database(self, embeddings: List[Any], metadata: Union[Dict, List]) -> None:
        """Dynamically update/replace vector index and metadata."""
        with self.lock:
            emb_matrix = np.array(embeddings, dtype=np.float32)
            if emb_matrix.ndim == 1:
                emb_matrix = np.expand_dims(emb_matrix, axis=0)
            norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            emb_matrix = emb_matrix / norms
            dim = emb_matrix.shape[1] if emb_matrix.ndim > 1 else 512

            if faiss is not None:
                try:
                    idx = faiss.IndexFlatL2(dim)
                    idx.add(emb_matrix)
                    self.index_adapter = FAISSIndexAdapter(idx)
                except Exception:
                    self.index_adapter = InMemoryIndexAdapter(emb_matrix, dimension=dim)
            else:
                self.index_adapter = InMemoryIndexAdapter(emb_matrix, dimension=dim)

            if isinstance(metadata, dict):
                self.metadata = metadata
            elif isinstance(metadata, list):
                self.metadata = {str(i): name for i, name in enumerate(metadata)}

    def _has_head_or_shoulder_presence(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        track_id: int,
        current_frame_num: int
    ) -> bool:
        """
        Validates whether candidate box contains upper-body or head evidence (pose keypoints
        0..6 [nose, eyes, ears, shoulders] with conf >= 0.35, or a detected face).
        Maintains a 15-frame TTL cache per track to preserve 30+ FPS throughput.
        """
        with self.lock:
            cached_exp = self.head_presence_cache.get(track_id, 0)
            if current_frame_num <= cached_exp:
                return True

        fx1, fy1, fx2, fy2 = box
        box_h = fy2 - fy1
        box_w = fx2 - fx1

        # 1. Fast Face Detection Check on Upper Torso/Head Crop (YuNet C++ OpenCV DNN: ~2-4ms)
        h_frame, w_frame = frame.shape[:2]
        crop_y1 = max(0, fy1 - int(box_h * 0.10))
        crop_y2 = min(h_frame, fy1 + int(box_h * 0.55))
        crop_x1 = max(0, fx1 - int(box_w * 0.15))
        crop_x2 = min(w_frame, fx2 + int(box_w * 0.15))

        if crop_y2 > crop_y1 and crop_x2 > crop_x1:
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
        # Evaluated only when YuNet face detection returns None (e.g. back turned or low lighting)
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

    # -----------------------------------------------------------------
    # Primary Seam: process_frame
    # -----------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, annotate: bool = True) -> ProcessResult:
        """
        Primary engine interface.
        Accepts raw video frame, executes tracking, spatial smoothing, smart gating,
        face recognition, and returns structured ProcessResult with zero leaked globals.
        """
        start_time = time.time()
        with self.lock:
            self.frame_count += 1
            current_frame_num = self.frame_count

        h_frame, w_frame, _ = frame.shape
        target_w, target_h = 640, 360
        scale_x = w_frame / float(target_w)
        scale_y = h_frame / float(target_h)

        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        annotated = frame.copy() if annotate else frame

        # 1. Motion Tracking via PersonDetector Seam
        detections = self.detector.detect_and_track(small_frame)

        # 2. Spatial Jitter Filtering & Geometric Validation
        raw_candidates: List[Dict[str, Any]] = []
        for det in detections:
            box = det.box
            conf = det.conf
            track_id = det.track_id
            # Dual-Threshold Track Initiation:
            # Active/confirmed tracks maintain continuity down to conf >= 0.25.
            # Brand new tracks require higher confidence (conf >= 0.45) to prevent spurious phantom initiation.
            is_active = (track_id in self.active_tracks) or (track_id in self.track_cache)
            min_conf = 0.25 if is_active else 0.45
            if float(conf) < min_conf:
                continue
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)

            fx1, fy1, fx2, fy2 = self.jitter_engine.filter_box(track_id, (x1, y1, x2, y2), confidence=float(conf))
            box_h = fy2 - fy1
            box_w = fx2 - fx1
            aspect_ratio = float(box_w) / float(box_h) if box_h > 0 else 0.0

            if box_h < 80 or box_w < 80:
                continue
            if aspect_ratio < 0.40 or aspect_ratio > 1.80:
                continue

            area = box_h * box_w
            is_confirmed = self.track_cache.get(track_id, {}).get("is_confirmed", False)

            # Pre-Admission Head/Shoulder Filter:
            # Drop bare limb candidates (hands, arms, legs) before they enter active_tracks or scanning
            if not is_confirmed:
                if not self._has_head_or_shoulder_presence(frame, (fx1, fy1, fx2, fy2), track_id, current_frame_num):
                    if track_id in self.active_tracks and self.active_tracks[track_id].startswith("Scanning"):
                        self.purge_track(track_id)
                    continue

            raw_candidates.append({
                "track_id": track_id,
                "box": (fx1, fy1, fx2, fy2),
                "conf": float(conf),
                "area": area,
                "is_confirmed": is_confirmed
            })

        # Sort candidates: Confirmed first, then larger body area, then higher confidence
        raw_candidates.sort(key=lambda c: (1 if c["is_confirmed"] else 0, c["area"], c["conf"]), reverse=True)

        # Spatio-Temporal Containment & Overlap NMS
        current_frame_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        for cand in raw_candidates:
            cid = cand["track_id"]
            cbox = cand["box"]
            c_confirmed = cand["is_confirmed"]

            # 1) Check against already accepted boxes in this frame
            suppressed = False
            for acc_id, acc_box in current_frame_boxes.items():
                iou = compute_box_iou(cbox, acc_box)
                containment = compute_box_containment(cbox, acc_box)
                if iou > 0.35 or containment > 0.40:
                    is_established_cid = c_confirmed or (cid in self.active_tracks) or (cid in self.track_cache)
                    is_established_acc = (acc_id in self.active_tracks) or (acc_id in self.track_cache)
                    if is_established_cid and is_established_acc and iou <= 0.65:
                        pass
                    else:
                        suppressed = True
                        break

            # 2) Check against recent confirmed tracks in history (within last 15 frames)
            if not suppressed and not c_confirmed:
                for hist_id, hist_info in self.recent_track_history.items():
                    if hist_id != cid and hist_info.get("confirmed", False):
                        if (current_frame_num - hist_info.get("last_seen", 0)) <= 15:
                            iou = compute_box_iou(cbox, hist_info["box"])
                            containment = compute_box_containment(cbox, hist_info["box"])
                            if iou > 0.35 or containment > 0.40:
                                if (cid in self.active_tracks) and iou <= 0.65:
                                    pass
                                else:
                                    suppressed = True
                                    break

            if suppressed:
                if not c_confirmed:
                    self.purge_track(cid)
                continue

            current_frame_boxes[cid] = cbox

        active_subject_list: List[TrackedSubject] = []
        new_audit_events: List[Dict[str, Any]] = []

        # Crossover Contention & Pairwise Overlap Evaluation
        with self.lock:
            # 1. Update Hysteresis for currently contended tracks
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
                    else:
                        self.contended_tracks[c_tid] = 0
                else:
                    self.contended_tracks.pop(c_tid, None)

            # 2. Detect New Pairwise Contention
            prev_contended = set(self.contended_tracks.keys())
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
                            t_cache = self.track_cache.setdefault(tid, {})
                            if t_cache.get("is_confirmed", False) or not self.active_tracks.get(tid, "").startswith("Scanning"):
                                t_cache["is_confirmed"] = False
                                t_cache["history"] = []
                                self.active_tracks[tid] = f"Contended Track {tid}"

                        if tid1 not in prev_contended or tid2 not in prev_contended:
                            new_audit_events.append({
                                "event_type": "CROSSOVER_CONTENTION",
                                "frame": current_frame_num,
                                "track_id_1": tid1,
                                "track_id_2": tid2,
                                "iou": round(float(iou_pair), 3),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "message": f"Tracks {tid1} and {tid2} entered crossover contention (IoU: {iou_pair:.2f}). Identities unlocked."
                            })

        # 3. Process Each Tracked Box
        for track_id, curr_box in current_frame_boxes.items():
            fx1, fy1, fx2, fy2 = curr_box
            box_h = fy2 - fy1
            box_w = fx2 - fx1

            with self.lock:
                self.track_last_seen[track_id] = current_frame_num
                self.track_first_seen.setdefault(track_id, current_frame_num)
                track_age = (current_frame_num - self.track_first_seen[track_id]) + 1

            # Privacy / Pose-Only Branch
            if self._pose_only_mode:
                kpts = self.pose_estimator.estimate_pose_keypoints(frame, curr_box)
                with self.lock:
                    self.active_tracks[track_id] = f"Pose Target {track_id}"
                    self.track_keypoints[track_id] = kpts

                subject = TrackedSubject(
                    track_id=track_id,
                    name=f"Pose Target {track_id}",
                    confidence=1.0,
                    box=curr_box,
                    is_confirmed=True,
                    is_scanning=False,
                    is_unknown=False,
                    status=TrackStatus.POSE_TARGET.value,
                    keypoints=kpts
                )
                session = self._get_or_create_session(track_id, curr_box, current_frame_num)
                session.mark_pose(kpts)
                active_subject_list.append(subject)
                continue

            # Standard Biometric Identification Branch
            # Focus face search on the upper torso/head region (top 55% of the body box)
            pad_y = int(box_h * 0.10)
            pad_x = int(box_w * 0.15)
            crop_y1 = max(0, fy1 - pad_y)
            crop_y2 = min(h_frame, fy1 + int(box_h * 0.55))
            crop_x1 = max(0, fx1 - pad_x)
            crop_x2 = min(w_frame, fx2 + pad_x)
            temp_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2] if (crop_y2 > crop_y1 and crop_x2 > crop_x1) else None

            with self.lock:
                # Spatial IoU inheritance across track swaps
                if track_id not in self.active_tracks:
                    best_iou = 0.0
                    inherited_name = None
                    for old_tid, old_info in list(self.recent_track_history.items()):
                        if current_frame_num - old_info["last_seen"] <= 45 and old_info.get("confirmed", False):
                            # Skip candidates blacklisted due to recent crossover contention
                            if old_tid in self.contention_blacklist and current_frame_num <= self.contention_blacklist[old_tid]:
                                continue
                            name_cand = old_info.get("name", "")
                            if name_cand not in ["Unknown", "Ambiguous Match", "Unknown Person"] and not name_cand.startswith("Scanning") and not name_cand.startswith("Contended"):
                                iou_val = compute_box_iou(curr_box, old_info["box"])
                                if iou_val > best_iou:
                                    best_iou = iou_val
                                    inherited_name = name_cand

                    if best_iou >= 0.40 and inherited_name is not None:
                        self.active_tracks[track_id] = inherited_name
                        t_c = self.track_cache.setdefault(track_id, {})
                        t_c["name"] = inherited_name
                        t_c["is_confirmed"] = False
                        t_c["best_dist"] = 0.50
                        t_c["best_norm"] = 1.0
                    else:
                        self.active_tracks[track_id] = f"Scanning Track {track_id}"

                subject_name = self.active_tracks.get(track_id, f"Scanning Track {track_id}")
                is_confirmed = self.track_cache.get(track_id, {}).get("is_confirmed", False)
                self.recent_track_history[track_id] = {
                    "box": curr_box,
                    "last_seen": current_frame_num,
                    "name": subject_name,
                    "confirmed": is_confirmed
                }

            # Tiny face distance cutoff
            if box_h < 90 or box_w < 45:
                with self.lock:
                    if track_id in self.active_tracks and self.active_tracks[track_id].startswith("Scanning"):
                        self.active_tracks[track_id] = "Unknown Person"
                        t_c = self.track_cache.setdefault(track_id, {})
                        t_c["name"] = "Unknown Person"
                        t_c["is_confirmed"] = True
                subject_name = "Unknown Person"
            else:
                last_sub_frame = self.track_last_submitted_frame.get(track_id, 0)
                is_erratic = self.jitter_engine.is_erratic_movement(track_id, threshold=80.0)
                is_in_cooldown = current_frame_num < self.limb_cooldowns.get(track_id, 0)

                # Bounding Box Contention Filter
                is_contended = (track_id in self.contended_tracks)
                if not is_contended and not is_confirmed:
                    for conf_tid, conf_info in self.recent_track_history.items():
                        if conf_tid != track_id and conf_info.get("confirmed", False):
                            if compute_box_iou(curr_box, conf_info["box"]) > 0.35 or compute_box_containment(curr_box, conf_info["box"]) > 0.40:
                                is_contended = True
                                break

                # Smart Tracker-Gating: Bypasses extraction once confirmed or while in contention
                should_extract_crop = (not is_erratic) and (not is_in_cooldown) and (not is_contended)
                should_extract_crop = should_extract_crop and (
                    ((not is_confirmed) and track_age >= 2) or (is_confirmed and (current_frame_num - last_sub_frame >= 15))
                )

                subsampler_ready = True if not is_confirmed else self.temporal_subsampler.should_process_biometrics(track_id)
                if should_extract_crop and subsampler_ready:
                    # Unconfirmed scanning tracks submit crops every 2 frames for fast identification (<0.3s).
                    # Once confirmed, interval relaxes to 15 frames to conserve CPU/GPU.
                    interval = 15 if is_confirmed else 2
                    if last_sub_frame == 0 or (current_frame_num - last_sub_frame >= interval):
                        if temp_crop is not None and crop_y2 - crop_y1 >= 80 and crop_x2 - crop_x1 >= 80:
                            self.track_last_submitted_frame[track_id] = current_frame_num
                            self._add_crop_to_embedder(track_id, temp_crop.copy(), current_frame_num)

            if subject_name is None:
                continue

            is_scanning = subject_name.startswith("Scanning")
            is_unverified = subject_name.startswith("Unverified")
            is_unknown = subject_name in ["Unknown", "Ambiguous Match", "Unknown Person"]

            # Anti-Limbo Expiration Timeout:
            # If an unconfirmed candidate has been scanning for >25 frames without any valid face detections,
            # transition from loud orange scanning alert to a muted unverified target.
            t_cache = self.track_cache.get(track_id, {})
            valid_face_count = t_cache.get("valid_face_crops", 0)
            if is_scanning and track_age > 25 and valid_face_count == 0:
                is_unverified = True
                is_scanning = False
                subject_name = f"Unverified ID {track_id}"
                with self.lock:
                    self.active_tracks[track_id] = subject_name

            # Metadata resolution
            meta_dict = {}
            if not is_scanning and not is_unverified and not is_unknown and isinstance(self.metadata, dict):
                for k, val in self.metadata.items():
                    if isinstance(val, dict) and val.get("name", "").lower() == subject_name.lower():
                        meta_dict = val
                        break

            is_contended_flag = (track_id in self.contended_tracks)
            curr_status = (
                TrackStatus.CONTENDED.value if is_contended_flag else
                (TrackStatus.UNVERIFIED.value if is_unverified else
                 (TrackStatus.SCANNING.value if is_scanning else
                  (TrackStatus.UNKNOWN.value if is_unknown else TrackStatus.CONFIRMED.value)))
            )
            session = self._get_or_create_session(track_id, curr_box, current_frame_num)
            session.status = TrackStatus(curr_status)
            session.confidence = 0.85 if is_confirmed else 0.50
            session.identity_name = subject_name if is_confirmed and not is_unknown else ""
            session.meta = meta_dict

            subject = TrackedSubject(
                track_id=track_id,
                name=subject_name,
                confidence=0.85 if is_confirmed else 0.50,
                box=curr_box,
                is_confirmed=is_confirmed,
                is_scanning=is_scanning,
                is_unknown=is_unknown,
                is_contended=is_contended_flag,
                status=curr_status,
                keypoints=self.track_keypoints.get(track_id),
                meta=meta_dict
            )
            active_subject_list.append(subject)

        # 5. Purge Expired Tracks (> 60 frames dead, or persistent headless phantom > 45 frames)
        with self.lock:
            expired_ids = []
            for tid, last_seen in list(self.track_last_seen.items()):
                age = (current_frame_num - self.track_first_seen.get(tid, current_frame_num)) + 1
                t_c = self.track_cache.get(tid, {})
                if current_frame_num - last_seen > 60:
                    expired_ids.append(tid)
                elif not t_c.get("is_confirmed", False) and t_c.get("valid_face_crops", 0) == 0 and age > 45:
                    expired_ids.append(tid)

            for tid in expired_ids:
                self.purge_track(tid)

        # 6. Visual Overlay HUD Rendering
        if annotate:
            annotated = self.annotator.annotate(annotated, active_subject_list)

        duration = max(1e-4, time.time() - start_time)
        fps = round(1.0 / duration, 1)

        return ProcessResult(
            annotated_frame=annotated,
            active_tracks=active_subject_list,
            audit_events=new_audit_events,
            frame_count=current_frame_num,
            fps=fps
        )

    # -----------------------------------------------------------------
    # Biometric Verification & Consensus Logic
    # -----------------------------------------------------------------

    def get_identity(self, idx: int) -> str:
        """Safely extract subject name from metadata across schema variants."""
        if idx < 0:
            return "Unknown"
        with self.lock:
            entry = None
            if isinstance(self.metadata, dict):
                entry = self.metadata.get(idx, self.metadata.get(str(idx), None))
            elif isinstance(self.metadata, list):
                if 0 <= idx < len(self.metadata):
                    entry = self.metadata[idx]

            if entry is None:
                return "Unknown"
            if isinstance(entry, dict):
                return str(entry.get("name", "Unknown"))
            elif isinstance(entry, str):
                return str(entry)
            return "Unknown"

    def verify_face(self, embedding: Optional[np.ndarray], face_center_ratio: float = 0.5) -> Tuple[str, float, float, float]:
        """Performs true Top-2 inter-subject margin gating and vector matching."""
        if embedding is None:
            return "Unknown", 999.0, 999.0, 0.0

        embedding = np.array(embedding, dtype=np.float32).flatten()
        if embedding.size != 512:
            return "Unknown", 999.0, 999.0, 0.0

        if face_center_ratio < 0.15 or face_center_ratio > 0.85:
            return "Ambiguous Match", 999.0, 999.0, 0.0

        quality_norm = float(np.linalg.norm(embedding))
        if quality_norm < 0.3:
            return "Unknown", 999.0, 999.0, quality_norm

        norm_embedding = embedding / (quality_norm + 1e-10)
        query_vector = np.expand_dims(norm_embedding, axis=0)

        with self.lock:
            total_in_index = self.index_adapter.ntotal
            if total_in_index == 0:
                return "Unknown", 999.0, 999.0, quality_norm

            k_search = min(10, total_in_index)
            distances, indices = self.index_adapter.search(query_vector, k=k_search)

        if distances.shape[1] == 0:
            return "Unknown", 999.0, 999.0, quality_norm

        dist1 = float(distances[0][0])
        idx1 = int(indices[0][0])
        name1 = self.get_identity(idx1)

        if dist1 > self.l2_threshold or name1 in ["Unknown", ""]:
            return "Unknown", dist1, 999.0, quality_norm

        # Find the first nearest neighbor belonging to a DIFFERENT subject (Inter-Subject Gating)
        dist_diff_subject = 999.0
        diff_subject_found = False

        for j in range(1, distances.shape[1]):
            idx_j = int(indices[0][j])
            if idx_j < 0:
                continue
            name_j = self.get_identity(idx_j)
            if name_j not in ["Unknown", ""] and name_j.lower() != name1.lower():
                dist_diff_subject = float(distances[0][j])
                diff_subject_found = True
                break

        # Only enforce margin threshold if a distinct subject was found in top-k
        if diff_subject_found:
            margin = abs(dist_diff_subject - dist1)
            if margin < self.margin_threshold:
                return "Ambiguous Match", dist1, dist_diff_subject, quality_norm
            dist2 = dist_diff_subject
        else:
            dist2 = 999.0

        return name1, dist1, dist2, quality_norm

    def process_track_frame(
        self,
        track_id: int,
        embedding: Optional[np.ndarray],
        face_center_ratio: float = 0.5,
        track_age: Optional[int] = None,
        active_track_ids: Optional[List[int]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Multi-frame consensus voting and track identity locking protocol."""
        with self.lock:
            # Clean up stale claims where the claimant is no longer active
            stale_claims = []
            for name, claimant_tid in list(self.active_identity_claims.items()):
                if claimant_tid == track_id:
                    continue
                last_seen_frame = self.track_last_seen.get(claimant_tid, 0)
                if (active_track_ids is not None and claimant_tid not in active_track_ids) or (self.frame_count - last_seen_frame > 15):
                    stale_claims.append(name)
            for name in stale_claims:
                self.active_identity_claims.pop(name, None)

            if active_track_ids is not None:
                dead_tids = [tid for tid in list(self.track_cache.keys()) if tid not in active_track_ids and tid != track_id]
                for tid in dead_tids:
                    if self.frame_count - self.track_last_seen.get(tid, 0) > 30:
                        self.purge_track(tid)

            track = self.track_cache.setdefault(track_id, {})
            track.setdefault("name", f"Scanning Track {track_id}")
            track.setdefault("is_confirmed", False)
            track.setdefault("is_spatial_collision", False)
            track.setdefault("best_norm", 0.0)
            track.setdefault("history", [])
            track.setdefault("attempts", 0)
            track.setdefault("valid_face_crops", 0)
            track.setdefault("failed_attempts", 0)
            track.setdefault("age", 0)
            track["age"] += 1

            effective_age = track_age if track_age is not None else max(3, track["age"])

            # Confirmed Identity Hysteresis Lock-In
            if track["is_confirmed"] and track["name"] not in ["Unknown Person", "Scanning", "Ambiguous Match"] and not track["name"].startswith("Scanning"):
                self.active_identity_claims[track["name"]] = track_id
                return track["name"], {
                    "bypassed": True,
                    "norm": float(np.linalg.norm(embedding)) if embedding is not None else 0.0,
                    "cached": track["name"],
                    "confirmed": True
                }

            # Smart bypass for confirmed Unknown Person
            if track["is_confirmed"] and track["name"] == "Unknown Person":
                if embedding is not None:
                    track["is_confirmed"] = False
                    track["is_spatial_collision"] = False
                    track["name"] = f"Scanning Track {track_id}"
                    track["attempts"] = 0
                    track["valid_face_crops"] = 0
                    track["failed_attempts"] = 0
                    track["history"] = []
                else:
                    return "Unknown Person", {
                        "bypassed": True,
                        "norm": 0.0,
                        "cached": "Unknown Person"
                    }

            # Tentative Track Gating (effective_age < 3)
            if effective_age < 3 and not track["is_confirmed"]:
                track["name"] = f"Scanning Track {track_id}"
                return f"Scanning Track {track_id}", {"tentative": True, "age": effective_age}

            # Non-face crop handling
            if embedding is None:
                track["failed_attempts"] += 1
                return track["name"], {"failed": True, "confirmed": track["is_confirmed"]}

            embedding = np.array(embedding, dtype=np.float32).flatten()
            matched_name, d1, d2, current_norm = self.verify_face(embedding, face_center_ratio)
            track["valid_face_crops"] += 1
            track["attempts"] += 1

            if matched_name not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]:
                existing_claimant = self.active_identity_claims.get(matched_name)
                if existing_claimant is not None and existing_claimant != track_id:
                    # Check if the existing claimant is genuinely alive and recently seen
                    claimant_seen = self.track_last_seen.get(existing_claimant, 0)
                    is_claimant_live = (active_track_ids is None or existing_claimant in active_track_ids) and (self.frame_count - claimant_seen <= 15)
                    if is_claimant_live:
                        matched_name = "Ambiguous Match"
                    else:
                        # Existing claimant was lost or dead; seamlessly transfer claim to active track
                        self.active_identity_claims[matched_name] = track_id
                else:
                    self.active_identity_claims[matched_name] = track_id

            if current_norm >= track["best_norm"] * 0.85 and matched_name not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]:
                track["history"].append(matched_name)
                if len(track["history"]) > self.history_window:
                    track["history"].pop(0)
                if current_norm > track["best_norm"]:
                    track["best_norm"] = current_norm

            history = track["history"]
            if len(history) >= self.consensus_votes:
                valid_votes = [n for n in history if n not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]]
                if valid_votes:
                    most_common = max(set(valid_votes), key=valid_votes.count)
                    vote_count = valid_votes.count(most_common)
                    if vote_count >= self.consensus_votes:
                        existing_claimant = self.active_identity_claims.get(most_common)
                        claimant_seen = self.track_last_seen.get(existing_claimant, 0) if existing_claimant is not None else 0
                        is_claimant_live = (existing_claimant is not None and existing_claimant != track_id and 
                                            (active_track_ids is None or existing_claimant in active_track_ids) and 
                                            (self.frame_count - claimant_seen <= 15))
                        if not is_claimant_live:
                            track["name"] = most_common
                            track["is_confirmed"] = True
                            self.active_identity_claims[most_common] = track_id
                        else:
                            track["name"] = "Ambiguous Match"

            # Require at least 3 valid face crops before categorizing as "Unknown Person"
            if not track["is_confirmed"] and track["valid_face_crops"] >= 3:
                track["name"] = "Unknown Person"
                track["is_confirmed"] = True

            current_display = track["name"] if track["is_confirmed"] else matched_name
            if current_display in ["Unknown", "Unknown Person", "Ambiguous Match"] and len(track["history"]) > 0:
                current_display = track["history"][-1]
            elif current_display in ["Unknown", "Ambiguous Match"]:
                if track["valid_face_crops"] >= 3 or track["is_confirmed"]:
                    current_display = "Unknown Person"
                else:
                    current_display = f"Scanning Track {track_id}"

            telemetry = {
                "bypassed": False,
                "current_match": matched_name,
                "dist1": d1,
                "dist2": d2,
                "norm": current_norm,
                "best_norm": track["best_norm"],
                "history": list(track["history"]),
                "confirmed": track["is_confirmed"],
                "attempts": track["attempts"],
                "valid_face_crops": track["valid_face_crops"]
            }

            return current_display, telemetry

    def purge_track(self, track_id: int) -> None:
        """Purges track state across all internal tables and sessions."""
        with self.lock:
            self._sessions.pop(track_id, None)
            self.active_tracks.pop(track_id, None)
            self.track_last_seen.pop(track_id, None)
            self.track_first_seen.pop(track_id, None)
            self.track_last_submitted_frame.pop(track_id, None)
            self.track_keypoints.pop(track_id, None)
            self.limb_cooldowns.pop(track_id, None)
            self.recent_track_history.pop(track_id, None)
            self.contended_tracks.pop(track_id, None)
            self.head_presence_cache.pop(track_id, None)
            if track_id in self.track_cache:
                del self.track_cache[track_id]
            claims_to_remove = [name for name, claimant in self.active_identity_claims.items() if claimant == track_id]
            for name in claims_to_remove:
                del self.active_identity_claims[name]
            self.jitter_engine.reset_track(track_id)
            self.temporal_subsampler.reset_track(track_id)

    # -----------------------------------------------------------------
    # Embedder Worker & Callback
    # -----------------------------------------------------------------

    def extract_face_embedding(self, img_or_crop: np.ndarray, align: bool = True) -> Optional[np.ndarray]:
        """
        Unified biometric extraction pipeline used across live tracking and enrollment.
        Aligns raw face using YuNet (Umeyama similarity transform) and extracts normalized 512D ArcFace vector.
        """
        return self.embedder.embed(img_or_crop, align=align, strict_alignment=False, upscale=False)

    def _add_crop_to_embedder(self, track_id: int, crop: np.ndarray, frame_num: int):
        with self._embedder_lock:
            # Replace existing crop for this track with fresh one
            self._embedder_queue = [item for item in self._embedder_queue if item["track_id"] != track_id]
            if len(self._embedder_queue) >= 16:
                self._embedder_queue.pop()
            self._embedder_queue.insert(0, {
                "track_id": track_id,
                "crop": crop,
                "frame_added": frame_num
            })

    def _embedder_worker(self):
        while self._embedder_running:
            self.flush()
            time.sleep(0.005)

    def flush(self) -> None:
        """Processes any queued embedding crops synchronously (for tests and shutdown)."""
        with self._embedder_lock:
            if not self._embedder_queue:
                return
            batch_to_process = list(self._embedder_queue)
            self._embedder_queue.clear()

        crops_to_sanitize = []
        embeddings: List[Optional[np.ndarray]] = []
        ratios: List[Optional[Tuple[float, float, float, float]]] = []

        for item in batch_to_process:
            crop = item["crop"]
            crops_to_sanitize.append(crop)
            emb = self.embedder.embed(crop, align=True, strict_alignment=True, upscale=True)
            embeddings.append(emb)
            ratios.append((0.1, 0.1, 0.8, 0.8) if emb is not None else None)

        for i, item in enumerate(batch_to_process):
            self._handle_embedding_result(item["track_id"], embeddings[i], ratios[i])

        GDPRMemorySanitizer.sanitize_crop_list(crops_to_sanitize)

    def _handle_embedding_result(self, track_id: int, emb: Optional[np.ndarray], ratio: Any):
        with self.lock:
            t_cache = self.track_cache.setdefault(track_id, {
                "failed_attempts": 0,
                "is_confirmed": False,
                "name": f"Scanning Track {track_id}"
            })

            if emb is None:
                t_cache["failed_attempts"] = t_cache.get("failed_attempts", 0) + 1
                if not t_cache.get("is_confirmed", False):
                    self.active_tracks.setdefault(track_id, f"Scanning Track {track_id}")
                if t_cache["failed_attempts"] >= 3:
                    self.limb_cooldowns[track_id] = self.frame_count + 10
                return

            track_age = (self.frame_count - self.track_first_seen.get(track_id, self.frame_count)) + 1
            display_name, telemetry = self.process_track_frame(
                track_id, emb, face_center_ratio=0.5, track_age=track_age, active_track_ids=list(self.active_tracks.keys())
            )

            if track_id in self.track_cache:
                self.track_cache[track_id]["failed_attempts"] = 0

            if display_name in ["Unknown", "Ambiguous Match"]:
                display_name = "Unknown Person"

            self.active_tracks[track_id] = display_name

            if telemetry.get("confirmed") and display_name not in ["Unknown", "Ambiguous Match", "Unknown Person"]:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_event_frame = self.identified_cooldowns.get(display_name, 0)
                if self.frame_count - last_event_frame > self.cooldown_frames:
                    self.identified_cooldowns[display_name] = self.frame_count
                    dist1 = telemetry.get("dist1", 0.0)
                    self.encrypted_audit_logger.log_event(now_str, track_id, display_name, dist1)

    # -----------------------------------------------------------------
    # Telemetry, Snapshot & Mode Interfaces
    # -----------------------------------------------------------------

    def get_active_tracks(self) -> List[TrackedSubject]:
        """Thread-safe snapshot of currently active subjects."""
        with self.lock:
            result = []
            for tid, name in self.active_tracks.items():
                if str(name).startswith("Scanning"):
                    continue
                meta_info = {}
                if isinstance(self.metadata, dict):
                    for k, val in self.metadata.items():
                        if isinstance(val, dict) and val.get("name", "").lower() == str(name).lower():
                            meta_info = val
                            break
                is_conf = self.track_cache.get(tid, {}).get("is_confirmed", False)
                is_cont = (tid in self.contended_tracks)
                box = self.recent_track_history.get(tid, {}).get("box", (0, 0, 0, 0))
                result.append(TrackedSubject(
                    track_id=int(tid),
                    name=str(name),
                    confidence=0.85 if is_conf else 0.5,
                    box=box,
                    is_confirmed=is_conf,
                    is_scanning=False,
                    is_unknown=(str(name) == "Unknown Person"),
                    is_contended=is_cont,
                    keypoints=self.track_keypoints.get(tid),
                    meta=meta_info
                ))
            return result

    def get_telemetry_snapshot(self) -> Dict[str, Any]:
        """Atomic read-only telemetry snapshot for dashboard and WebSocket streaming."""
        with self.lock:
            active_list = [
                {
                    "track_id": s.track_id,
                    "name": s.name,
                    "match_distance": 0.45,
                    "meta": s.meta,
                    "keypoints": s.keypoints
                }
                for s in self.get_active_tracks()
            ]
            total_prof = self.index_adapter.ntotal
            return {
                "active_tracks": active_list,
                "total_profiles": total_prof,
                "pose_only_mode": self._pose_only_mode,
                "frame_count": self.frame_count
            }

    def set_pose_only(self, enabled: bool) -> None:
        """Controls privacy-compliant pose-only estimation mode."""
        with self.lock:
            self._pose_only_mode = enabled
            print(f"[BiometricEngine] PoseOnlyTracking Mode set to: {enabled}")

    @property
    def is_pose_only(self) -> bool:
        return self._pose_only_mode

    def stop(self) -> None:
        """Gracefully terminates background worker threads."""
        self._embedder_running = False
        if self._embedder_thread.is_alive():
            self._embedder_thread.join(timeout=1.0)
        if hasattr(self.encrypted_audit_logger, 'stop'):
            self.encrypted_audit_logger.stop()
        print("[BiometricEngine] Engine stopped cleanly.")


# =====================================================================
# Canonical Backward-Compatibility Aliases
# =====================================================================
BiometricVerificationEngine = BiometricTrackingEngine
BiometricVerificationEngineV5 = BiometricTrackingEngine


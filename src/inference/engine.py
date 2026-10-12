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
from src.rendering.annotator import (
    Annotator,
    CyberpunkHUDAnnotator,
    FrameAnnotator,
    NoOpAnnotator,
    ScriptedAnnotator
)
from src.domain.subject_session import SubjectSession, TrackStatus
from src.domain.track_registry import (
    TrackRegistry,
    ActiveTracksProxy,
    TrackCacheProxy,
    TrackLastSeenProxy,
    TrackFirstSeenProxy,
    TrackKeypointsProxy
)
from src.inference.consensus import (
    IdentityConsensus,
    MatchCandidate,
    ConsensusDecision
)
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
)
from src.database.audit_repository import (
    AuditLogger,
    EncryptedWALAuditLogger,
    ScriptedAuditLogger,
    AESEncryptedWALAuditLogger,
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
from src.inference.tracking_pre_filter import (
    TrackingPreFilter,
    PreFilterResult,
    FilteredCandidate,
    compute_box_iou,
    compute_box_containment,
)
from src.inference.biometric_scheduler import BiometricIngestionScheduler


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
        audit_logger: Optional[AuditLogger] = None,
        pose_estimator: Optional[PoseOnlyEstimator] = None,
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
        annotator: Optional[Annotator] = None,
        consensus: Optional[IdentityConsensus] = None,
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

        # Identity Consensus Seam
        self.consensus: IdentityConsensus = consensus or IdentityConsensus(
            consensus_votes=consensus_votes,
            history_window=history_window,
            margin_threshold=margin_threshold,
            side_profile_threshold=0.30,
            max_scanning_attempts=12,
            min_unknown_crops=12
        )

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

        # Person Detector Seam (Lazy loaded unless provided)
        self._detector: Optional[PersonDetector] = detector

        self.jitter_engine = JitterFilterEngine(erratic_threshold=80.0)
        self.face_aligner = YuNetFaceAligner()
        self.fsrcnn_upscaler = FSRCNNUpscaler()

        # Face Embedder Seam (Lazy loaded unless provided)
        self._embedder: Optional[FaceEmbedder] = embedder

        self.temporal_subsampler = TemporalSubsampler(subsample_interval=3)

        # Unified Audit Logger Seam (EncryptedWALAuditLogger or Scripted test adapter)
        if audit_logger is not None:
            self.audit_logger = audit_logger
        else:
            self.audit_logger = EncryptedWALAuditLogger(db_path=self.db_path)
        self.encrypted_audit_logger = self.audit_logger  # Backward-compatibility alias

        # Lazy Pose Estimator
        self._pose_estimator: Optional[PoseOnlyEstimator] = pose_estimator
        self.annotator: Annotator = annotator if annotator is not None else CyberpunkHUDAnnotator()

        # Privacy mode flag
        self._pose_only_mode = False

        # Deep Domain Seam: TrackRegistry manages all SubjectSession state
        self.registry = TrackRegistry()
        self.frame_count = 0
        self.identified_cooldowns: Dict[str, int] = {}

        # Deep Ingestion & Pre-Filtering Modules
        self.pre_filter = TrackingPreFilter(
            face_aligner=self.face_aligner,
            pose_estimator=self.pose_estimator,
            jitter_engine=self.jitter_engine,
        )
        self.biometric_scheduler = BiometricIngestionScheduler(
            jitter_engine=self.jitter_engine,
            temporal_subsampler=self.temporal_subsampler,
        )

        # Batch Face Embedder Setup
        self._embedder_queue: List[Dict[str, Any]] = []
        self._embedder_lock = threading.RLock()
        self._embedder_running = True

        self._embedder_thread = threading.Thread(target=self._embedder_worker, daemon=True)
        self._embedder_thread.start()

    @property
    def active_identity_claims(self) -> Dict[str, int]:
        """Active identity leases managed by IdentityConsensus."""
        return self.consensus.active_identity_claims

    @active_identity_claims.setter
    def active_identity_claims(self, value: Dict[str, int]) -> None:
        self.consensus.active_identity_claims = value

    @property
    def detector(self) -> PersonDetector:
        if self._detector is None:
            self._detector = UltralyticsPersonDetector(
                tracker_cfg=self.bytetrack_cfg,
                enable_haar=True
            )
        return self._detector

    @detector.setter
    def detector(self, value: PersonDetector) -> None:
        self._detector = value

    @property
    def embedder(self) -> FaceEmbedder:
        if self._embedder is None:
            self._embedder = ArcFaceEmbedder(
                model_name=self.model_name,
                face_aligner=self.face_aligner,
                upscaler=self.fsrcnn_upscaler
            )
        return self._embedder

    @embedder.setter
    def embedder(self, value: FaceEmbedder) -> None:
        self._embedder = value

    @property
    def pose_estimator(self) -> Optional[PoseOnlyEstimator]:
        if self._pose_estimator is None:
            try:
                self._pose_estimator = PoseOnlyEstimator(pose_model_spec=DEFAULT_MODEL_CONFIG.get_effective_pose_model())
            except Exception as e:
                print(f"[BiometricTrackingEngine Warning] Lazy PoseEstimator init failed: {e}")
                self._pose_estimator = None
        return self._pose_estimator

    @pose_estimator.setter
    def pose_estimator(self, value: Optional[PoseOnlyEstimator]) -> None:
        self._pose_estimator = value
        if hasattr(self, "pre_filter") and self.pre_filter is not None:
            self.pre_filter.pose_estimator = value

    # Backward-compatibility property facades for tests inspecting internal tables
    @property
    def head_presence_cache(self) -> Dict[int, int]:
        return self.pre_filter.head_presence_cache

    @head_presence_cache.setter
    def head_presence_cache(self, val: Dict[int, int]) -> None:
        self.pre_filter.head_presence_cache = val

    @property
    def contended_tracks(self) -> Dict[int, int]:
        return self.pre_filter.contended_tracks

    @contended_tracks.setter
    def contended_tracks(self, val: Dict[int, int]) -> None:
        self.pre_filter.contended_tracks = val

    @property
    def contention_blacklist(self) -> Dict[int, int]:
        return self.pre_filter.contention_blacklist

    @contention_blacklist.setter
    def contention_blacklist(self, val: Dict[int, int]) -> None:
        self.pre_filter.contention_blacklist = val

    @property
    def recent_track_history(self) -> Dict[int, Dict[str, Any]]:
        return self.pre_filter.recent_track_history

    @recent_track_history.setter
    def recent_track_history(self, val: Dict[int, Dict[str, Any]]) -> None:
        self.pre_filter.recent_track_history = val

    @property
    def limb_cooldowns(self) -> Dict[int, int]:
        return self.biometric_scheduler.limb_cooldowns

    @limb_cooldowns.setter
    def limb_cooldowns(self, val: Dict[int, int]) -> None:
        self.biometric_scheduler.limb_cooldowns = val

    @property
    def track_last_submitted_frame(self) -> Dict[int, int]:
        return self.biometric_scheduler.track_last_submitted_frame

    @track_last_submitted_frame.setter
    def track_last_submitted_frame(self, val: Dict[int, int]) -> None:
        self.biometric_scheduler.track_last_submitted_frame = val

    @property
    def _sessions(self) -> Dict[int, SubjectSession]:
        return self.registry._sessions

    @_sessions.setter
    def _sessions(self, val: Dict[int, SubjectSession]) -> None:
        self.registry._sessions = val

    @property
    def active_tracks(self) -> ActiveTracksProxy:
        return self.registry.active_tracks

    @active_tracks.setter
    def active_tracks(self, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.registry.active_tracks[k] = v

    @property
    def track_cache(self) -> TrackCacheProxy:
        return self.registry.track_cache

    @track_cache.setter
    def track_cache(self, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.registry.track_cache[k] = v

    @property
    def track_last_seen(self) -> TrackLastSeenProxy:
        return self.registry.track_last_seen

    @track_last_seen.setter
    def track_last_seen(self, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.registry.track_last_seen[k] = v

    @property
    def track_first_seen(self) -> TrackFirstSeenProxy:
        return self.registry.track_first_seen

    @track_first_seen.setter
    def track_first_seen(self, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.registry.track_first_seen[k] = v

    @property
    def track_keypoints(self) -> TrackKeypointsProxy:
        return self.registry.track_keypoints

    @track_keypoints.setter
    def track_keypoints(self, val: Any) -> None:
        if isinstance(val, dict):
            for k, v in val.items():
                self.registry.track_keypoints[k] = v

    def _get_or_create_session(self, track_id: int, box: Tuple[int, int, int, int], frame_num: int) -> SubjectSession:
        return self.registry.get_or_create(track_id, box, frame_num)

    def get_session(self, track_id: int) -> Optional[SubjectSession]:
        """Thread-safe retrieval of a specific active SubjectSession."""
        return self.registry.get(track_id)

    def get_all_sessions(self) -> Dict[int, SubjectSession]:
        """Thread-safe snapshot of all active SubjectSession instances."""
        return dict(self.registry._sessions)

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
        if self._detector is None:
            self._detector = UltralyticsPersonDetector(tracker_cfg=self.bytetrack_cfg, enable_haar=True)

    @property
    def _keras_model(self) -> Any:
        if self._embedder is not None:
            return getattr(self._embedder, "keras_model", None)
        return None

    @_keras_model.setter
    def _keras_model(self, model: Any) -> None:
        if self._embedder is not None and hasattr(self._embedder, "keras_model"):
            self._embedder.keras_model = model

    def _init_embedder_model(self):
        if self._embedder is not None and hasattr(self._embedder, "_init_model"):
            self._embedder._init_model()

    def reset_state(self) -> None:
        """Flushes all active tracking sessions, track caches, and consensus histories."""
        with self.lock:
            self.active_tracks.clear()
            self._sessions.clear()
            self.track_last_seen.clear()
            self.track_first_seen.clear()
            self.track_keypoints.clear()
            self.identified_cooldowns.clear()
            self.track_cache.clear()
            if hasattr(self, "consensus") and hasattr(self.consensus, "reset"):
                self.consensus.reset()
            if hasattr(self, "jitter_engine") and hasattr(self.jitter_engine, "track_positions"):
                self.jitter_engine.track_positions.clear()
            print("[BiometricEngine] State successfully reset: flushed tracks and consensus caches.")

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
        """Validates upper-body or head evidence via TrackingPreFilter deep module."""
        return self.pre_filter.has_head_or_shoulder_presence(frame, box, track_id, current_frame_num)

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
        annotated = frame.copy() if (annotate and not isinstance(self.annotator, NoOpAnnotator)) else frame

        # 1. Motion Tracking via PersonDetector Seam
        detections = self.detector.detect_and_track(small_frame)

        # 2. TrackingPreFilter Deep Seam: Jitter Filtering, NMS & Crossover Contention
        with self.lock:
            active_tids = set(self.active_tracks.keys())
            confirmed_tids = {tid for tid, c in self.track_cache.items() if c.get("is_confirmed", False)}
            for tid, sess in self._sessions.items():
                if sess.is_confirmed:
                    confirmed_tids.add(tid)

        pre_filter_res = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=detections,
            scale_x=scale_x,
            scale_y=scale_y,
            current_frame_num=current_frame_num,
            active_track_ids=active_tids,
            confirmed_track_ids=confirmed_tids,
        )

        current_frame_boxes = pre_filter_res.current_frame_boxes
        active_subject_list: List[TrackedSubject] = []
        new_audit_events: List[Dict[str, Any]] = []

        with self.lock:
            # Purge suppressed active scanning tracks
            for s_tid in pre_filter_res.suppressed_track_ids:
                if s_tid in self.active_tracks and self.active_tracks[s_tid].startswith("Scanning"):
                    self.purge_track(s_tid)

            # Handle resolved contention tracks
            for r_tid in pre_filter_res.resolved_contended_track_ids:
                if r_tid in self._sessions:
                    self._sessions[r_tid].reset_contention()
                t_c = self.track_cache.setdefault(r_tid, {})
                if not t_c.get("is_confirmed", False):
                    self.active_tracks[r_tid] = f"Scanning Track {r_tid}"
                    t_c["name"] = f"Scanning Track {r_tid}"
                    t_c["is_confirmed"] = False

            # Handle newly contended tracks & audit events
            for tid in pre_filter_res.newly_contended_track_ids:
                session = self._get_or_create_session(tid, current_frame_boxes[tid], current_frame_num)
                t_cache = self.track_cache.setdefault(tid, {})
                if t_cache.get("is_confirmed", False) or session.is_confirmed:
                    # Strong Identity Persistence: Confirmed track retains identity lock
                    continue
                session.enter_contention(blacklist_expiration_frame=current_frame_num + 10)
                t_cache["is_confirmed"] = False
                t_cache["history"] = []
                self.active_tracks[tid] = f"Contended Track {tid}"

            for tid1, tid2, iou_val in pre_filter_res.newly_contended_pairs:
                new_audit_events.append({
                    "event_type": "CROSSOVER_CONTENTION",
                    "frame": current_frame_num,
                    "track_id_1": tid1,
                    "track_id_2": tid2,
                    "iou": round(float(iou_val), 3),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "message": f"Tracks {tid1} and {tid2} entered crossover contention (IoU: {iou_val:.2f}). Identities unlocked."
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

                session = self._get_or_create_session(track_id, curr_box, current_frame_num)
                session.mark_pose(kpts)
                active_subject_list.append(session.to_tracked_subject())
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
                        t_c["is_confirmed"] = True
                        t_c["best_dist"] = 0.35
                        t_c["best_norm"] = 1.0
                        session = self._get_or_create_session(track_id, curr_box, current_frame_num)
                        session.confirm_identity(inherited_name, distance=0.35, norm=1.0)
                        self.consensus.active_identity_claims[inherited_name] = track_id
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

            # Biometric Scheduler Deep Seam: Governs extraction admission, cooldowns, and cadence
            session = self._get_or_create_session(track_id, curr_box, current_frame_num)
            is_contended = (track_id in self.contended_tracks)
            if not is_contended and not is_confirmed:
                for conf_tid, conf_info in self.recent_track_history.items():
                    if conf_tid != track_id and conf_info.get("confirmed", False):
                        if compute_box_iou(curr_box, conf_info["box"]) > 0.35 or compute_box_containment(curr_box, conf_info["box"]) > 0.40:
                            is_contended = True
                            break

            crop_shape = (crop_y2 - crop_y1, crop_x2 - crop_x1) if temp_crop is not None else None
            if self.biometric_scheduler.should_extract(session, crop_shape, current_frame_num, is_contended=is_contended):
                self.biometric_scheduler.record_submission(track_id, current_frame_num)
                self._add_crop_to_embedder(track_id, temp_crop.copy(), current_frame_num)

            if subject_name is None:
                continue

            is_contended_flag = (track_id in self.contended_tracks)
            if is_contended_flag and not is_confirmed:
                subject_name = f"Contended Track {track_id}"

            is_scanning = subject_name.startswith("Scanning") or (subject_name.startswith("Contended") and not is_contended_flag)
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
            if not is_scanning and not is_unverified and not is_unknown and not is_contended_flag and isinstance(self.metadata, dict):
                for k, val in self.metadata.items():
                    if isinstance(val, dict) and val.get("name", "").lower() == subject_name.lower():
                        meta_dict = val
                        break
            curr_status = (
                TrackStatus.CONTENDED.value if is_contended_flag else
                (TrackStatus.UNVERIFIED.value if is_unverified else
                 (TrackStatus.SCANNING.value if is_scanning else
                  (TrackStatus.UNKNOWN.value if is_unknown else TrackStatus.CONFIRMED.value)))
            )
            session = self._get_or_create_session(track_id, curr_box, current_frame_num)
            session.status = TrackStatus(curr_status)
            session.confidence = 0.85 if is_confirmed else 0.50
            session.identity_name = f"Contended Track {track_id}" if is_contended_flag else (subject_name if is_confirmed and not is_unknown else (subject_name if not is_scanning else ""))
            session.keypoints = self.track_keypoints.get(track_id)
            session.meta = meta_dict

            # Keep legacy cache and active track dictionaries in sync
            self.track_cache[track_id] = session.to_cache_dict()
            self.active_tracks[track_id] = session.display_name

            active_subject_list.append(session.to_tracked_subject(meta=meta_dict))

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
        if annotate and not isinstance(self.annotator, NoOpAnnotator):
            duration_so_far = max(1e-4, time.time() - start_time)
            fps_so_far = round(1.0 / duration_so_far, 1)
            annotated = self.annotator.annotate(annotated, active_subject_list, fps=fps_so_far)

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

    def verify_face(
        self,
        embedding: Optional[np.ndarray],
        face_center_ratio: float = 0.5,
        is_side_profile: bool = False
    ) -> Tuple[str, float, float, float]:
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

        # Pose-aware strict threshold: side profile crops require tighter distance (squared L2 < 0.30 vs default 0.68)
        effective_threshold = min(self.l2_threshold, 0.30) if is_side_profile else self.l2_threshold
        if dist1 > effective_threshold or name1 in ["Unknown", ""]:
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
        effective_margin = max(self.margin_threshold, 0.20) if is_side_profile else self.margin_threshold
        if diff_subject_found:
            margin = abs(dist_diff_subject - dist1)
            if margin < effective_margin:
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
        active_track_ids: Optional[List[int]] = None,
        is_side_profile: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Multi-frame consensus voting and track identity locking protocol."""
        with self.lock:
            # Clean up dead tracks if active_track_ids provided
            if active_track_ids is not None:
                dead_tids = [tid for tid in list(self.track_last_seen.keys()) if tid not in active_track_ids and tid != track_id]
                for tid in dead_tids:
                    if self.frame_count - self.track_last_seen.get(tid, 0) > 30:
                        self.purge_track(tid)

            # Retrieve or create session
            session = self._sessions.get(track_id)
            if session is None:
                box = self.track_cache.get(track_id, {}).get("box", (0, 0, 0, 0))
                session = self._get_or_create_session(track_id, box, self.frame_count)
            else:
                session.last_seen = max(session.last_seen, self.frame_count)

            if track_age is not None:
                session.first_seen = session.last_seen - track_age + 1

            # Prepare match candidate if embedding is present
            candidate = None
            if embedding is not None:
                embedding = np.array(embedding, dtype=np.float32).flatten()
                matched_name, d1, d2, current_norm = self.verify_face(
                    embedding, face_center_ratio, is_side_profile=is_side_profile
                )
                candidate = MatchCandidate(
                    name=matched_name,
                    dist1=d1,
                    dist2=d2,
                    norm=current_norm,
                    is_side_profile=is_side_profile
                )

            # Delegate to IdentityConsensus
            last_seen_fn = lambda tid: self.track_last_seen.get(tid, 0)
            decision = self.consensus.evaluate(
                session=session,
                candidate=candidate,
                current_frame=self.frame_count,
                active_track_ids=active_track_ids,
                last_seen_lookup=last_seen_fn,
                effective_age=track_age
            )

            # Keep track_cache synchronized for backwards-compatibility
            self.track_cache[track_id] = session.to_cache_dict()
            self.track_cache[track_id]["history"] = list(self.consensus.get_history(track_id))

            telemetry = dict(decision.telemetry)
            telemetry.setdefault("bypassed", decision.telemetry.get("bypassed", False))
            telemetry.setdefault("confirmed", decision.is_confirmed)
            telemetry.setdefault("norm", session.best_norm)
            telemetry.setdefault("best_norm", session.best_norm)
            telemetry.setdefault("history", list(self.consensus.get_history(track_id)))
            telemetry.setdefault("attempts", session.attempts)
            telemetry.setdefault("valid_face_crops", session.valid_face_crops)
            if candidate is not None:
                telemetry.setdefault("current_match", candidate.name)
                telemetry.setdefault("dist1", candidate.dist1)
                telemetry.setdefault("dist2", candidate.dist2)

            return decision.display_name, telemetry

    def purge_track(self, track_id: int) -> None:
        """Purges track state across all internal tables, sessions, and deep modules."""
        with self.lock:
            self.registry.purge(track_id)
            self.consensus.purge_track(track_id)
            self.pre_filter.purge_track(track_id)
            self.biometric_scheduler.purge_track(track_id)
            if self.jitter_engine is not None and hasattr(self.jitter_engine, "reset_track"):
                self.jitter_engine.reset_track(track_id)
            if self.temporal_subsampler is not None and hasattr(self.temporal_subsampler, "reset_track"):
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
        is_sides: List[bool] = []

        for item in batch_to_process:
            crop = item["crop"]
            crops_to_sanitize.append(crop)
            is_side = False
            if hasattr(self.embedder, "is_side_profile"):
                try:
                    is_side = self.embedder.is_side_profile(crop)
                except Exception:
                    is_side = False
            emb = self.embedder.embed(crop, align=True, strict_alignment=True, upscale=True)
            embeddings.append(emb)
            is_sides.append(is_side)
            ratios.append((0.1, 0.1, 0.8, 0.8) if emb is not None else None)

        for i, item in enumerate(batch_to_process):
            self._handle_embedding_result(item["track_id"], embeddings[i], ratios[i], is_side_profile=is_sides[i])

        GDPRMemorySanitizer.sanitize_crop_list(crops_to_sanitize)
        if hasattr(self.audit_logger, "flush"):
            self.audit_logger.flush()

    def _handle_embedding_result(self, track_id: int, emb: Optional[np.ndarray], ratio: Any, is_side_profile: bool = False):
        with self.lock:
            # If the track is currently in crossover contention, drop in-flight embedding.
            # Prevents face extraction from confirming identity or turning green during occlusion.
            if track_id in self.contended_tracks:
                return

            t_cache = self.track_cache.setdefault(track_id, {
                "failed_attempts": 0,
                "is_confirmed": False,
                "name": f"Scanning Track {track_id}"
            })

            if emb is None:
                t_cache["failed_attempts"] = t_cache.get("failed_attempts", 0) + 1
                if track_id in self._sessions:
                    self._sessions[track_id].failed_attempts = t_cache["failed_attempts"]
                if not t_cache.get("is_confirmed", False):
                    self.active_tracks.setdefault(track_id, f"Scanning Track {track_id}")
                if t_cache["failed_attempts"] >= 3:
                    self.limb_cooldowns[track_id] = self.frame_count + 10
                    if track_id in self._sessions:
                        self._sessions[track_id].limb_cooldown_until = self.frame_count + 10
                return

            track_age = (self.frame_count - self.track_first_seen.get(track_id, self.frame_count)) + 1
            display_name, telemetry = self.process_track_frame(
                track_id, emb, face_center_ratio=0.5, track_age=track_age, active_track_ids=list(self.active_tracks.keys()),
                is_side_profile=is_side_profile
            )

            if track_id in self.track_cache:
                self.track_cache[track_id]["failed_attempts"] = 0

            self.active_tracks[track_id] = display_name
            if track_id in self._sessions:
                session = self._sessions[track_id]
                session.failed_attempts = 0
                if telemetry.get("confirmed", False):
                    if display_name == "Unknown Person":
                        session.mark_unknown()
                    elif not display_name.startswith("Scanning") and not display_name.startswith("Unverified") and not display_name.startswith("Contended"):
                        session.confirm_identity(display_name, telemetry.get("dist1", 0.5), telemetry.get("norm", 1.0))
                else:
                    if session.is_confirmed and session.status == TrackStatus.UNKNOWN and not display_name.startswith("Unknown"):
                        session.status = TrackStatus.SCANNING
                    session.identity_name = display_name

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
                if str(name).startswith("Scanning") or str(name).startswith("Unverified") or str(name) == "Unknown Person":
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
                    is_unknown=False,
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
        if hasattr(self.audit_logger, 'stop'):
            self.audit_logger.stop()
        print("[BiometricEngine] Engine stopped cleanly.")


# =====================================================================
# Canonical Backward-Compatibility Aliases
# =====================================================================
BiometricVerificationEngine = BiometricTrackingEngine
BiometricVerificationEngineV5 = BiometricTrackingEngine


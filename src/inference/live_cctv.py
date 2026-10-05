import os
import sys
import math
import threading
import numpy as np
from types import ModuleType
from typing import Dict, Any, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import (
    BiometricTrackingEngine,
    TrackedSubject,
    ProcessResult,
    compute_box_iou,
    compute_box_containment
)
from src.inference.multi_camera import BufferlessVideoCapture
from src.utils.face_align import apply_clahe_contrast_lifting

# Backward-compatibility aliases
BiometricVerificationEngine = BiometricTrackingEngine
BiometricVerificationEngineV5 = BiometricTrackingEngine

# =====================================================================
# Canonical Shared Engine Singleton (Lazy Composition Root)
# =====================================================================
_DEFAULT_ENGINE: Optional[BiometricTrackingEngine] = None
_ENGINE_LOCK = threading.RLock()

# Shared legacy dictionary caches (kept for backward-compatible standalone test mutation)
track_votes: Dict[int, List[str]] = {}
track_consensus_name: Dict[int, str] = {}
frame_count: int = 0


def get_engine() -> BiometricTrackingEngine:
    """Canonical composition root: thread-safe lazy initializer for the shared engine."""
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        with _ENGINE_LOCK:
            if _DEFAULT_ENGINE is None:
                _DEFAULT_ENGINE = BiometricTrackingEngine()
    return _DEFAULT_ENGINE


def set_engine(engine_instance: Optional[BiometricTrackingEngine]) -> None:
    """Allows test fixtures or custom applications to override the shared engine."""
    global _DEFAULT_ENGINE
    with _ENGINE_LOCK:
        _DEFAULT_ENGINE = engine_instance


def reset_engine() -> None:
    """Resets the shared engine instance (primarily for isolated test teardown)."""
    global _DEFAULT_ENGINE
    with _ENGINE_LOCK:
        if _DEFAULT_ENGINE is not None:
            try:
                _DEFAULT_ENGINE.stop()
            except Exception:
                pass
        _DEFAULT_ENGINE = None


def get_audit_logger():
    """Returns the thread-safe encrypted audit logger."""
    return get_engine().encrypted_audit_logger


def set_pose_only_mode(enabled: bool) -> None:
    """Configures privacy-compliant pose-only estimation mode."""
    get_engine().set_pose_only(enabled)


def get_pose_only_mode() -> bool:
    return get_engine().is_pose_only


def reload_biometric_db() -> None:
    """Hot-reloads vector index from disk."""
    get_engine().reload_index()


def l2_normalize(vector) -> np.ndarray:
    """Calculates unit-length L2 normalized vector."""
    v = np.array(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        return v
    return v / norm


def query_vector_db(query_emb, index_obj, meta_dict, threshold=0.68) -> Tuple[str, float]:
    """Legacy vector query helper."""
    if index_obj is None or getattr(index_obj, 'ntotal', 0) == 0:
        return "Unknown", 999.0
    norm_v = l2_normalize(query_emb)
    q = np.expand_dims(norm_v, axis=0)
    distances, indices = index_obj.search(q, 1)
    dist = float(distances[0][0])
    idx = int(indices[0][0])
    if dist < threshold and idx != -1:
        entry = meta_dict.get(str(idx), meta_dict.get(idx, "Unknown"))
        if isinstance(entry, dict):
            return entry.get("name", "Unknown"), dist
        return str(entry), dist
    return "Unknown", dist


def handle_face_embedding_result(track_id: int, emb: Optional[np.ndarray], ratio: Any) -> None:
    """Delegates face embedding result directly to BiometricTrackingEngine."""
    eng = get_engine()
    eng.frame_count = frame_count
    eng._handle_embedding_result(track_id, emb, ratio)


def process_single_frame(frame: np.ndarray) -> np.ndarray:
    """Processes frame through the canonical BiometricTrackingEngine."""
    res = get_engine().process_frame(frame, annotate=True)
    return res.annotated_frame


def __getattr__(name: str) -> Any:
    """
    Dynamic lazy resolution for engine singleton, tracking state dicts, and configuration.
    Provides 100% backward-compatibility for legacy callers without paying eager model loading costs.
    """
    if name in ("engine", "biometric_engine"):
        return get_engine()
    if name == "state_lock":
        return get_engine().lock
    if name == "active_tracks":
        return get_engine().active_tracks
    if name == "track_last_seen":
        return get_engine().track_last_seen
    if name == "track_first_seen":
        return get_engine().track_first_seen
    if name == "track_keypoints":
        return get_engine().track_keypoints
    if name == "limb_cooldowns":
        return get_engine().limb_cooldowns
    if name == "identified_cooldowns":
        return get_engine().identified_cooldowns
    if name in ("encrypted_audit_logger", "audit_logger"):
        return get_engine().encrypted_audit_logger
    if name == "index":
        return get_engine().index
    if name == "metadata":
        return get_engine().metadata
    if name == "POSE_ONLY_MODE":
        return get_engine().is_pose_only
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


class _LiveCCTVModule(ModuleType):
    """Custom ModuleType subclass for live_cctv supporting legacy property setters and state overrides."""
    def __getattr__(self, name: str) -> Any:
        return __getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("engine", "biometric_engine"):
            set_engine(value)
            return
        if name == "active_tracks":
            if _DEFAULT_ENGINE is not None:
                _DEFAULT_ENGINE.active_tracks = value
            super().__setattr__(name, value)
            return
        if name == "track_last_seen":
            if _DEFAULT_ENGINE is not None:
                _DEFAULT_ENGINE.track_last_seen = value
            super().__setattr__(name, value)
            return
        if name == "track_first_seen":
            if _DEFAULT_ENGINE is not None:
                _DEFAULT_ENGINE.track_first_seen = value
            super().__setattr__(name, value)
            return
        if name == "frame_count":
            global frame_count
            frame_count = value
            if _DEFAULT_ENGINE is not None:
                _DEFAULT_ENGINE.frame_count = value
            super().__setattr__(name, value)
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LiveCCTVModule

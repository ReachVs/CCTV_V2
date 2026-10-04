import os
import sys
import math
import numpy as np
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

# =====================================================================
# Canonical Shared Engine Singleton
# =====================================================================
_DEFAULT_ENGINE = BiometricTrackingEngine()

# Primary module-level references
engine = _DEFAULT_ENGINE
biometric_engine = _DEFAULT_ENGINE
state_lock = _DEFAULT_ENGINE.lock

# Backward-compatibility aliases
BiometricVerificationEngine = BiometricTrackingEngine
BiometricVerificationEngineV5 = BiometricTrackingEngine

# Expose state dictionaries directly for legacy introspection
active_tracks = _DEFAULT_ENGINE.active_tracks
track_last_seen = _DEFAULT_ENGINE.track_last_seen
track_first_seen = _DEFAULT_ENGINE.track_first_seen
track_keypoints = _DEFAULT_ENGINE.track_keypoints
limb_cooldowns = _DEFAULT_ENGINE.limb_cooldowns
identified_cooldowns = _DEFAULT_ENGINE.identified_cooldowns
track_votes: Dict[int, List[str]] = {}
track_consensus_name: Dict[int, str] = {}

encrypted_audit_logger = _DEFAULT_ENGINE.encrypted_audit_logger


def get_audit_logger():
    """Returns the thread-safe encrypted audit logger."""
    return _DEFAULT_ENGINE.encrypted_audit_logger


def set_pose_only_mode(enabled: bool) -> None:
    """Configures privacy-compliant pose-only estimation mode."""
    _DEFAULT_ENGINE.set_pose_only(enabled)


def get_pose_only_mode() -> bool:
    return _DEFAULT_ENGINE.is_pose_only


def reload_biometric_db() -> None:
    """Hot-reloads vector index from disk."""
    _DEFAULT_ENGINE.reload_index()


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


frame_count = 0


def handle_face_embedding_result(track_id: int, emb: Optional[np.ndarray], ratio: Any) -> None:
    """Delegates face embedding result directly to BiometricTrackingEngine."""
    global frame_count
    _DEFAULT_ENGINE.frame_count = frame_count
    _DEFAULT_ENGINE._handle_embedding_result(track_id, emb, ratio)


def process_single_frame(frame: np.ndarray) -> np.ndarray:
    """Processes frame through the canonical BiometricTrackingEngine."""
    res = _DEFAULT_ENGINE.process_frame(frame, annotate=True)
    return res.annotated_frame


def __getattr__(name):
    """Dynamic resolution for index, metadata, and mode properties."""
    if name == "index":
        return _DEFAULT_ENGINE.index
    if name == "metadata":
        return _DEFAULT_ENGINE.metadata
    if name == "frame_count":
        return _DEFAULT_ENGINE.frame_count
    if name == "POSE_ONLY_MODE":
        return _DEFAULT_ENGINE.is_pose_only
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

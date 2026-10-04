"""
SubjectSession domain entity encapsulating track lifecycle, typed status state machine,
biometric confirmation consensus, crossover contention hysteresis, and pose metadata.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional, List, Dict, Any


class TrackStatus(str, Enum):
    """Explicit lifecycle states for a tracked subject."""
    SCANNING = "SCANNING"
    CONFIRMED = "CONFIRMED"
    CONTENDED = "CONTENDED"
    UNVERIFIED = "UNVERIFIED"
    UNKNOWN = "UNKNOWN"
    POSE_TARGET = "POSE_TARGET"


@dataclass
class SubjectSession:
    """
    Cohesive domain entity representing a subject tracked across video frames.
    Consolidates the 12 loose dictionary states into a single, atomic state machine.
    """
    track_id: int
    first_seen: int
    last_seen: int
    box: Tuple[int, int, int, int]
    status: TrackStatus = TrackStatus.SCANNING
    identity_name: str = ""
    confidence: float = 0.50
    best_dist: float = 999.0
    best_norm: float = 0.0
    valid_face_crops: int = 0
    failed_attempts: int = 0
    attempts: int = 0
    contention_hysteresis: int = 0
    contention_blacklist_frame: int = 0
    head_presence_valid_until: int = 0
    limb_cooldown_until: int = 0
    last_submitted_frame: int = 0
    history: List[str] = field(default_factory=list)
    keypoints: Optional[List[Dict[str, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> int:
        return max(1, (self.last_seen - self.first_seen) + 1)

    @property
    def display_name(self) -> str:
        """Dynamically computes the visual display label based on status and identity."""
        if self.status == TrackStatus.POSE_TARGET:
            return f"Pose Target {self.track_id}"
        elif self.status == TrackStatus.CONTENDED:
            return f"Contended Track {self.track_id}"
        elif self.status == TrackStatus.UNVERIFIED:
            return f"Unverified ID {self.track_id}"
        elif self.status == TrackStatus.CONFIRMED:
            return self.identity_name or f"Confirmed {self.track_id}"
        elif self.status == TrackStatus.UNKNOWN:
            return "Unknown Person"
        else:
            return f"Scanning Track {self.track_id}"

    @property
    def is_confirmed(self) -> bool:
        return self.status in (TrackStatus.CONFIRMED, TrackStatus.UNKNOWN, TrackStatus.POSE_TARGET)

    @property
    def is_scanning(self) -> bool:
        return self.status == TrackStatus.SCANNING

    @property
    def is_contended(self) -> bool:
        return self.status == TrackStatus.CONTENDED

    @property
    def is_unverified(self) -> bool:
        return self.status == TrackStatus.UNVERIFIED

    @property
    def is_unknown(self) -> bool:
        return self.status == TrackStatus.UNKNOWN

    def confirm_identity(self, name: str, distance: float, norm: float = 1.0) -> None:
        """Transitions subject to CONFIRMED identity."""
        self.status = TrackStatus.CONFIRMED
        self.identity_name = name
        self.confidence = 0.85
        self.best_dist = distance
        self.best_norm = max(self.best_norm, norm)

    def mark_unknown(self) -> None:
        """Transitions subject to confirmed UNKNOWN state."""
        self.status = TrackStatus.UNKNOWN
        self.identity_name = "Unknown Person"
        self.confidence = 0.50

    def mark_unverified(self) -> None:
        """Transitions subject to UNVERIFIED (scanning timed out with no valid faces)."""
        self.status = TrackStatus.UNVERIFIED
        self.confidence = 0.50

    def mark_pose(self, keypoints: Optional[List[Dict[str, float]]] = None) -> None:
        """Transitions subject to POSE_TARGET mode (anonymous privacy)."""
        self.status = TrackStatus.POSE_TARGET
        self.confidence = 1.0
        self.keypoints = keypoints

    def enter_contention(self, blacklist_expiration_frame: int) -> None:
        """Invalidates identity lock and places track in crossover contention."""
        self.status = TrackStatus.CONTENDED
        self.contention_hysteresis = 0
        self.contention_blacklist_frame = blacklist_expiration_frame
        self.history.clear()

    def record_separation(self) -> bool:
        """Increments separation hysteresis counter. Returns True if hysteresis >= 3 frames."""
        self.contention_hysteresis += 1
        return self.contention_hysteresis >= 3

    def reset_contention(self) -> None:
        """Exits contention and reverts to scanning state for re-verification."""
        if self.status == TrackStatus.CONTENDED:
            self.status = TrackStatus.SCANNING
        self.contention_hysteresis = 0

    def update_position(self, box: Tuple[int, int, int, int], frame_num: int, conf: float) -> None:
        """Updates spatial bounding box and last observed frame."""
        self.box = box
        self.last_seen = frame_num
        self.confidence = conf

    def to_tracked_subject(self, meta: Optional[Dict[str, Any]] = None) -> Any:
        """Projects this session state into an immutable TrackedSubject contract."""
        from src.inference.engine import TrackedSubject
        return TrackedSubject(
            track_id=self.track_id,
            name=self.display_name,
            confidence=self.confidence,
            box=self.box,
            is_confirmed=self.is_confirmed,
            is_scanning=self.is_scanning,
            is_unknown=self.is_unknown,
            is_contended=self.is_contended,
            status=self.status.value if hasattr(self.status, 'value') else str(self.status),
            keypoints=self.keypoints,
            meta=meta or self.meta
        )

    def to_cache_dict(self) -> Dict[str, Any]:
        """Backwards-compatibility dictionary matching legacy engine.track_cache shape."""
        return {
            "name": self.display_name,
            "is_confirmed": self.is_confirmed,
            "is_spatial_collision": False,
            "best_norm": self.best_norm,
            "best_dist": self.best_dist,
            "history": self.history,
            "attempts": self.attempts,
            "valid_face_crops": self.valid_face_crops,
            "failed_attempts": self.failed_attempts,
            "age": self.age,
            "box": self.box,
            "last_seen": self.last_seen
        }

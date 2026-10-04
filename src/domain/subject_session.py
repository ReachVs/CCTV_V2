"""
SubjectSession domain entity encapsulating track lifecycle, biometric confirmation,
crossover contention hysteresis, and pose metadata.
"""
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any


@dataclass
class SubjectSession:
    """
    Cohesive lifecycle representation of a subject tracked across video frames.
    Replaces loose, scattered dictionary state with typed domain logic.
    """
    track_id: int
    first_seen: int
    last_seen: int
    box: Tuple[int, int, int, int]
    name: str = ""
    is_confirmed: bool = False
    confidence: float = 0.50
    best_dist: float = 999.0
    best_norm: float = 1.0
    valid_face_crops: int = 0
    contention_hysteresis: int = 0
    is_contended: bool = False
    history: List[Any] = field(default_factory=list)
    keypoints: Optional[List[Dict[str, float]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            self.name = f"Scanning Track {self.track_id}"

    @property
    def age(self) -> int:
        return max(1, (self.last_seen - self.first_seen) + 1)

    @property
    def is_scanning(self) -> bool:
        return self.name.startswith("Scanning")

    @property
    def is_unverified(self) -> bool:
        return self.name.startswith("Unverified")

    @property
    def is_unknown(self) -> bool:
        return self.name in ["Unknown", "Ambiguous Match", "Unknown Person"]

    def confirm_identity(self, name: str, distance: float, norm: float = 1.0) -> None:
        """Confirm identity of tracked subject."""
        self.name = name
        self.is_confirmed = True
        self.confidence = 0.85
        self.best_dist = distance
        self.best_norm = norm

    def mark_unknown(self) -> None:
        """Mark subject as confirmed unknown after verification failure or small box cutoff."""
        self.name = "Unknown Person"
        self.is_confirmed = True
        self.confidence = 0.50

    def mark_unverified(self) -> None:
        """Mark subject as unverified when scanning timeout expires without a face."""
        self.name = f"Unverified ID {self.track_id}"
        self.is_confirmed = False
        self.confidence = 0.50

    def update_position(self, box: Tuple[int, int, int, int], frame_num: int, confidence: float) -> None:
        """Update box and update last seen timestamp."""
        self.box = box
        self.last_seen = frame_num
        self.confidence = confidence

    def to_cache_dict(self) -> Dict[str, Any]:
        """Backwards compatibility adapter for legacy tests asserting on engine.track_cache."""
        return {
            "name": self.name,
            "is_confirmed": self.is_confirmed,
            "best_dist": self.best_dist,
            "best_norm": self.best_norm,
            "valid_face_crops": self.valid_face_crops,
            "history": self.history,
            "box": self.box,
            "last_seen": self.last_seen
        }

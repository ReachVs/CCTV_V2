"""
Deep Module: TrackRegistry
==========================
Authoritative single source of truth for active TrackedSubject lifecycles and SubjectSession state.
Replaces fragmented, parallel tracking tables (active_tracks, track_cache, track_last_seen,
track_first_seen, track_keypoints) behind a cohesive, self-synchronized domain seam.
"""
from collections.abc import MutableMapping
from dataclasses import dataclass
import threading
from typing import Dict, List, Optional, Tuple, Any, Iterable, Iterator

from src.domain.subject_session import SubjectSession, TrackStatus


class TrackSessionCacheAdapter(MutableMapping):
    """Compatibility adapter exposing dictionary view over a SubjectSession."""

    def __init__(self, session: SubjectSession):
        self._session = session
        self._custom: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        if key == "name":
            return self._session.display_name
        elif key == "is_confirmed":
            return self._session.is_confirmed
        elif key == "failed_attempts":
            return self._session.failed_attempts
        elif key == "valid_face_crops":
            return self._session.valid_face_crops
        elif key == "attempts":
            return self._session.attempts
        elif key == "best_dist":
            return self._session.best_dist
        elif key == "best_norm":
            return self._session.best_norm
        elif key == "history":
            return self._session.history
        elif key == "box":
            return self._session.box
        return self._custom.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "name":
            if value == "Unknown Person":
                self._session.mark_unknown()
            elif value.startswith("Scanning"):
                self._session.status = TrackStatus.SCANNING
                self._session.identity_name = value
            elif value.startswith("Contended"):
                self._session.status = TrackStatus.CONTENDED
            elif value.startswith("Unverified"):
                self._session.mark_unverified()
            elif self._session.is_confirmed:
                self._session.identity_name = value
            else:
                self._session.identity_name = value
        elif key == "is_confirmed":
            if value and not self._session.is_confirmed:
                name = self._session.identity_name or f"Confirmed {self._session.track_id}"
                self._session.confirm_identity(name, self._session.best_dist, self._session.best_norm)
            elif not value and self._session.is_confirmed:
                self._session.status = TrackStatus.SCANNING
        elif key == "failed_attempts":
            self._session.failed_attempts = int(value)
        elif key == "valid_face_crops":
            self._session.valid_face_crops = int(value)
        elif key == "attempts":
            self._session.attempts = int(value)
        elif key == "best_dist":
            self._session.best_dist = float(value)
        elif key == "best_norm":
            self._session.best_norm = float(value)
        elif key == "history":
            self._session.history = list(value)
        elif key == "box":
            self._session.box = value
        else:
            self._custom[key] = value

    def __delitem__(self, key: str) -> None:
        self._custom.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        keys = {"name", "is_confirmed", "failed_attempts", "valid_face_crops", "attempts", "best_dist", "best_norm", "history", "box"}
        return iter(keys.union(self._custom.keys()))

    def __len__(self) -> int:
        return len(list(iter(self)))

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class ActiveTracksProxy(MutableMapping):
    """Thread-safe proxy presenting active_tracks dict facade backed by TrackRegistry."""

    def __init__(self, registry: "TrackRegistry"):
        self._registry = registry

    def __getitem__(self, track_id: int) -> str:
        s = self._registry.get(track_id)
        if s is None:
            raise KeyError(track_id)
        return s.display_name

    def __setitem__(self, track_id: int, name: str) -> None:
        s = self._registry.get(track_id)
        if s is None:
            s = self._registry.get_or_create(track_id, (0, 0, 0, 0), 1)
        if name == "Unknown Person":
            s.mark_unknown()
        elif name.startswith("Scanning"):
            s.status = TrackStatus.SCANNING
            s.identity_name = name
        elif name.startswith("Contended"):
            s.status = TrackStatus.CONTENDED
        elif name.startswith("Unverified"):
            s.mark_unverified()
        else:
            s.confirm_identity(name, 0.5)

    def __delitem__(self, track_id: int) -> None:
        if self._registry.purge(track_id) is None:
            raise KeyError(track_id)

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._registry._sessions.keys()))

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._registry

    def get(self, track_id: int, default: Any = None) -> Any:
        s = self._registry.get(track_id)
        return s.display_name if s is not None else default

    def setdefault(self, track_id: int, default: str = "") -> str:
        if track_id not in self:
            self[track_id] = default
        return self[track_id]


class TrackCacheProxy(MutableMapping):
    """Thread-safe proxy presenting track_cache dict facade backed by TrackRegistry."""

    def __init__(self, registry: "TrackRegistry"):
        self._registry = registry

    def __getitem__(self, track_id: int) -> TrackSessionCacheAdapter:
        s = self._registry.get(track_id)
        if s is None:
            raise KeyError(track_id)
        return TrackSessionCacheAdapter(s)

    def __setitem__(self, track_id: int, value: Any) -> None:
        s = self._registry.get(track_id)
        if s is None:
            s = self._registry.get_or_create(track_id, (0, 0, 0, 0), 1)
        adapter = TrackSessionCacheAdapter(s)
        if isinstance(value, dict):
            for k, v in value.items():
                adapter[k] = v

    def __delitem__(self, track_id: int) -> None:
        if self._registry.purge(track_id) is None:
            raise KeyError(track_id)

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._registry._sessions.keys()))

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._registry

    def get(self, track_id: int, default: Any = None) -> Any:
        s = self._registry.get(track_id)
        return TrackSessionCacheAdapter(s) if s is not None else default

    def setdefault(self, track_id: int, default: Any = None) -> TrackSessionCacheAdapter:
        if track_id not in self:
            self[track_id] = default or {}
        return self[track_id]


class TrackLastSeenProxy(MutableMapping):
    """Facade for track_last_seen."""

    def __init__(self, registry: "TrackRegistry"):
        self._registry = registry

    def __getitem__(self, track_id: int) -> int:
        s = self._registry.get(track_id)
        if s is None:
            raise KeyError(track_id)
        return s.last_seen

    def __setitem__(self, track_id: int, value: int) -> None:
        s = self._registry.get(track_id)
        if s is None:
            s = self._registry.get_or_create(track_id, (0, 0, 0, 0), value)
        s.last_seen = int(value)

    def __delitem__(self, track_id: int) -> None:
        self._registry.purge(track_id)

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._registry._sessions.keys()))

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._registry

    def get(self, track_id: int, default: Any = None) -> Any:
        s = self._registry.get(track_id)
        return s.last_seen if s is not None else default


class TrackFirstSeenProxy(MutableMapping):
    """Facade for track_first_seen."""

    def __init__(self, registry: "TrackRegistry"):
        self._registry = registry

    def __getitem__(self, track_id: int) -> int:
        s = self._registry.get(track_id)
        if s is None:
            raise KeyError(track_id)
        return s.first_seen

    def __setitem__(self, track_id: int, value: int) -> None:
        s = self._registry.get(track_id)
        if s is None:
            s = self._registry.get_or_create(track_id, (0, 0, 0, 0), value)
        s.first_seen = int(value)

    def __delitem__(self, track_id: int) -> None:
        self._registry.purge(track_id)

    def __iter__(self) -> Iterator[int]:
        return iter(list(self._registry._sessions.keys()))

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._registry

    def get(self, track_id: int, default: Any = None) -> Any:
        s = self._registry.get(track_id)
        return s.first_seen if s is not None else default


class TrackKeypointsProxy(MutableMapping):
    """Facade for track_keypoints."""

    def __init__(self, registry: "TrackRegistry"):
        self._registry = registry

    def __getitem__(self, track_id: int) -> List[Dict[str, float]]:
        s = self._registry.get(track_id)
        if s is None or s.keypoints is None:
            raise KeyError(track_id)
        return s.keypoints

    def __setitem__(self, track_id: int, value: List[Dict[str, float]]) -> None:
        s = self._registry.get(track_id)
        if s is None:
            s = self._registry.get_or_create(track_id, (0, 0, 0, 0), 1)
        s.keypoints = value

    def __delitem__(self, track_id: int) -> None:
        s = self._registry.get(track_id)
        if s is not None:
            s.keypoints = None

    def __iter__(self) -> Iterator[int]:
        return iter([tid for tid, s in self._registry._sessions.items() if s.keypoints is not None])

    def __len__(self) -> int:
        return len(list(iter(self)))

    def __contains__(self, track_id: int) -> bool:
        s = self._registry.get(track_id)
        return s is not None and s.keypoints is not None

    def get(self, track_id: int, default: Any = None) -> Any:
        s = self._registry.get(track_id)
        return s.keypoints if (s is not None and s.keypoints is not None) else default


class TrackRegistry:
    """
    Deep Domain Module: Manages active SubjectSessions across continuous video ingestion.
    Thread-safe via internal RLock. Consolidates track instantiation, spatial updates,
    keypoints association, lifecycle transitions, and stale track garbage collection.
    """

    def __init__(self):
        self._sessions: Dict[int, SubjectSession] = {}
        self._lock = threading.RLock()
        self.active_tracks = ActiveTracksProxy(self)
        self.track_cache = TrackCacheProxy(self)
        self.track_last_seen = TrackLastSeenProxy(self)
        self.track_first_seen = TrackFirstSeenProxy(self)
        self.track_keypoints = TrackKeypointsProxy(self)

    def get(self, track_id: int) -> Optional[SubjectSession]:
        """Returns the SubjectSession for track_id, or None if not found."""
        with self._lock:
            return self._sessions.get(track_id)

    def get_or_create(
        self,
        track_id: int,
        box: Tuple[int, int, int, int],
        current_frame: int
    ) -> SubjectSession:
        """Retrieves existing session or creates a new SubjectSession."""
        with self._lock:
            session = self._sessions.get(track_id)
            if session is None:
                session = SubjectSession(
                    track_id=track_id,
                    first_seen=current_frame,
                    last_seen=current_frame,
                    box=box,
                    status=TrackStatus.SCANNING,
                )
                self._sessions[track_id] = session
            else:
                session.last_seen = max(session.last_seen, current_frame)
                session.box = box
            return session

    def update_box(
        self,
        track_id: int,
        box: Tuple[int, int, int, int],
        current_frame: int
    ) -> Optional[SubjectSession]:
        """Updates bounding box and last seen frame for an active track."""
        with self._lock:
            session = self._sessions.get(track_id)
            if session is not None:
                session.box = box
                session.last_seen = max(session.last_seen, current_frame)
            return session

    def update_keypoints(
        self,
        track_id: int,
        keypoints: Optional[List[Dict[str, float]]]
    ) -> None:
        """Associates pose keypoints with a subject session."""
        with self._lock:
            session = self._sessions.get(track_id)
            if session is not None:
                session.keypoints = keypoints

    def purge(self, track_id: int) -> Optional[SubjectSession]:
        """Removes a session from the registry and returns it."""
        with self._lock:
            return self._sessions.pop(track_id, None)

    def purge_stale(
        self,
        current_frame: int,
        active_track_ids: Optional[Iterable[int]] = None,
        max_missing_frames: int = 30
    ) -> List[int]:
        """
        Garbage-collects dead or missing tracks.
        Returns the list of purged track_ids.
        """
        purged: List[int] = []
        with self._lock:
            active_set = set(active_track_ids) if active_track_ids is not None else None
            for tid, session in list(self._sessions.items()):
                is_missing = active_set is not None and tid not in active_set
                is_timed_out = (current_frame - session.last_seen) > max_missing_frames
                if is_missing and is_timed_out:
                    self._sessions.pop(tid, None)
                    purged.append(tid)
        return purged

    def active_sessions(self) -> List[SubjectSession]:
        """Returns a snapshot of all active SubjectSession instances."""
        with self._lock:
            return list(self._sessions.values())

    def clear(self) -> None:
        """Clears all sessions from the registry."""
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def __contains__(self, track_id: int) -> bool:
        with self._lock:
            return track_id in self._sessions

    def __iter__(self) -> Iterator[SubjectSession]:
        with self._lock:
            return iter(list(self._sessions.values()))

"""
Deep Module: IdentityObserver & Identity Event Dispatcher
==========================================================
Defines the authoritative domain event and observer seam for verified biometric identities.
Decouples core inference and track consensus from side-effect consumers like
Automated Attendance Recording, Encrypted Forensic Audit Logging, and Real-Time Telemetry.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import threading
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class IdentityConfirmedEvent:
    """Immutable domain event emitted when a tracked subject reaches consensus identity verification."""
    track_id: int
    person_name: str
    match_distance: float
    timestamp_str: str
    frame_num: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UnusualPersonDetectedEvent:
    """Immutable domain event emitted when an unrecognized face is tracked for >= 15 frames."""
    track_id: int
    first_seen_frame: int
    frame_num: int
    timestamp_str: str
    snapshot_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityObserver(ABC):
    """Abstract Seam: Observer interface for verified identity lifecycle events."""

    @abstractmethod
    def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
        """Invoked when a tracked subject reaches consensus verification."""
        pass

    def on_unusual_person_detected(self, event: UnusualPersonDetectedEvent) -> None:
        """Optional hook invoked when an unknown/unusual subject is tracked for >= 15 frames."""
        pass


class AttendanceIdentityObserver(IdentityObserver):
    """Adapter bridging IdentityConfirmedEvent to AttendanceService.record_check_in."""

    def __init__(self, attendance_service: Any):
        self.attendance_service = attendance_service

    def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
        if self.attendance_service is not None and hasattr(self.attendance_service, "record_check_in"):
            try:
                course_id = None
                if getattr(event, "metadata", None) and isinstance(event.metadata, dict):
                    course_id = event.metadata.get("course_id")
                self.attendance_service.record_check_in(
                    person_name=event.person_name,
                    timestamp_str=event.timestamp_str,
                    track_id=event.track_id,
                    match_distance=event.match_distance,
                    course_id=course_id
                )
            except Exception as e:
                print(f"[AttendanceIdentityObserver Warning] Check-in error for {event.person_name}: {e}")


class AuditIdentityObserver(IdentityObserver):
    """Adapter bridging IdentityConfirmedEvent to AuditLogger with frame cooldown throttling."""

    def __init__(
        self,
        audit_logger: Any,
        cooldown_frames: int = 150,
        identified_cooldowns: Optional[Dict[str, int]] = None
    ):
        self.audit_logger = audit_logger
        self.cooldown_frames = cooldown_frames
        self.identified_cooldowns: Dict[str, int] = identified_cooldowns if identified_cooldowns is not None else {}
        self._lock = threading.RLock()

    def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
        if self.audit_logger is None:
            return
        with self._lock:
            last_frame = self.identified_cooldowns.get(event.person_name, None)
            if last_frame is None or (event.frame_num - last_frame > self.cooldown_frames):
                self.identified_cooldowns[event.person_name] = event.frame_num
                try:
                    self.audit_logger.log_event(
                        timestamp=event.timestamp_str,
                        track_id=event.track_id,
                        person_name=event.person_name,
                        match_distance=event.match_distance,
                    )
                except Exception as e:
                    print(f"[AuditIdentityObserver Warning] Audit logging error for {event.person_name}: {e}")



class IdentityEventDispatcher:
    """Thread-safe event dispatcher maintaining registered IdentityObserver adapters."""

    def __init__(self, observers: Optional[List[IdentityObserver]] = None):
        self._observers: List[IdentityObserver] = list(observers) if observers else []
        self._lock = threading.RLock()

    def register(self, observer: IdentityObserver) -> None:
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)

    def unregister(self, observer: IdentityObserver) -> None:
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)

    def dispatch(self, event: IdentityConfirmedEvent) -> None:
        with self._lock:
            observers_copy = list(self._observers)
        for obs in observers_copy:
            try:
                obs.on_identity_confirmed(event)
            except Exception as e:
                print(f"[IdentityEventDispatcher Warning] Observer {obs} raised: {e}")

    def dispatch_unusual_person(self, event: UnusualPersonDetectedEvent) -> None:
        with self._lock:
            observers_copy = list(self._observers)
        for obs in observers_copy:
            try:
                obs.on_unusual_person_detected(event)
            except Exception as e:
                print(f"[IdentityEventDispatcher Warning] Observer {obs} raised in on_unusual_person_detected: {e}")

    @property
    def observers(self) -> List[IdentityObserver]:
        with self._lock:
            return list(self._observers)

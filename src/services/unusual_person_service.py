"""
Unusual Person / Intruder Detection & Alert Management Service.
==============================================================
Subscribes to UnusualPersonDetectedEvent across the IdentityObserver seam,
stores cropped face snapshot artifacts, and manages alert lifecycle states
(ACTIVE, DISMISSED, ENROLLED) for pinned UI dashboard banners.
"""
import os
import cv2
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

from src.domain.identity_observer import IdentityObserver, IdentityConfirmedEvent, UnusualPersonDetectedEvent


class UnusualPersonService(IdentityObserver):
    """
    Deep Module: Manages intruder and unusual person alerts detected by the CCTV stream.
    Provides thread-safe in-memory caching and snapshot storage for real-time dashboard notifications.
    """

    def __init__(self, storage_dir: str, url_prefix: str = "/events/unusual"):
        self.storage_dir = os.path.abspath(storage_dir)
        self.url_prefix = url_prefix.rstrip("/")
        self._lock = threading.RLock()
        self._alerts: Dict[int, Dict[str, Any]] = {}
        os.makedirs(self.storage_dir, exist_ok=True)

    def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
        """If a track is later confirmed as a known person, auto-resolve any unusual alert."""
        with self._lock:
            if event.track_id in self._alerts and self._alerts[event.track_id]["status"] == "ACTIVE":
                self._alerts[event.track_id]["status"] = "ENROLLED"
                self._alerts[event.track_id]["enrolled_name"] = event.person_name

    def on_unusual_person_detected(self, event: UnusualPersonDetectedEvent) -> None:
        """Observer callback when an unrecognized face is tracked for >= 15 frames."""
        with self._lock:
            track_id = event.track_id
            if track_id in self._alerts:
                self._alerts[track_id]["last_seen"] = event.timestamp_str
                self._alerts[track_id]["frame_num"] = event.frame_num
                return

            # Create new alert record
            snapshot_path = event.snapshot_path
            snapshot_url = ""
            if snapshot_path and os.path.exists(snapshot_path):
                filename = os.path.basename(snapshot_path)
                snapshot_url = f"{self.url_prefix}/{filename}"

            self._alerts[track_id] = {
                "track_id": track_id,
                "first_seen": event.timestamp_str,
                "last_seen": event.timestamp_str,
                "frame_num": event.frame_num,
                "snapshot_path": snapshot_path,
                "snapshot_url": snapshot_url,
                "status": "ACTIVE",
                "metadata": dict(event.metadata)
            }

    def record_alert(
        self,
        track_id: int,
        snapshot_img: Optional[np.ndarray] = None,
        timestamp_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Manually or programmatically creates/updates an unusual person alert with an image crop."""
        now_ts = timestamp_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            snapshot_path = ""
            snapshot_url = ""

            if snapshot_img is not None and isinstance(snapshot_img, np.ndarray) and snapshot_img.size > 0:
                clean_time = now_ts.replace(":", "-").replace(" ", "_")
                filename = f"unusual_track_{track_id}_{clean_time}.jpg"
                snapshot_path = os.path.join(self.storage_dir, filename)
                try:
                    cv2.imwrite(snapshot_path, snapshot_img)
                    snapshot_url = f"{self.url_prefix}/{filename}"
                except Exception as e:
                    print(f"[UnusualPersonService Warning] Failed to write crop snapshot: {e}")

            alert = {
                "track_id": track_id,
                "first_seen": now_ts,
                "last_seen": now_ts,
                "frame_num": 0,
                "snapshot_path": snapshot_path,
                "snapshot_url": snapshot_url,
                "status": "ACTIVE",
                "metadata": {}
            }
            self._alerts[track_id] = alert
            return alert

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Returns all ACTIVE alerts sorted by first_seen descending."""
        with self._lock:
            active = [a for a in self._alerts.values() if a.get("status") == "ACTIVE"]
            active.sort(key=lambda x: x.get("first_seen", ""), reverse=True)
            return list(active)

    def dismiss_alert(self, track_id: int) -> bool:
        """Dismisses an unusual person alert (e.g. recognized visitor or passerby)."""
        with self._lock:
            if track_id in self._alerts:
                self._alerts[track_id]["status"] = "DISMISSED"
                self._alerts[track_id]["dismissed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return True
            return False

    def enroll_alert(
        self,
        track_id: int,
        person_name: str,
        enrollment_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Quick-enrolls the captured unusual person snapshot directly to a person identity.
        If enrollment_manager is provided, commits the biometric profile to the database.
        """
        clean_name = person_name.strip()
        if not clean_name:
            raise ValueError("Student/person name must not be empty.")

        with self._lock:
            if track_id not in self._alerts:
                raise ValueError(f"No alert found for track_id {track_id}.")

            alert = self._alerts[track_id]
            snapshot_path = alert.get("snapshot_path", "")

            if enrollment_manager is not None and snapshot_path and os.path.exists(snapshot_path):
                try:
                    img = cv2.imread(snapshot_path)
                    if img is not None and hasattr(enrollment_manager, "enroll_subject"):
                        # Convert to JPEG bytes or enroll
                        _, buf = cv2.imencode(".jpg", img)
                        enrollment_manager.enroll_subject(name=clean_name, images=[buf.tobytes()])
                except Exception as e:
                    print(f"[UnusualPersonService Warning] Failed to enroll snapshot: {e}")

            alert["status"] = "ENROLLED"
            alert["enrolled_name"] = clean_name
            alert["enrolled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return {
                "status": "success",
                "track_id": track_id,
                "person_name": clean_name,
                "message": f"Successfully enrolled track {track_id} as '{clean_name}'."
            }

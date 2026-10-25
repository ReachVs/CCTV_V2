"""
Unit Tests for IdentityObserver Seam and Event Dispatcher
==========================================================
Verifies hermetic behavior of IdentityConfirmedEvent, IdentityEventDispatcher,
AttendanceIdentityObserver, and AuditIdentityObserver adapters.
"""
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

from src.domain.identity_observer import (
    IdentityConfirmedEvent,
    IdentityObserver,
    AttendanceIdentityObserver,
    AuditIdentityObserver,
    IdentityEventDispatcher,
)


class TestIdentityObserverSeam(unittest.TestCase):

    def test_identity_confirmed_event_immutability(self):
        """Ensures IdentityConfirmedEvent is frozen and immutable."""
        event = IdentityConfirmedEvent(
            track_id=1,
            person_name="Dr. Alice Smith",
            match_distance=0.18,
            timestamp_str="2026-10-21 23:00:00",
            frame_num=100,
            metadata={"norm": 1.0}
        )
        self.assertEqual(event.track_id, 1)
        self.assertEqual(event.person_name, "Dr. Alice Smith")
        with self.assertRaises(FrozenInstanceError):
            event.person_name = "Tampered"

    def test_event_dispatcher_register_unregister_and_dispatch(self):
        """Verifies observer lifecycle and dispatching."""
        dispatcher = IdentityEventDispatcher()
        received_events = []

        class TestObserver(IdentityObserver):
            def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
                received_events.append(event)

        observer = TestObserver()
        dispatcher.register(observer)
        self.assertIn(observer, dispatcher.observers)

        event = IdentityConfirmedEvent(
            track_id=42,
            person_name="Bob Jones",
            match_distance=0.22,
            timestamp_str="2026-10-21 23:05:00",
            frame_num=50
        )
        dispatcher.dispatch(event)
        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].person_name, "Bob Jones")

        # Unregister
        dispatcher.unregister(observer)
        self.assertNotIn(observer, dispatcher.observers)
        dispatcher.dispatch(event)
        self.assertEqual(len(received_events), 1)  # No new event received

    def test_event_dispatcher_exception_isolation(self):
        """Ensures one failing observer does not prevent other observers from executing."""
        dispatcher = IdentityEventDispatcher()
        calls = []

        class FailingObserver(IdentityObserver):
            def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
                raise RuntimeError("Explosion in observer")

        class ResilientObserver(IdentityObserver):
            def on_identity_confirmed(self, event: IdentityConfirmedEvent) -> None:
                calls.append(event.person_name)

        dispatcher.register(FailingObserver())
        dispatcher.register(ResilientObserver())

        event = IdentityConfirmedEvent(
            track_id=5,
            person_name="Ceaser",
            match_distance=0.15,
            timestamp_str="2026-10-21 23:10:00",
            frame_num=10
        )
        # Should not raise exception
        dispatcher.dispatch(event)
        self.assertEqual(calls, ["Ceaser"])

    def test_attendance_identity_observer_adapter(self):
        """Verifies AttendanceIdentityObserver bridges IdentityConfirmedEvent to record_check_in."""
        mock_attendance_service = MagicMock()
        observer = AttendanceIdentityObserver(mock_attendance_service)

        event = IdentityConfirmedEvent(
            track_id=7,
            person_name="Alice Smith",
            match_distance=0.12,
            timestamp_str="2026-10-21 08:55:00",
            frame_num=120
        )
        observer.on_identity_confirmed(event)

        mock_attendance_service.record_check_in.assert_called_once_with(
            person_name="Alice Smith",
            timestamp_str="2026-10-21 08:55:00",
            track_id=7,
            match_distance=0.12,
        )

    def test_audit_identity_observer_cooldown_throttling(self):
        """Verifies AuditIdentityObserver enforces frame cooldown before logging."""
        mock_audit_logger = MagicMock()
        cooldown_dict = {}
        observer = AuditIdentityObserver(
            audit_logger=mock_audit_logger,
            cooldown_frames=150,
            identified_cooldowns=cooldown_dict
        )

        event1 = IdentityConfirmedEvent(
            track_id=1,
            person_name="Student X",
            match_distance=0.20,
            timestamp_str="2026-10-21 10:00:00",
            frame_num=10
        )
        observer.on_identity_confirmed(event1)
        self.assertEqual(mock_audit_logger.log_event.call_count, 1)
        self.assertEqual(cooldown_dict.get("Student X"), 10)

        # Immediate next frame (should be throttled)
        event2 = IdentityConfirmedEvent(
            track_id=1,
            person_name="Student X",
            match_distance=0.19,
            timestamp_str="2026-10-21 10:00:01",
            frame_num=15
        )
        observer.on_identity_confirmed(event2)
        self.assertEqual(mock_audit_logger.log_event.call_count, 1)

        # After cooldown (frame 165 > 10 + 150)
        event3 = IdentityConfirmedEvent(
            track_id=1,
            person_name="Student X",
            match_distance=0.18,
            timestamp_str="2026-10-21 10:05:00",
            frame_num=165
        )
        observer.on_identity_confirmed(event3)
        self.assertEqual(mock_audit_logger.log_event.call_count, 2)
        self.assertEqual(cooldown_dict.get("Student X"), 165)

    def test_unusual_person_detected_event_dispatch(self):
        """Verifies UnusualPersonDetectedEvent immutability and dispatch_unusual_person routing."""
        from src.domain.identity_observer import UnusualPersonDetectedEvent

        event = UnusualPersonDetectedEvent(
            track_id=88,
            first_seen_frame=10,
            frame_num=25,
            timestamp_str="2026-10-22 10:30:00",
            snapshot_path="/data/unusual/crop_88.jpg",
            metadata={"valid_face_crops": 15}
        )
        self.assertEqual(event.track_id, 88)
        self.assertEqual(event.snapshot_path, "/data/unusual/crop_88.jpg")
        with self.assertRaises(FrozenInstanceError):
            event.track_id = 99

        dispatcher = IdentityEventDispatcher()
        received_unusual = []

        class UnusualObserver(IdentityObserver):
            def on_identity_confirmed(self, e: IdentityConfirmedEvent) -> None:
                pass

            def on_unusual_person_detected(self, e: UnusualPersonDetectedEvent) -> None:
                received_unusual.append(e)

        obs = UnusualObserver()
        dispatcher.register(obs)
        dispatcher.dispatch_unusual_person(event)

        self.assertEqual(len(received_unusual), 1)
        self.assertEqual(received_unusual[0].track_id, 88)
        self.assertEqual(received_unusual[0].snapshot_path, "/data/unusual/crop_88.jpg")


if __name__ == "__main__":
    unittest.main()


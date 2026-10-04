"""
Unit tests for SubjectSession domain entity & TrackStatus lifecycle state machine (Candidate 1).
Verifies:
1. SubjectSession lifecycle transitions:
   - SCANNING -> CONFIRMED via confirm_identity()
   - SCANNING -> UNVERIFIED via mark_unverified()
   - SCANNING -> UNKNOWN via mark_unknown()
   - SCANNING -> POSE_TARGET via mark_pose()
   - CONFIRMED -> CONTENDED via enter_contention()
   - CONTENDED -> separation hysteresis (record_separation) -> reset_contention() -> SCANNING
2. Property invariants:
   - display_name calculation across all states
   - is_confirmed, is_scanning, is_unknown, is_contended flags
3. Projection to immutable TrackedSubject contract via to_tracked_subject()
4. Backward-compatible dictionary export via to_cache_dict()
5. Engine integration:
   - engine.get_session(track_id) returns active SubjectSession
   - engine.get_all_sessions() returns all active sessions
"""
import unittest
from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.engine import BiometricTrackingEngine, TrackedSubject
from src.inference.person_detector import ScriptedPersonDetector, DetectedBox
from src.inference.face_embedder import ScriptedFaceEmbedder
import numpy as np


class TestSubjectSessionLifecycle(unittest.TestCase):

    def setUp(self):
        self.session = SubjectSession(
            track_id=42,
            first_seen=100,
            last_seen=105,
            box=(50, 60, 150, 200)
        )

    def test_initial_state(self):
        """Verify default initialization state."""
        self.assertEqual(self.session.track_id, 42)
        self.assertEqual(self.session.status, TrackStatus.SCANNING)
        self.assertTrue(self.session.is_scanning)
        self.assertFalse(self.session.is_confirmed)
        self.assertFalse(self.session.is_contended)
        self.assertFalse(self.session.is_unknown)
        self.assertFalse(self.session.is_unverified)
        self.assertEqual(self.session.display_name, "Scanning Track 42")
        self.assertEqual(self.session.age, 6)

    def test_confirm_identity_transition(self):
        """Verify transition to CONFIRMED identity."""
        self.session.confirm_identity("Alice", distance=0.42, norm=0.95)
        self.assertEqual(self.session.status, TrackStatus.CONFIRMED)
        self.assertTrue(self.session.is_confirmed)
        self.assertFalse(self.session.is_scanning)
        self.assertEqual(self.session.identity_name, "Alice")
        self.assertEqual(self.session.display_name, "Alice")
        self.assertEqual(self.session.best_dist, 0.42)
        self.assertEqual(self.session.confidence, 0.85)

    def test_mark_unknown_transition(self):
        """Verify transition to confirmed UNKNOWN state."""
        self.session.mark_unknown()
        self.assertEqual(self.session.status, TrackStatus.UNKNOWN)
        self.assertTrue(self.session.is_confirmed)
        self.assertTrue(self.session.is_unknown)
        self.assertFalse(self.session.is_scanning)
        self.assertEqual(self.session.display_name, "Unknown Person")

    def test_mark_unverified_transition(self):
        """Verify transition to UNVERIFIED state."""
        self.session.mark_unverified()
        self.assertEqual(self.session.status, TrackStatus.UNVERIFIED)
        self.assertFalse(self.session.is_confirmed)
        self.assertTrue(self.session.is_unverified)
        self.assertFalse(self.session.is_scanning)
        self.assertEqual(self.session.display_name, "Unverified ID 42")

    def test_mark_pose_transition(self):
        """Verify transition to anonymous POSE_TARGET state."""
        dummy_kpts = [{"x": 10.0, "y": 20.0, "confidence": 0.9}]
        self.session.mark_pose(dummy_kpts)
        self.assertEqual(self.session.status, TrackStatus.POSE_TARGET)
        self.assertTrue(self.session.is_confirmed)
        self.assertEqual(self.session.display_name, "Pose Target 42")
        self.assertEqual(self.session.keypoints, dummy_kpts)

    def test_contention_and_hysteresis_resolution(self):
        """Verify crossover contention entry, separation hysteresis, and reset."""
        self.session.confirm_identity("Alice", distance=0.40)
        self.assertTrue(self.session.is_confirmed)

        # Enter contention
        self.session.enter_contention(blacklist_expiration_frame=150)
        self.assertEqual(self.session.status, TrackStatus.CONTENDED)
        self.assertTrue(self.session.is_contended)
        self.assertFalse(self.session.is_confirmed)
        self.assertEqual(self.session.display_name, "Contended Track 42")
        self.assertEqual(self.session.contention_blacklist_frame, 150)
        self.assertEqual(self.session.history, [])

        # Frame 1 separation
        sep1 = self.session.record_separation()
        self.assertFalse(sep1)
        self.assertEqual(self.session.contention_hysteresis, 1)

        # Frame 2 separation
        sep2 = self.session.record_separation()
        self.assertFalse(sep2)
        self.assertEqual(self.session.contention_hysteresis, 2)

        # Frame 3 separation (hysteresis satisfied)
        sep3 = self.session.record_separation()
        self.assertTrue(sep3)
        self.assertEqual(self.session.contention_hysteresis, 3)

        # Reset contention
        self.session.reset_contention()
        self.assertEqual(self.session.status, TrackStatus.SCANNING)
        self.assertFalse(self.session.is_contended)
        self.assertTrue(self.session.is_scanning)

    def test_to_tracked_subject_projection(self):
        """Verify projection to immutable TrackedSubject contract."""
        self.session.confirm_identity("Alice", distance=0.35)
        meta = {"department": "Security", "clearance": "LEVEL 4"}
        subj = self.session.to_tracked_subject(meta=meta)

        self.assertIsInstance(subj, TrackedSubject)
        self.assertEqual(subj.track_id, 42)
        self.assertEqual(subj.name, "Alice")
        self.assertEqual(subj.box, (50, 60, 150, 200))
        self.assertTrue(subj.is_confirmed)
        self.assertFalse(subj.is_scanning)
        self.assertFalse(subj.is_unknown)
        self.assertEqual(subj.status, "CONFIRMED")
        self.assertEqual(subj.meta["department"], "Security")

    def test_to_cache_dict_compatibility(self):
        """Verify backward-compatible cache dictionary generation."""
        self.session.confirm_identity("Bob", distance=0.45, norm=1.2)
        self.session.valid_face_crops = 4
        self.session.attempts = 5
        self.session.history = ["Bob", "Bob"]

        cache_dict = self.session.to_cache_dict()
        self.assertEqual(cache_dict["name"], "Bob")
        self.assertTrue(cache_dict["is_confirmed"])
        self.assertEqual(cache_dict["valid_face_crops"], 4)
        self.assertEqual(cache_dict["attempts"], 5)
        self.assertEqual(cache_dict["history"], ["Bob", "Bob"])
        self.assertEqual(cache_dict["age"], 6)


class TestEngineSessionIntegration(unittest.TestCase):

    def test_engine_session_lifecycle_and_lookup(self):
        """Verify BiometricTrackingEngine maintains and exposes SubjectSession instances."""
        detector = ScriptedPersonDetector([
            DetectedBox(box=[50.0, 50.0, 150.0, 250.0], conf=0.95, track_id=10)
        ])
        embedder = ScriptedFaceEmbedder(vector=np.ones(512, dtype=np.float32))

        engine = BiometricTrackingEngine(
            detector=detector,
            embedder=embedder
        )
        engine.head_presence_cache[10] = 100

        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = engine.process_frame(dummy_frame)

        # Verify session was created and is queryable
        session = engine.get_session(10)
        self.assertIsNotNone(session)
        self.assertEqual(session.track_id, 10)
        self.assertEqual(session.box, (100, 100, 300, 500))

        # Verify all sessions snapshot
        sessions = engine.get_all_sessions()
        self.assertIn(10, sessions)
        self.assertEqual(sessions[10].track_id, 10)

        # Purge and verify clean eviction
        engine.purge_track(10)
        self.assertIsNone(engine.get_session(10))
        self.assertNotIn(10, engine.get_all_sessions())


if __name__ == "__main__":
    unittest.main()

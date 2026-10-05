"""
Unit test for Crossover Contention recovery:
When a track enters crossover contention and then separates,
it must reset to SCANNING and confirm as the real identity (e.g. Ceaser),
NEVER locking in 'Contended Track X' as a confirmed identity.
"""
import unittest
from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.engine import BiometricTrackingEngine


class TestCrossoverContentionRecovery(unittest.TestCase):
    def test_contended_track_resets_cleanly_without_locking_contended_as_identity(self):
        engine = BiometricTrackingEngine()
        track_id = 10

        # Simulate track entering contention
        engine.contended_tracks[track_id] = 0
        engine.active_tracks[track_id] = f"Contended Track {track_id}"
        session = engine._get_or_create_session(track_id, (100, 100, 300, 400), 1)
        session.enter_contention(blacklist_expiration_frame=45)

        self.assertEqual(session.status, TrackStatus.CONTENDED)
        self.assertEqual(session.display_name, "Contended Track 10")

        # Now simulate separation: contention clears
        del engine.contended_tracks[track_id]
        session.reset_contention()
        engine.active_tracks[track_id] = f"Scanning Track {track_id}"

        self.assertEqual(session.status, TrackStatus.SCANNING)
        self.assertNotEqual(session.status, TrackStatus.CONFIRMED)
        self.assertNotIn("Contended", session.display_name)

        # Now Ceaser face arrives
        display_name, tel = engine.process_track_frame(track_id, None, track_age=3)
        self.assertNotIn("Contended", display_name)


if __name__ == "__main__":
    unittest.main()

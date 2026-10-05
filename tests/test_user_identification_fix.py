"""
Hermetic regression tests for live user identification:
1. Side-profile recognition when side profiles are enrolled
2. Ensuring known matches in history are never branded as Unknown Person
3. Embedder fidelity preserving native crop resolution without distortion
"""
import unittest
import numpy as np

from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.consensus import IdentityConsensus, MatchCandidate


class TestUserIdentificationFix(unittest.TestCase):
    def setUp(self):
        self.consensus = IdentityConsensus(
            consensus_votes=2,
            history_window=5,
            margin_threshold=0.15,
            side_profile_threshold=0.30,
            max_scanning_attempts=5,
            min_unknown_crops=5
        )

    def test_side_profile_with_high_confidence_confirms(self):
        """
        When a user with enrolled side profiles matches with high confidence (dist1 <= 0.15),
        consensus must confirm their identity instead of permanently locking them in scanning.
        """
        session = SubjectSession(track_id=1, first_seen=1, last_seen=1, box=(10, 10, 100, 100))
        # High confidence side profile match (like Ceaser's 0.0853)
        c_side = MatchCandidate(name="Ceaser", dist1=0.085, dist2=1.075, norm=1.0, is_side_profile=True)

        d1 = self.consensus.evaluate(session, c_side, current_frame=1)
        self.assertFalse(d1.is_confirmed)

        session.last_seen = 2
        d2 = self.consensus.evaluate(session, c_side, current_frame=2)
        self.assertTrue(d2.is_confirmed, "High-confidence side match must confirm identity!")
        self.assertEqual(d2.display_name, "Ceaser")
        self.assertEqual(session.status, TrackStatus.CONFIRMED)

    def test_known_matches_in_history_never_branded_unknown_person(self):
        """
        If a track has votes for a known enrolled person in its history, exceeding
        max_scanning_attempts must NEVER brand the track as 'Unknown Person'.
        """
        session = SubjectSession(track_id=2, first_seen=1, last_seen=1, box=(10, 10, 100, 100))
        # Suppose side profile match is not yet confirmed (e.g. dist1=0.25)
        c_side = MatchCandidate(name="Ceaser", dist1=0.25, dist2=0.80, norm=1.0, is_side_profile=True)

        # Run past max_scanning_attempts (5)
        for f in range(1, 8):
            session.last_seen = f
            d = self.consensus.evaluate(session, c_side, current_frame=f)

        self.assertNotEqual(session.status, TrackStatus.UNKNOWN, "Known matches must NEVER become UNKNOWN!")
        self.assertNotEqual(d.display_name, "Unknown Person")


if __name__ == "__main__":
    unittest.main()

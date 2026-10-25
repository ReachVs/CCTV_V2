"""
Hermetic unit tests for IdentityConsensus domain module.
Tests multi-frame voting, side-profile safety, claim leases, and timeout transitions.
"""
import unittest
from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.consensus import IdentityConsensus, MatchCandidate, ConsensusDecision


class TestIdentityConsensus(unittest.TestCase):
    def setUp(self):
        self.consensus = IdentityConsensus(
            consensus_votes=2,
            history_window=5,
            margin_threshold=0.15,
            side_profile_threshold=0.30,
            max_scanning_attempts=5
        )

    def test_multi_frame_voting_lock_in(self):
        session = SubjectSession(track_id=1, first_seen=1, last_seen=3, box=(10, 10, 50, 50))
        # Frame 1 match
        c1 = MatchCandidate(name="Ceaser", dist1=0.20, dist2=0.85, norm=1.2, is_side_profile=False)
        d1 = self.consensus.evaluate(session, c1, current_frame=3)
        self.assertFalse(d1.is_confirmed)
        self.assertEqual(session.status, TrackStatus.SCANNING)

        # Frame 2 match
        session.last_seen = 4
        c2 = MatchCandidate(name="Ceaser", dist1=0.22, dist2=0.80, norm=1.1, is_side_profile=False)
        d2 = self.consensus.evaluate(session, c2, current_frame=4)
        self.assertTrue(d2.is_confirmed)
        self.assertEqual(session.status, TrackStatus.CONFIRMED)
        self.assertEqual(d2.display_name, "Ceaser")
        self.assertEqual(self.consensus.active_identity_claims["Ceaser"], 1)

    def test_side_profile_consensus_guardrail(self):
        """Side-profile matches alone must NOT confirm identity without frontal confirmation."""
        session = SubjectSession(track_id=2, first_seen=1, last_seen=3, box=(20, 20, 60, 60))

        # 2 consecutive side-profile matches
        c_side = MatchCandidate(name="Ceaser", dist1=0.25, dist2=0.70, norm=1.0, is_side_profile=True)
        d1 = self.consensus.evaluate(session, c_side, current_frame=3)
        self.assertFalse(d1.is_confirmed)

        session.last_seen = 4
        d2 = self.consensus.evaluate(session, c_side, current_frame=4)
        # Even with 2 votes, side profile guardrail must refuse confirmation!
        self.assertFalse(d2.is_confirmed)
        self.assertEqual(session.status, TrackStatus.SCANNING)

        # Now supply a frontal match
        session.last_seen = 5
        c_front = MatchCandidate(name="Ceaser", dist1=0.20, dist2=0.80, norm=1.2, is_side_profile=False)
        d3 = self.consensus.evaluate(session, c_front, current_frame=5)
        # Frontal confirmation unlocked consensus!
        self.assertTrue(d3.is_confirmed)
        self.assertEqual(session.status, TrackStatus.CONFIRMED)
        self.assertEqual(d3.display_name, "Ceaser")

    def test_identity_claim_collision_arbitration(self):
        """Concurrent live tracks cannot both claim the same identity."""
        session1 = SubjectSession(track_id=10, first_seen=1, last_seen=10, box=(10, 10, 50, 50))
        session2 = SubjectSession(track_id=20, first_seen=1, last_seen=10, box=(100, 100, 150, 150))

        # Confirm session1 as Alice
        c = MatchCandidate(name="Alice", dist1=0.15, dist2=0.90, norm=1.2, is_side_profile=False)
        self.consensus.evaluate(session1, c, current_frame=10)
        self.consensus.evaluate(session1, c, current_frame=11)
        self.assertTrue(session1.is_confirmed)

        # Now session2 sees Alice while session1 is live
        last_seen_lookup = lambda tid: 11 if tid == 10 else 10
        d_conflict = self.consensus.evaluate(
            session2, c, current_frame=12,
            active_track_ids=[10, 20],
            last_seen_lookup=last_seen_lookup
        )
        # Must prevent claim usurpation
        self.assertFalse(d_conflict.is_confirmed)
        self.assertNotEqual(session2.identity_name, "Alice")

    def test_claim_transfer_when_previous_claimant_dead(self):
        """When previous claimant is dead (>15 frames unseen), claim is transferred."""
        session1 = SubjectSession(track_id=10, first_seen=1, last_seen=5, box=(10, 10, 50, 50))
        c = MatchCandidate(name="Bob", dist1=0.15, dist2=0.90, norm=1.2, is_side_profile=False)
        self.consensus.evaluate(session1, c, current_frame=5)
        self.consensus.evaluate(session1, c, current_frame=6)
        self.assertEqual(self.consensus.active_identity_claims["Bob"], 10)

        # Session 2 attempts match at frame 30 (session1 died at frame 5, >15 frames ago)
        session2 = SubjectSession(track_id=20, first_seen=25, last_seen=30, box=(100, 100, 150, 150))
        last_seen_lookup = lambda tid: 5 if tid == 10 else 30
        self.consensus.evaluate(session2, c, current_frame=30, active_track_ids=[20], last_seen_lookup=last_seen_lookup)
        self.consensus.evaluate(session2, c, current_frame=31, active_track_ids=[20], last_seen_lookup=last_seen_lookup)
        self.assertTrue(session2.is_confirmed)
        self.assertEqual(session2.identity_name, "Bob")
        self.assertEqual(self.consensus.active_identity_claims["Bob"], 20)

    def test_max_attempts_transition_to_unknown(self):
        session = SubjectSession(track_id=30, first_seen=1, last_seen=3, box=(0, 0, 50, 50))
        c_unknown = MatchCandidate(name="Unknown", dist1=0.99, dist2=1.05, norm=0.8, is_side_profile=False)
        for f in range(3, 9):
            session.last_seen = f
            d = self.consensus.evaluate(session, c_unknown, current_frame=f)

        self.assertTrue(session.is_unknown)
        self.assertEqual(session.status, TrackStatus.UNKNOWN)
        self.assertEqual(session.display_name, "Unknown Person")

    def test_unknown_person_recovers_when_known_face_arrives(self):
        """A track locked in UNKNOWN must reopen and confirm when recognized face candidate arrives."""
        session = SubjectSession(track_id=40, first_seen=1, last_seen=5, box=(0, 0, 50, 50))
        session.mark_unknown()
        self.assertTrue(session.is_unknown)

        c_ceaser = MatchCandidate(name="Ceaser", dist1=0.20, dist2=0.85, norm=1.2, is_side_profile=False)
        # Frame 6: known face arrives, consensus unlocks from UNKNOWN back to SCANNING
        d1 = self.consensus.evaluate(session, c_ceaser, current_frame=6)
        self.assertEqual(session.status, TrackStatus.SCANNING)
        self.assertFalse(d1.is_confirmed)

        # Frame 7: second vote confirms identity
        d2 = self.consensus.evaluate(session, c_ceaser, current_frame=7)
        self.assertTrue(d2.is_confirmed)
        self.assertEqual(session.status, TrackStatus.CONFIRMED)
        self.assertEqual(d2.display_name, "Ceaser")

    def test_k4_consecutive_consensus_and_verifying_display(self):
        """Verifies K=4 consensus accumulation display: Verifying: Name (x/4) -> Confirmed."""
        k4_consensus = IdentityConsensus(
            consensus_votes=4,
            history_window=6,
            margin_threshold=0.15,
            side_profile_threshold=0.30,
            max_scanning_attempts=15
        )
        session = SubjectSession(track_id=55, first_seen=1, last_seen=3, box=(0, 0, 50, 50))
        cand = MatchCandidate(name="John Doe", dist1=0.20, dist2=0.85, norm=1.2, is_side_profile=False)

        # Vote 1
        d1 = k4_consensus.evaluate(session, cand, current_frame=3)
        self.assertFalse(d1.is_confirmed)
        self.assertEqual(d1.display_name, "Verifying: John Doe (1/4)")

        # Vote 2
        session.last_seen = 4
        d2 = k4_consensus.evaluate(session, cand, current_frame=4)
        self.assertFalse(d2.is_confirmed)
        self.assertEqual(d2.display_name, "Verifying: John Doe (2/4)")

        # Vote 3
        session.last_seen = 5
        d3 = k4_consensus.evaluate(session, cand, current_frame=5)
        self.assertFalse(d3.is_confirmed)
        self.assertEqual(d3.display_name, "Verifying: John Doe (3/4)")

        # Vote 4 -> Reaches K=4 consensus!
        session.last_seen = 6
        d4 = k4_consensus.evaluate(session, cand, current_frame=6)
        self.assertTrue(d4.is_confirmed)
        self.assertEqual(session.status, TrackStatus.CONFIRMED)
        self.assertEqual(d4.display_name, "John Doe")


if __name__ == "__main__":
    unittest.main()


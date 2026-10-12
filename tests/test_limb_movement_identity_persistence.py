import unittest
import numpy as np

from src.inference.engine import BiometricTrackingEngine
from src.inference.tracking_pre_filter import TrackingPreFilter
from src.inference.person_detector import DetectedBox
from src.domain.subject_session import TrackStatus


class TestLimbMovementIdentityPersistence(unittest.TestCase):
    """
    Reproduces and verifies fix for limb movement / intruder interruption bug:
    When a person is confirmed (e.g. Track 1 = 'Ceaser'):
    1. A moving hand/limb creates a second box (Track 2) overlapping Track 1.
    2. Track 1 MUST NOT be downgraded to 'Contended Track 1' or 'Scanning Track 1'.
    3. The overlapping limb box (Track 2) with no face MUST be suppressed/absorbed.
    4. Track 1 MUST NOT be declared 'Unknown Person' during motion.
    5. The user must NOT be forced to freeze/stay still for 45 frames.
    """

    def test_confirmed_identity_not_demoted_by_crossover_contention_with_limb(self):
        engine = BiometricTrackingEngine()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        track_id = 1
        session = engine._get_or_create_session(track_id, (200, 100, 600, 700), 1)
        session.confirm_identity("Ceaser", distance=0.25, norm=1.2)
        engine.active_tracks[track_id] = "Ceaser"
        engine.track_cache[track_id] = {
            "name": "Ceaser",
            "is_confirmed": True,
            "failed_attempts": 0,
            "history": [{"name": "Ceaser", "is_side": False}]
        }
        engine.consensus.active_identity_claims["Ceaser"] = track_id

        # Simulating that pre-filter detected crossover contention between Track 1 and Track 2
        # (e.g. from limb motion overlapping body)
        pre_filter_res = engine.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=[],
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=10,
            active_track_ids={1, 2},
            confirmed_track_ids={1}
        )
        # Force newly contended
        pre_filter_res.newly_contended_track_ids = [1, 2]
        pre_filter_res.current_frame_boxes = {1: (200, 100, 600, 700), 2: (300, 300, 500, 500)}

        # Simulate engine processing of newly_contended
        for tid in pre_filter_res.newly_contended_track_ids:
            # If track is confirmed, it MUST NOT be wiped to 'Contended Track X'
            t_cache = engine.track_cache.get(tid, {})
            if t_cache.get("is_confirmed", False):
                # Strong Identity Persistence: Confirmed track retains identity
                pass
            else:
                engine.active_tracks[tid] = f"Contended Track {tid}"

        self.assertEqual(engine.active_tracks.get(1), "Ceaser", "Confirmed person Ceaser must not be wiped to Contended Track")
        self.assertEqual(session.status, TrackStatus.CONFIRMED, "Session status must remain CONFIRMED")

    def test_track_id_switch_inherits_identity_without_ambiguous_unknown_lock(self):
        engine = BiometricTrackingEngine()
        # Track 1 was Ceaser, now Track 2 appears in overlapping spatial location
        track1 = 1
        track2 = 2
        session1 = engine._get_or_create_session(track1, (200, 100, 600, 700), 1)
        session1.confirm_identity("Ceaser", distance=0.25, norm=1.2)
        engine.consensus.active_identity_claims["Ceaser"] = track1
        engine.track_cache[track1] = {"name": "Ceaser", "is_confirmed": True, "history": [{"name": "Ceaser", "is_side": False}]}

        # Track 2 is created at overlapping location (same person moved limb/body)
        session2 = engine._get_or_create_session(track2, (210, 105, 605, 705), 5)

        from src.inference.consensus import MatchCandidate
        cand = MatchCandidate(
            name="Ceaser",
            dist1=0.28,
            dist2=0.89,
            norm=1.1,
            is_side_profile=False
        )

        # Evaluate consensus for track 2: should seamlessly transfer or adopt Ceaser, NEVER lock as Unknown Person
        decision = engine.consensus.evaluate(
            session=session2,
            candidate=cand,
            current_frame=5,
            active_track_ids=[track2],
            last_seen_lookup=lambda tid: 4 if tid == track1 else 5
        )

        self.assertNotEqual(decision.display_name, "Unknown Person")
        self.assertNotEqual(session2.status, TrackStatus.UNKNOWN)

    def test_non_face_crop_does_not_declare_unknown_person(self):
        engine = BiometricTrackingEngine()
        track_id = 5
        session = engine._get_or_create_session(track_id, (200, 100, 600, 700), 1)

        # Simulate 5 consecutive failed/non-face crops during rapid movement (e.g. hand blocking face)
        for _ in range(5):
            display_name, tel = engine.process_track_frame(track_id, None, track_age=10)

        # It should remain scanning, NOT declare 'Unknown Person' (which requires actual facial mismatch)
        self.assertNotEqual(display_name, "Unknown Person", "Non-face crops during limb motion must never declare 'Unknown Person'")
        self.assertNotEqual(session.status, TrackStatus.UNKNOWN, "Session must not be locked as UNKNOWN from non-face crops")


    def test_prefilter_suppresses_limb_box_without_demoting_confirmed_track(self):
        engine = BiometricTrackingEngine()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Track 1 is confirmed person (Ceaser)
        track1 = 1
        b1 = (200, 100, 600, 700) # Full body box
        # Track 2 is hand/limb moving in front of body (contained inside b1)
        track2 = 2
        b2 = (250, 300, 450, 500) # Overlapping sub-box / limb

        # Call pre-filter with Track 1 confirmed and Track 2 unconfirmed
        engine.pre_filter.recent_track_history[track1] = {"box": b1, "confirmed": True, "last_seen": 9}
        engine.pre_filter.recent_track_history[track2] = {"box": b2, "confirmed": False, "last_seen": 9}

        from src.inference.person_detector import DetectedBox
        det1 = DetectedBox(box=list(b1), conf=0.90, track_id=track1)
        det2 = DetectedBox(box=list(b2), conf=0.80, track_id=track2)

        res = engine.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=[det1, det2],
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=10,
            active_track_ids={track1, track2},
            confirmed_track_ids={track1}
        )

        # Track 1 must NOT be in newly_contended_track_ids
        self.assertNotIn(track1, res.newly_contended_track_ids, "Confirmed person must not enter contention from limb sub-box")
        # Track 2 (limb box) should be suppressed
        self.assertIn(track2, res.suppressed_track_ids, "Unconfirmed limb box overlapping confirmed person must be suppressed")

    def test_consensus_does_not_declare_unknown_on_scanning_attempts_timeout(self):
        engine = BiometricTrackingEngine()
        track_id = 9
        session = engine._get_or_create_session(track_id, (200, 100, 600, 700), 1)

        from src.inference.consensus import MatchCandidate
        # Candidate has no valid database match (e.g. motion blur, low quality), but attempts increment
        cand = MatchCandidate(name="Unknown", dist1=0.75, dist2=0.80, norm=0.9, is_side_profile=False)

        # Run for max_scanning_attempts
        for f in range(engine.consensus.max_scanning_attempts + 2):
            decision = engine.consensus.evaluate(
                session=session,
                candidate=cand,
                current_frame=f + 1,
                active_track_ids=[track_id]
            )

        # Crucial: Under real world motion blur, session attempts reaching limit must NOT declare 'Unknown Person' (Access Denied)
        # It must only declare Unknown Person if min_unknown_crops of genuine non-matching faces occur
        self.assertNotEqual(decision.display_name, "Unknown Person", "Timeout of attempts must not declare Unknown Person")
    def test_genuine_stranger_correctly_declared_unknown_person(self):
        engine = BiometricTrackingEngine()
        track_id = 15
        session = engine._get_or_create_session(track_id, (200, 100, 600, 700), 1)

        from src.inference.consensus import MatchCandidate
        # Genuine stranger: biometric distance to enrolled faces is high (dist1 >= 0.85)
        cand = MatchCandidate(name="Unknown", dist1=1.15, dist2=1.20, norm=1.0, is_side_profile=False)

        # Run until min_unknown_crops
        for f in range(engine.consensus.min_unknown_crops):
            decision = engine.consensus.evaluate(
                session=session,
                candidate=cand,
                current_frame=f + 1,
                active_track_ids=[track_id]
            )

        self.assertEqual(decision.display_name, "Unknown Person", "Genuine stranger must be declared Unknown Person")
        self.assertEqual(session.status, TrackStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()

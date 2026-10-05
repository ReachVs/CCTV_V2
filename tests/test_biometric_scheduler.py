import unittest
from typing import Tuple, Optional

from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.biometric_scheduler import BiometricIngestionScheduler


class DummyJitterEngine:
    def __init__(self, erratic_tracks=None):
        self.erratic_tracks = erratic_tracks or set()

    def is_erratic_movement(self, track_id: int, threshold: float = 80.0) -> bool:
        return track_id in self.erratic_tracks


class DummyTemporalSubsampler:
    def __init__(self, should_process: bool = True):
        self._should_process = should_process

    def should_process_biometrics(self, track_id: int) -> bool:
        return self._should_process


class TestBiometricIngestionScheduler(unittest.TestCase):
    def setUp(self):
        self.jitter_engine = DummyJitterEngine()
        self.temporal_subsampler = DummyTemporalSubsampler(should_process=True)
        self.scheduler = BiometricIngestionScheduler(
            jitter_engine=self.jitter_engine,
            temporal_subsampler=self.temporal_subsampler,
        )

    def test_unconfirmed_track_cadence_and_age(self):
        session = SubjectSession(track_id=1, first_seen=1, last_seen=1, box=(100, 100, 250, 400))
        # Frame 1: track_age = 1 -> should NOT extract yet (requires track_age >= 2)
        should_extract = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=1,
            is_contended=False,
        )
        self.assertFalse(should_extract)

        # Frame 2: track_age = 2 -> should extract
        session.last_seen = 2
        should_extract_f2 = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=2,
            is_contended=False,
        )
        self.assertTrue(should_extract_f2)

        # Record submission at frame 2
        self.scheduler.record_submission(track_id=1, frame_num=2)

        # Frame 3: unconfirmed interval is 2 -> frame 3 - 2 = 1 < 2 -> should NOT extract
        should_extract_f3 = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=3,
            is_contended=False,
        )
        self.assertFalse(should_extract_f3)

        # Frame 4: interval met (4 - 2 >= 2) -> should extract
        should_extract_f4 = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=4,
            is_contended=False,
        )
        self.assertTrue(should_extract_f4)

    def test_confirmed_track_cadence(self):
        session = SubjectSession(track_id=2, first_seen=1, last_seen=10, box=(100, 100, 250, 400))
        session.confirm_identity("Alice", distance=0.3, norm=1.0)
        self.scheduler.record_submission(track_id=2, frame_num=10)

        # Frame 20: 20 - 10 = 10 < 15 frames interval for confirmed -> False
        should_extract_f20 = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=20,
            is_contended=False,
        )
        self.assertFalse(should_extract_f20)

        # Frame 25: 25 - 10 = 15 -> True
        should_extract_f25 = self.scheduler.should_extract(
            session=session,
            crop_shape=(100, 100),
            current_frame=25,
            is_contended=False,
        )
        self.assertTrue(should_extract_f25)

    def test_gating_on_erratic_movement_and_cooldown(self):
        session = SubjectSession(track_id=3, first_seen=1, last_seen=5, box=(100, 100, 250, 400))
        # Make track erratic
        self.jitter_engine.erratic_tracks.add(3)
        self.assertFalse(
            self.scheduler.should_extract(session, crop_shape=(100, 100), current_frame=5, is_contended=False)
        )

        # Clear erratic, put in cooldown
        self.jitter_engine.erratic_tracks.clear()
        self.scheduler.set_cooldown(track_id=3, until_frame=10)
        self.assertFalse(
            self.scheduler.should_extract(session, crop_shape=(100, 100), current_frame=8, is_contended=False)
        )
        # Passed cooldown
        self.assertTrue(
            self.scheduler.should_extract(session, crop_shape=(100, 100), current_frame=11, is_contended=False)
        )

    def test_gating_on_crop_dimensions(self):
        session = SubjectSession(track_id=4, first_seen=1, last_seen=5, box=(100, 100, 250, 400))
        # Crop too small (<80x80)
        self.assertFalse(
            self.scheduler.should_extract(session, crop_shape=(60, 60), current_frame=5, is_contended=False)
        )
        # Crop None
        self.assertFalse(
            self.scheduler.should_extract(session, crop_shape=None, current_frame=5, is_contended=False)
        )
        # Crop valid
        self.assertTrue(
            self.scheduler.should_extract(session, crop_shape=(90, 90), current_frame=5, is_contended=False)
        )


if __name__ == "__main__":
    unittest.main()

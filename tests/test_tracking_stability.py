import unittest
import numpy as np

from src.inference.person_detector import UltralyticsPersonDetector, DetectedBox
from src.inference.tracking_pre_filter import TrackingPreFilter
from src.domain.track_registry import TrackRegistry, TrackStatus
from src.utils.jitter_filter import JitterFilterEngine


class DummyFaceAligner:
    def __init__(self, should_align: bool = True):
        self.should_align = should_align

    def align_face(self, crop, strict: bool = False):
        if self.should_align and crop is not None and crop.size > 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return None


class TestTrackingStability(unittest.TestCase):
    def setUp(self):
        self.face_aligner = DummyFaceAligner(should_align=True)
        self.jitter_engine = JitterFilterEngine()
        self.pre_filter = TrackingPreFilter(
            face_aligner=self.face_aligner,
            pose_estimator=None,
            jitter_engine=self.jitter_engine,
        )

    def test_seated_upper_body_crop_preserves_full_face(self):
        """Webcam/seated boxes (aspect >= 0.55) must preserve chin and not slice off the bottom 45%."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        # Typical seated webcam box: 600 wide by 500 tall (aspect = 1.2)
        box = (200, 100, 800, 600)
        has_presence = self.pre_filter.has_head_or_shoulder_presence(frame, box, track_id=1, current_frame_num=1)
        self.assertTrue(has_presence)

    def test_single_frame_suppression_does_not_purge_jitter_filter(self):
        """Temporarily suppressed detections must retain Kalman smoothing state and not be purged."""
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        # Prime track 1 in jitter filter
        box1 = (100, 100, 300, 400)
        _ = self.jitter_engine.filter_box(1, box1, confidence=0.85)
        self.assertIn(1, self.jitter_engine.smoothed_states)
        self.assertIn(1, self.jitter_engine.kalman_filters)

        # Candidate 2 heavily contained inside candidate 1
        class Det:
            def __init__(self, box, conf, track_id):
                self.box = box
                self.conf = conf
                self.track_id = track_id

        dets = [
            Det((100, 100, 300, 400), 0.90, 1),
            Det((120, 120, 280, 380), 0.50, 2),
        ]
        res = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=dets,
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=1,
            active_track_ids={1},
            confirmed_track_ids=set()
        )
        # Track 2 is suppressed in this frame
        self.assertIn(2, res.suppressed_track_ids)
        # But track 1's smoothing state is intact
        self.assertIn(1, self.jitter_engine.smoothed_states)

    def test_actively_tracked_candidate_not_expired_as_phantom(self):
        """Subjects actively tracked each frame must never be purged as phantoms at frame 46."""
        registry = TrackRegistry()
        s = registry.get_or_create(1, (100, 100, 300, 400), current_frame=1)
        # Actively updated at frame 50 (age = 50, last_seen = 50)
        s.last_seen = 50
        s.valid_face_crops = 0

        expired = registry.find_expired_tracks(current_frame=50, max_phantom_age=45)
        self.assertNotIn(1, expired, "Actively seen track was falsely purged as a phantom!")

    def test_missing_track_with_zero_faces_expires_normally(self):
        """Tracks missing for > 15 frames with 0 face crops must still expire as phantoms."""
        registry = TrackRegistry()
        s = registry.get_or_create(1, (100, 100, 300, 400), current_frame=1)
        s.last_seen = 20  # Missing for 30 frames at frame 50
        s.valid_face_crops = 0

        expired = registry.find_expired_tracks(current_frame=50, max_phantom_age=45)
        self.assertIn(1, expired)


if __name__ == "__main__":
    unittest.main()

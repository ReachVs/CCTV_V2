import unittest
import numpy as np
from typing import Dict, Any, List

from src.inference.tracking_pre_filter import TrackingPreFilter, FilteredCandidate


class DummyFaceAligner:
    def __init__(self, should_align: bool = True):
        self.should_align = should_align

    def align_face(self, crop, strict: bool = True):
        if self.should_align:
            return np.zeros((112, 112, 3), dtype=np.uint8)
        return None


class DummyPoseEstimator:
    def __init__(self, keypoints=None):
        self.keypoints = keypoints or []
        self.yolo_pose = object()

    def estimate_pose_keypoints(self, frame, box):
        return self.keypoints


class DummyJitterEngine:
    def filter_box(self, track_id, box, confidence=1.0):
        return box


class TestTrackingPreFilter(unittest.TestCase):
    def setUp(self):
        self.face_aligner = DummyFaceAligner(should_align=True)
        self.pose_estimator = DummyPoseEstimator(keypoints=[{"id": 0, "conf": 0.9}])
        self.jitter_engine = DummyJitterEngine()
        self.pre_filter = TrackingPreFilter(
            face_aligner=self.face_aligner,
            pose_estimator=self.pose_estimator,
            jitter_engine=self.jitter_engine,
        )

    def test_filter_candidate_filters_by_confidence(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        class Det:
            def __init__(self, box, conf, track_id):
                self.box = box
                self.conf = conf
                self.track_id = track_id

        # Low conf candidate (< 0.45 for new track)
        dets = [Det((100, 100, 200, 300), 0.30, 1)]
        res = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=dets,
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=1,
            active_track_ids=set(),
            confirmed_track_ids=set()
        )
        self.assertEqual(len(res.candidates), 0)

        # High conf candidate (>= 0.45)
        dets_valid = [Det((100, 100, 250, 320), 0.75, 1)]
        res_valid = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=dets_valid,
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=1,
            active_track_ids=set(),
            confirmed_track_ids=set()
        )
        self.assertEqual(len(res_valid.candidates), 1)
        self.assertEqual(res_valid.candidates[0].track_id, 1)

    def test_spatial_overlap_suppression(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        class Det:
            def __init__(self, box, conf, track_id):
                self.box = box
                self.conf = conf
                self.track_id = track_id

        # Two overlapping boxes: track 1 has larger area/higher conf, track 2 is inside track 1
        dets = [
            Det((100, 100, 300, 300), 0.85, 1),
            Det((120, 120, 280, 280), 0.60, 2),  # Severe containment
        ]
        res = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=dets,
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=1,
            active_track_ids=set(),
            confirmed_track_ids=set()
        )
        # Track 2 should be suppressed by spatial containment
        self.assertEqual(len(res.candidates), 1)
        self.assertEqual(res.candidates[0].track_id, 1)

    def test_pairwise_crossover_contention_and_hysteresis(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        class Det:
            def __init__(self, box, conf, track_id):
                self.box = box
                self.conf = conf
                self.track_id = track_id

        # Frame 1: Two established tracks cross over with IoU ~ 0.42
        dets = [
            Det((100, 100, 300, 500), 0.80, 1),
            Det((180, 100, 380, 500), 0.80, 2),
        ]
        res1 = self.pre_filter.filter_and_arbitrate(
            frame=frame,
            detections=dets,
            scale_x=1.0,
            scale_y=1.0,
            current_frame_num=1,
            active_track_ids={1, 2},
            confirmed_track_ids=set()
        )
        self.assertIn(1, self.pre_filter.contended_tracks)
        self.assertIn(2, self.pre_filter.contended_tracks)

        # Separate the tracks (IoU < 0.15) for 3 consecutive frames
        separated_dets = [
            Det((50, 100, 150, 300), 0.80, 1),
            Det((300, 100, 400, 300), 0.80, 2),
        ]
        # Frame 2 (separation count = 1)
        self.pre_filter.filter_and_arbitrate(frame, separated_dets, 1.0, 1.0, 2, {1, 2}, set())
        self.assertEqual(self.pre_filter.contended_tracks.get(1), 1)

        # Frame 3 (separation count = 2)
        self.pre_filter.filter_and_arbitrate(frame, separated_dets, 1.0, 1.0, 3, {1, 2}, set())
        self.assertEqual(self.pre_filter.contended_tracks.get(1), 2)

        # Frame 4 (separation count reaches 3 -> exits contended tracks)
        res4 = self.pre_filter.filter_and_arbitrate(frame, separated_dets, 1.0, 1.0, 4, {1, 2}, set())
        self.assertNotIn(1, self.pre_filter.contended_tracks)
        self.assertNotIn(2, self.pre_filter.contended_tracks)
        self.assertIn(1, res4.resolved_contended_track_ids)
        self.assertIn(2, res4.resolved_contended_track_ids)


if __name__ == "__main__":
    unittest.main()

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import (
    BiometricTrackingEngine,
    InMemoryIndexAdapter
)

class MockBox:
    def __init__(self, xyxy, conf, tid):
        self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.id = torch.tensor(tid, dtype=torch.int32)

    def __len__(self):
        return len(self.xyxy)

class MockResult:
    def __init__(self, boxes):
        self.boxes = boxes

class MockYOLO:
    def __init__(self, detections_fn):
        self.detections_fn = detections_fn

    def track(self, *args, **kwargs):
        boxes_data = self.detections_fn()
        return [MockResult(boxes_data)]


class MockPoseEstimator:
    def __init__(self, keypoints_by_box=None):
        self.yolo_pose = True  # Flag to indicate pose estimator is active
        self.keypoints_by_box = keypoints_by_box or {}

    def estimate_pose_keypoints(self, frame, bbox):
        bbox_tuple = tuple(bbox)
        return self.keypoints_by_box.get(bbox_tuple, [])


class TestLimbNoiseSuppression(unittest.TestCase):

    def setUp(self):
        self.adapter = InMemoryIndexAdapter(dimension=512)
        self.engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            metadata_path=None,
            l2_threshold=0.68,
            margin_threshold=0.15
        )
        self.engine.metadata = {}
        self.engine.use_cuda = False
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def test_bare_limb_rejection_at_inception(self):
        """Test 1: Bare arm/hand box with NO head or shoulder keypoints and no face
        must be dropped prior to active_tracks admission."""
        limb_box_full = (100, 100, 250, 400)
        limb_box_yolo = [c / 2.0 for c in limb_box_full]

        # Mock pose returning only wrist/elbow keypoints (indices 7, 8, 9, 10 - NO 0..6 head/shoulders)
        mock_kpts = [
            {"id": 7, "x": 150.0, "y": 200.0, "conf": 0.85},   # L-Elbow
            {"id": 9, "x": 160.0, "y": 300.0, "conf": 0.90},   # L-Wrist
        ]
        self.engine.pose_estimator = MockPoseEstimator({
            limb_box_full: mock_kpts
        })

        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [limb_box_yolo],
            [0.85],
            [50]
        ))

        res = self.engine.process_frame(self.frame, annotate=False)
        active_ids = [s.track_id for s in res.active_tracks]

        self.assertNotIn(50, active_ids, "Bare limb track 50 must NOT be present in active_tracks")
        self.assertNotIn(50, self.engine.active_tracks, "Track 50 must NOT enter engine.active_tracks")
        print("[PASS] Test 1: Bare limb candidate successfully rejected at inception.")

    def test_head_or_shoulder_presence_acceptance(self):
        """Test 2: Candidate boxes with shoulder or head keypoints must be accepted."""
        # 1. Candidate with Shoulder keypoint (index 5, conf=0.75)
        shoulder_box = (100, 100, 300, 500)
        shoulder_kpts = [
            {"id": 5, "x": 180.0, "y": 150.0, "conf": 0.75}  # L-Shoulder
        ]
        self.engine.pose_estimator = MockPoseEstimator({
            shoulder_box: shoulder_kpts
        })
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in shoulder_box]],
            [0.85],
            [10]
        ))

        res1 = self.engine.process_frame(self.frame, annotate=False)
        active_ids1 = [s.track_id for s in res1.active_tracks]
        self.assertIn(10, active_ids1, "Track with shoulder keypoint must be accepted")

        # 2. Candidate with Nose keypoint (index 0, conf=0.80)
        head_box = (400, 100, 600, 500)
        head_kpts = [
            {"id": 0, "x": 500.0, "y": 140.0, "conf": 0.80}  # Nose
        ]
        self.engine.pose_estimator = MockPoseEstimator({
            head_box: head_kpts
        })
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in head_box]],
            [0.85],
            [11]
        ))

        res2 = self.engine.process_frame(self.frame, annotate=False)
        active_ids2 = [s.track_id for s in res2.active_tracks]
        self.assertIn(11, active_ids2, "Track with facial/head keypoint must be accepted")
        print("[PASS] Test 2: Upper-body (shoulder/head) candidates accepted into tracking pipeline.")

    def test_fifteen_frame_ttl_cache(self):
        """Test 3: Verified head/shoulder presence is cached for 15 frames,
        ensuring fast throughput without calling pose estimation on every frame."""
        box = (100, 100, 300, 500)
        call_count = [0]

        class CountingPoseEstimator:
            def __init__(self):
                self.yolo_pose = True

            def estimate_pose_keypoints(self, frame, bbox):
                call_count[0] += 1
                return [{"id": 0, "x": 150.0, "y": 150.0, "conf": 0.85}]

        self.engine.pose_estimator = CountingPoseEstimator()
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in box]],
            [0.85],
            [20]
        ))

        # Frame 1: Validates and caches
        self.engine.process_frame(self.frame, annotate=False)
        self.assertEqual(call_count[0], 1, "Expected pose check on frame 1")
        self.assertIn(20, self.engine.head_presence_cache)
        exp_frame = self.engine.head_presence_cache[20]
        self.assertEqual(exp_frame, self.engine.frame_count + 15)

        # Frames 2 to 10: Should reuse cache and NOT call estimate_pose_keypoints
        for _ in range(9):
            self.engine.process_frame(self.frame, annotate=False)

        self.assertEqual(call_count[0], 1, f"Pose check should not be re-called within 15-frame TTL; called {call_count[0]} times")
        print("[PASS] Test 3: 15-frame TTL cache successfully bypassed redundant pose calls.")

    def test_persistent_anti_limbo_mutation(self):
        """Test 4: An unconfirmed track reaching >25 frames with 0 valid face crops
        must persist 'Unverified ID {id}' directly into engine.active_tracks."""
        box = (100, 100, 300, 500)
        self.engine.pose_estimator = MockPoseEstimator({
            box: [{"id": 5, "x": 150.0, "y": 150.0, "conf": 0.85}]
        })
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in box]],
            [0.85],
            [30]
        ))

        # Run 26 frames
        for _ in range(26):
            res = self.engine.process_frame(self.frame, annotate=False)

        # Track 30 should now be persistent Unverified
        self.assertEqual(self.engine.active_tracks.get(30), "Unverified ID 30")
        subj = next((s for s in res.active_tracks if s.track_id == 30), None)
        self.assertIsNotNone(subj)
        self.assertEqual(subj.name, "Unverified ID 30")
        self.assertFalse(subj.is_scanning)

        # Frame 27 should still be Unverified ID 30 (not reverting to Scanning)
        res_next = self.engine.process_frame(self.frame, annotate=False)
        subj_next = next((s for s in res_next.active_tracks if s.track_id == 30), None)
        self.assertEqual(subj_next.name, "Unverified ID 30")
        print("[PASS] Test 4: Anti-limbo successfully mutates and persists 'Unverified ID {id}'.")

    def test_headless_eviction_after_45_frames(self):
        """Test 5: An unconfirmed track with 0 valid face crops persisting past 45 frames
        must be purged from the engine."""
        box = (100, 100, 300, 500)
        self.engine.pose_estimator = MockPoseEstimator({
            box: [{"id": 5, "x": 150.0, "y": 150.0, "conf": 0.85}]
        })
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in box]],
            [0.85],
            [40]
        ))

        # Run 46 frames
        for _ in range(46):
            self.engine.process_frame(self.frame, annotate=False)

        # Track 40 must now be purged from engine state
        self.assertNotIn(40, self.engine.active_tracks, "Track 40 should be purged after 45 frames of 0 face crops")
        self.assertNotIn(40, self.engine.track_last_seen, "Track 40 should be purged from track_last_seen")
        print("[PASS] Test 5: Headless track successfully evicted after 45 frames of zero face crops.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

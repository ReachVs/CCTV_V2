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
    compute_box_iou,
    compute_box_containment,
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


class TestCrossoverContentionResolution(unittest.TestCase):

    def setUp(self):
        self.adapter = InMemoryIndexAdapter(dimension=512)
        # Vector for Alice
        self.vec_alice = np.zeros((1, 512), dtype=np.float32)
        self.vec_alice[0, 0] = 1.0
        self.adapter.add(self.vec_alice)

        # Vector for Bob
        self.vec_bob = np.zeros((1, 512), dtype=np.float32)
        self.vec_bob[0, 1] = 1.0
        self.adapter.add(self.vec_bob)

        self.engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            metadata_path=None,
            l2_threshold=0.68,
            margin_threshold=0.15,
            consensus_votes=2
        )
        self.engine.metadata = {
            0: {"name": "Alice", "role": "Employee"},
            1: {"name": "Bob", "role": "Visitor"}
        }
        self.engine.use_cuda = False
        class MockPersonPose:
            yolo_pose = True
            def estimate_pose_keypoints(self, frame, bbox):
                return [{"id": 0, "x": (bbox[0] + bbox[2]) / 2.0, "y": (bbox[1] + bbox[3]) / 2.0, "conf": 0.85}]
        self.engine.pose_estimator = MockPersonPose()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _setup_confirmed_tracks(self, t1_box, t2_box):
        """Helper to preload two confirmed tracks in the engine."""
        with self.engine.lock:
            # Track 1: Alice
            self.engine.active_tracks[1] = "Alice"
            self.engine.track_cache[1] = {
                "name": "Alice",
                "is_confirmed": True,
                "history": ["Alice", "Alice"],
                "best_dist": 0.20,
                "best_norm": 1.0
            }
            self.engine.recent_track_history[1] = {
                "box": t1_box,
                "last_seen": 1,
                "name": "Alice",
                "confirmed": True
            }
            self.engine.track_first_seen[1] = 1
            self.engine.track_last_seen[1] = 1

            # Track 2: Bob
            self.engine.active_tracks[2] = "Bob"
            self.engine.track_cache[2] = {
                "name": "Bob",
                "is_confirmed": True,
                "history": ["Bob", "Bob"],
                "best_dist": 0.22,
                "best_norm": 1.0
            }
            self.engine.recent_track_history[2] = {
                "box": t2_box,
                "last_seen": 1,
                "name": "Bob",
                "confirmed": True
            }
            self.engine.track_first_seen[2] = 1
            self.engine.track_last_seen[2] = 1

    def test_pairwise_crossover_trigger_and_state_wipe(self):
        """Test 1: When two confirmed tracks cross and overlap (IoU > 0.30),
        both must immediately revert to is_confirmed=False, is_contended=True,
        have voting history wiped, and generate a real-time CROSSOVER_CONTENTION event."""
        # Overlapping boxes (IoU ~ 0.38)
        box1_full = (100, 100, 300, 500)
        box2_full = (180, 100, 380, 500)
        self._setup_confirmed_tracks(box1_full, box2_full)

        # Scale down for YOLO input (640x360 vs 1280x720 -> scale 2.0)
        box1_yolo = [coord / 2.0 for coord in box1_full]
        box2_yolo = [coord / 2.0 for coord in box2_full]

        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [box1_yolo, box2_yolo],
            [0.85, 0.85],
            [1, 2]
        ))

        result = self.engine.process_frame(self.frame, annotate=True)

        # Check audit events
        crossover_events = [e for e in result.audit_events if e.get("event_type") == "CROSSOVER_CONTENTION"]
        self.assertEqual(len(crossover_events), 1, "Expected 1 CROSSOVER_CONTENTION audit event")
        self.assertEqual(crossover_events[0]["track_id_1"], 1)
        self.assertEqual(crossover_events[0]["track_id_2"], 2)

        # Check TrackedSubjects returned
        subj_map = {s.track_id: s for s in result.active_tracks}
        self.assertIn(1, subj_map)
        self.assertIn(2, subj_map)

        # Both must be marked contended
        self.assertTrue(subj_map[1].is_contended, "Track 1 must be marked is_contended=True")
        self.assertTrue(subj_map[2].is_contended, "Track 2 must be marked is_contended=True")

        # Both must be unconfirmed
        self.assertFalse(subj_map[1].is_confirmed, "Track 1 must have is_confirmed=False")
        self.assertFalse(subj_map[2].is_confirmed, "Track 2 must have is_confirmed=False")

        # Internal cache history must be wiped
        self.assertEqual(self.engine.track_cache[1]["history"], [], "Track 1 history must be empty")
        self.assertEqual(self.engine.track_cache[2]["history"], [], "Track 2 history must be empty")

        # Engine contended_tracks state must register both
        self.assertIn(1, self.engine.contended_tracks)
        self.assertIn(2, self.engine.contended_tracks)

        print("[PASS] Test 1: Pairwise crossover contention triggered and clean state wipe verified.")

    def test_face_extraction_freeze_during_contention(self):
        """Test 2: While tracks are contended, face crop extraction must be frozen
        to avoid sampling overlapping bodies and cross-contaminating embeddings."""
        box1_full = (100, 100, 300, 500)
        box2_full = (180, 100, 380, 500)
        self._setup_confirmed_tracks(box1_full, box2_full)

        box1_yolo = [c / 2.0 for c in box1_full]
        box2_yolo = [c / 2.0 for c in box2_full]

        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [box1_yolo, box2_yolo],
            [0.85, 0.85],
            [1, 2]
        ))

        # Clear embedder queue
        with self.engine._embedder_lock:
            self.engine._embedder_queue.clear()

        # Process frame where they cross
        self.engine.process_frame(self.frame, annotate=False)

        # Verify that no face crops were submitted to embedder worker
        with self.engine._embedder_lock:
            queue_len = len(self.engine._embedder_queue)
        self.assertEqual(queue_len, 0, f"Expected 0 crops submitted during contention, got {queue_len}")

        print("[PASS] Test 2: Face crop extraction successfully frozen during contention.")

    def test_three_frame_separation_hysteresis(self):
        """Test 3: Verify that after separating (IoU < 0.15), tracks must remain in
        extraction freeze for exactly 3 consecutive frames before freeze is lifted."""
        # 1. Start in contention
        box1_full = (100, 100, 300, 500)
        box2_full = (180, 100, 380, 500)
        self._setup_confirmed_tracks(box1_full, box2_full)

        # Trigger contention
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in box1_full], [c / 2.0 for c in box2_full]],
            [0.85, 0.85],
            [1, 2]
        ))
        self.engine.process_frame(self.frame, annotate=False)
        self.assertIn(1, self.engine.contended_tracks)
        self.assertEqual(self.engine.contended_tracks[1], 0)

        # 2. Now separate the boxes (IoU = 0.0)
        sep_box1 = (100, 100, 250, 500)
        sep_box2 = (600, 100, 750, 500)
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in sep_box1], [c / 2.0 for c in sep_box2]],
            [0.85, 0.85],
            [1, 2]
        ))

        # Frame 1 of separation
        res1 = self.engine.process_frame(self.frame, annotate=False)
        self.assertIn(1, self.engine.contended_tracks, "Frame 1: Track 1 must still be in contended_tracks")
        self.assertEqual(self.engine.contended_tracks[1], 1, "Frame 1 separation count should be 1")

        # Frame 2 of separation
        res2 = self.engine.process_frame(self.frame, annotate=False)
        self.assertIn(1, self.engine.contended_tracks, "Frame 2: Track 1 must still be in contended_tracks")
        self.assertEqual(self.engine.contended_tracks[1], 2, "Frame 2 separation count should be 2")

        # Frame 3 of separation -> Hysteresis threshold reached, should exit contended_tracks!
        res3 = self.engine.process_frame(self.frame, annotate=False)
        self.assertNotIn(1, self.engine.contended_tracks, "Frame 3: Track 1 must exit contended_tracks after 3 frames")
        self.assertNotIn(2, self.engine.contended_tracks, "Frame 3: Track 2 must exit contended_tracks after 3 frames")

        # Verify TrackedSubject reports is_contended=False
        subj_map = {s.track_id: s for s in res3.active_tracks}
        self.assertFalse(subj_map[1].is_contended, "Track 1 is_contended should be False on frame 3")
        self.assertFalse(subj_map[2].is_contended, "Track 2 is_contended should be False on frame 3")

        print("[PASS] Test 3: 3-frame separation hysteresis successfully enforced.")

    def test_spatial_inheritance_contention_blacklist(self):
        """Test 4: Tracks involved in crossover contention are blacklisted for 45 frames.
        A newly spawned track ID in the crossing zone must NOT inherit the identity via spatial IoU."""
        box1_full = (100, 100, 300, 500)
        box2_full = (180, 100, 380, 500)
        self._setup_confirmed_tracks(box1_full, box2_full)

        # Trigger contention
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [[c / 2.0 for c in box1_full], [c / 2.0 for c in box2_full]],
            [0.85, 0.85],
            [1, 2]
        ))
        self.engine.process_frame(self.frame, annotate=False)

        # Both tracks should now be blacklisted
        self.assertIn(1, self.engine.contention_blacklist)
        self.assertIn(2, self.engine.contention_blacklist)

        # Now simulate Track 1 dropped by ByteTrack and re-appearing as Track 3 at the same spot
        # Under old naive spatial IoU, Track 3 would inherit "Alice" because IoU > 0.40!
        box3_yolo = [c / 2.0 for c in box1_full]
        self.engine._yolo_model = MockYOLO(lambda: MockBox(
            [box3_yolo],
            [0.85],
            [3]
        ))

        res = self.engine.process_frame(self.frame, annotate=False)
        subj3 = next((s for s in res.active_tracks if s.track_id == 3), None)
        self.assertIsNotNone(subj3)
        self.assertNotEqual(subj3.name, "Alice", "Track 3 must NOT inherit Alice while blacklisted")
        self.assertTrue(subj3.name.startswith("Scanning Track 3"), f"Track 3 should start as Scanning, got {subj3.name}")

        print("[PASS] Test 4: Spatial inheritance correctly blocked by contention blacklist.")

    def test_independent_post_separation_verification(self):
        """Test 5: After separation, when Track 1 faces camera and reaches consensus for Alice,
        Track 2 remains unconfirmed/scanning and does not steal or share the identity."""
        # 1. Track 1 and Track 2 separated
        box1 = (100, 100, 250, 500)
        box2 = (600, 100, 750, 500)
        self._setup_confirmed_tracks(box1, box2)

        # Make them active scanning tracks post-separation
        with self.engine.lock:
            self.engine.active_tracks[1] = "Scanning Track 1"
            self.engine.track_cache[1] = {"name": "Scanning Track 1", "is_confirmed": False, "history": []}
            self.engine.active_tracks[2] = "Scanning Track 2"
            self.engine.track_cache[2] = {"name": "Scanning Track 2", "is_confirmed": False, "history": []}

        # Submit Alice embedding twice for Track 1 to reach consensus (consensus_votes=2)
        disp1, telem1 = self.engine.process_track_frame(track_id=1, embedding=self.vec_alice, track_age=10)
        self.assertFalse(telem1["confirmed"])  # 1 vote, not yet confirmed

        disp2, telem2 = self.engine.process_track_frame(track_id=1, embedding=self.vec_alice, track_age=11)
        self.assertTrue(telem2["confirmed"])  # 2 votes, confirmed as Alice!
        self.assertEqual(disp2, "Alice")

        # Verify Track 2 state is unaffected
        self.assertEqual(self.engine.active_tracks[2], "Scanning Track 2")
        self.assertFalse(self.engine.track_cache[2].get("is_confirmed", False))

        print("[PASS] Test 5: Independent post-separation verification verified.")

if __name__ == "__main__":
    unittest.main(verbosity=2)

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import BiometricTrackingEngine, InMemoryIndexAdapter
from src.utils.privacy_compliance import PoseOnlyEstimator


class TestHandUnverifiedRepro(unittest.TestCase):

    def test_hand_crop_pose_keypoints_not_synthetic(self):
        """Bug 1 Repro: PoseOnlyEstimator on an empty/hand crop must NOT return synthetic keypoints
        with high confidence (0.85), which tricks the head/shoulder presence gate."""
        estimator = PoseOnlyEstimator()
        # Blank or noise image representing a hand/window background without a person
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        hand_box = (100, 100, 300, 400)

        kpts = estimator.estimate_pose_keypoints(blank_frame, hand_box)
        # Check if head/shoulder keypoints (0..6) have high confidence on a blank/hand crop
        head_shoulder_confs = [kp["conf"] for kp in kpts if 0 <= kp.get("id", -1) <= 6]
        max_conf = max(head_shoulder_confs) if head_shoulder_confs else 0.0
        
        print(f"[REPRO CHECK] Head/Shoulder max confidence on blank/hand box: {max_conf}")
        self.assertLess(max_conf, 0.35, f"Expected conf < 0.35 on hand/blank crop, but got {max_conf} (synthetic keypoints leaking!)")

    def test_unverified_box_color_not_green(self):
        """Bug 2 Repro: An unverified subject (e.g. Unverified ID 8) must NOT be drawn in GREEN (0, 255, 0).
        It must be drawn in muted gray (130, 130, 130) and is_unverified must be True."""
        adapter = InMemoryIndexAdapter(dimension=512)
        engine = BiometricTrackingEngine(index_adapter=adapter, metadata_path=None)
        
        # Simulate track 8 as already transitioned to "Unverified ID 8"
        with engine.lock:
            engine.active_tracks[8] = "Unverified ID 8"
            engine.track_cache[8] = {
                "name": "Unverified ID 8",
                "is_confirmed": False,
                "valid_face_crops": 0
            }
            engine.track_first_seen[8] = 1
            engine.track_last_seen[8] = 1

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Mock YOLO returning track 8 box
        class MockBox:
            def __init__(self):
                import torch
                self.xyxy = torch.tensor([[100/2.0, 100/2.0, 300/2.0, 400/2.0]], dtype=torch.float32)
                self.conf = torch.tensor([0.85], dtype=torch.float32)
                self.id = torch.tensor([8], dtype=torch.int32)
            def __len__(self):
                return 1
        class MockResult:
            def __init__(self):
                self.boxes = MockBox()
        class MockYOLO:
            def track(self, *args, **kwargs):
                return [MockResult()]

        engine._yolo_model = MockYOLO()
        # Bypass head check for this specific rendering test
        engine.head_presence_cache[8] = 9999

        res = engine.process_frame(frame, annotate=True)
        subj8 = next((s for s in res.active_tracks if s.track_id == 8), None)
        self.assertIsNotNone(subj8)
        self.assertFalse(subj8.is_confirmed, "Unverified track 8 must NOT be confirmed")

        # Now check pixel colors in annotated frame around label box
        # Green is (0, 255, 0). Gray is (130, 130, 130).
        pixel_color = res.annotated_frame[100 - 10, 100 + 5].tolist()
        print(f"[REPRO CHECK] Track 8 name: {subj8.name}, is_confirmed: {subj8.is_confirmed}, pixel color: {pixel_color}")
        self.assertFalse(pixel_color[1] == 255 and pixel_color[0] == 0 and pixel_color[2] == 0,
                         f"Box was drawn in GREEN {pixel_color} instead of GRAY (130, 130, 130)!")


if __name__ == "__main__":
    unittest.main()

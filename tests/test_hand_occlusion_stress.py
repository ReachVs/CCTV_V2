import os
import sys
import unittest
import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import BiometricTrackingEngine, InMemoryIndexAdapter
from src.inference.person_detector import DetectedBox, ScriptedPersonDetector
from src.inference.face_embedder import FaceEmbedder


class MockFaceEmbedder(FaceEmbedder):
    def __init__(self, target_embedding=None):
        self.target_embedding = target_embedding
        self.blocked = False

    def embed(self, img_or_crop, align=True, strict_alignment=False, upscale=False):
        if self.blocked:
            return None
        return self.target_embedding


class TestHandOcclusionStress(unittest.TestCase):
    """
    Regression suite validating performance and tracking stability
    during multi-person stress tests with hand/limb face occlusion.
    """

    def setUp(self):
        self.adapter = InMemoryIndexAdapter(dimension=512)
        # Register Profile "Alice"
        self.alice_vec = np.random.randn(512).astype(np.float32)
        self.alice_vec /= np.linalg.norm(self.alice_vec)
        self.adapter.add(self.alice_vec.reshape(1, -1))

        self.detector = ScriptedPersonDetector()
        self.embedder = MockFaceEmbedder(target_embedding=self.alice_vec)

        self.engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            detector=self.detector,
            embedder=self.embedder,
            metadata_path=None,
            consensus_votes=2
        )
        self.engine.metadata = {"0": "Alice"}
        # Pre-seed positive head presence cache so track 1 is admitted as a valid person
        self.engine.pre_filter.head_presence_cache[1] = 9999

    def test_active_track_not_purged_when_hand_blocks_face(self):
        """
        Verify that an active person whose face is blocked by hands/limbs
        is NOT purged from active_tracks and does NOT trigger ID thrashing.
        """
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        box = (100, 100, 300, 500)

        # 1. Admit Track 1 into tracking
        self.detector.set_detections([DetectedBox(box=box, conf=0.88, track_id=1)])
        res1 = self.engine.process_frame(frame, annotate=False)
        self.assertIn(1, self.engine.active_tracks, "Track 1 should be admitted into active_tracks")

        # 2. Hand blocks face for 20 frames (embedder returns None, face aligner returns None)
        self.embedder.blocked = True
        for f in range(20):
            res = self.engine.process_frame(frame, annotate=False)
            self.assertIn(1, self.engine.active_tracks, f"Track 1 must NOT be purged at frame {f+1} while face is blocked by hand")

        # 3. Verify track state transitioned to Unverified gracefully after 25 frames
        for f in range(10):
            res = self.engine.process_frame(frame, annotate=False)

        subj = next((s for s in res.active_tracks if s.track_id == 1), None)
        self.assertIsNotNone(subj)
        self.assertTrue(subj.name.startswith("Unverified") or subj.name.startswith("Scanning"), 
                        f"Expected Unverified/Scanning name, got {subj.name}")
        self.assertFalse(subj.is_confirmed, "Occluded subject must not be falsely confirmed")

    def test_negative_caching_prevents_per_frame_pose_re_execution(self):
        """
        Verify that negative head presence outcomes are cached, preventing
        repeated synchronous neural inference on every frame for blocked candidates.
        """
        pre_filter = self.engine.pre_filter
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dummy_box = (100, 100, 200, 300)

        # Call on blank candidate (both YuNet and Pose fail)
        res1 = pre_filter.has_head_or_shoulder_presence(frame, dummy_box, track_id=99, current_frame_num=10)
        self.assertFalse(res1)
        self.assertIn(99, pre_filter.head_presence_negative_cache)
        self.assertGreaterEqual(pre_filter.head_presence_negative_cache[99], 25)

        # Subsequent call at frame 11 must hit cache immediately
        res2 = pre_filter.has_head_or_shoulder_presence(frame, dummy_box, track_id=99, current_frame_num=11)
        self.assertFalse(res2)


if __name__ == "__main__":
    unittest.main()

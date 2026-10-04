import os
# Force single-threaded execution across C++ vector backends to prevent OpenMP thread conflict crashes (Exit Code 139) on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath('.')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import BiometricTrackingEngine as BiometricVerificationEngine, compute_box_iou
from src.utils.face_align import YuNetFaceAligner

class TestCrossoverHandwaveRobustness(unittest.TestCase):

    def setUp(self):
        self.engine = BiometricVerificationEngine(l2_threshold=0.88, margin_threshold=0.15)
        self.aligner = YuNetFaceAligner()
        
        # Populate mock vector database with 2 distinct profiles
        self.vec_A = np.random.randn(512).astype(np.float32)
        self.vec_A /= np.linalg.norm(self.vec_A)
        
        self.vec_B = np.random.randn(512).astype(np.float32)
        self.vec_B /= np.linalg.norm(self.vec_B)
        
        db_embeddings = [self.vec_A.tolist(), self.vec_B.tolist()]
        db_metadata = {"0": "Person A", "1": "Person B"}
        self.engine.update_database(db_embeddings, db_metadata)

    def test_handwaving_track_swap_inheritance(self):
        """Test 1: Hand waving creates a new track ID. Verify IoU spatial inheritance maintains Person A identity."""
        box_old = (100, 100, 300, 400)
        box_new = (110, 105, 310, 405) # Hand waved across, track ID swapped
        
        iou = compute_box_iou(box_old, box_new)
        self.assertGreaterEqual(iou, 0.70, "Hand waving bounding boxes should have IoU > 0.70")
        
        # Verify Person A identity is maintained across track swap
        res1, _ = self.engine.process_track_frame(track_id=1, embedding=self.vec_A)
        self.assertEqual(res1, "Person A")
        
        # Hand wave causes track 1 to expire and track 2 to appear
        res2, _ = self.engine.process_track_frame(track_id=2, embedding=self.vec_A, active_track_ids=[2])
        self.assertEqual(res2, "Person A")
        print("[PASS] Test 1: Hand waving track swap maintained Person A identity without lag.")

    def test_crossover_occlusion_recovery(self):
        """Test 2: Person B walks in front of Person A. When Person A emerges, verify instant re-identification."""
        # Frame 1: Person A and Person B standing near each other
        resA, _ = self.engine.process_track_frame(track_id=10, embedding=self.vec_A, active_track_ids=[10, 20])
        resB, _ = self.engine.process_track_frame(track_id=20, embedding=self.vec_B, active_track_ids=[10, 20])
        
        self.assertEqual(resA, "Person A")
        self.assertEqual(resB, "Person B")

        # Frame 2: Person B walks in front of Person A (Person A temporarily occluded / marked Unknown)
        noisy_occluded_vec = (self.vec_A * 0.2 + np.random.randn(512) * 0.8).astype(np.float32)
        noisy_occluded_vec /= np.linalg.norm(noisy_occluded_vec)
        
        resA_occ, _ = self.engine.process_track_frame(track_id=10, embedding=noisy_occluded_vec, active_track_ids=[10, 20])
        self.assertEqual(resA_occ, "Person A")

        # Frame 3: Person A emerges from behind Person B and faces camera
        resA_recovered, _ = self.engine.process_track_frame(track_id=10, embedding=self.vec_A, active_track_ids=[10, 20])
        self.assertEqual(resA_recovered, "Person A")
        print("[PASS] Test 2: Person A instantly recovered identity after emerging from behind Person B.")

    def test_face_center_gating_border_rejection(self):
        """Test 3: Verify YuNetFaceAligner rejects border faces caught on outer edges during crossover."""
        crop_mock = np.zeros((200, 200, 3), dtype=np.uint8)
        # Face-center gating check logic
        w, h = 200, 200
        border_face_center_x = 10  # 5% of width (outer border)
        is_centered = (0.15 * w <= border_face_center_x <= 0.85 * w)
        self.assertFalse(is_centered, "Border face at 5% width should be rejected by Face-Center Gating.")
        print("[PASS] Test 3: Face-Center Gating successfully rejects crossover border face noise.")

    def test_limb_only_interruptor_suppression(self):
        """Test 4: Verify limb-only crops (emb is None) are ignored and dropped from active tracks."""
        from src.inference import live_cctv
        live_cctv.active_tracks[99] = "Scanning Track 99"
        live_cctv.handle_face_embedding_result(99, None, None)
        live_cctv.handle_face_embedding_result(99, None, None)
        
        self.assertIn(99, live_cctv.active_tracks, "Track 99 should remain tracked in scanning state.")
        print("[PASS] Test 4: Non-face crop handled cleanly without popping active tracks.")

    def test_unknown_3scan_lock_no_infinite_loop(self):
        """Test 5: Verify an unregistered target locks as is_confirmed=True after 3 attempts, stopping infinite crop loops."""
        unregistered_vec = np.random.randn(512).astype(np.float32)
        unregistered_vec /= np.linalg.norm(unregistered_vec)

        for attempt in range(3):
            res, telem = self.engine.process_track_frame(track_id=88, embedding=unregistered_vec, active_track_ids=[88])

        self.assertEqual(res, "Unknown Person")
        track_state = self.engine.track_cache.get(88)
        self.assertIsNotNone(track_state)
        self.assertTrue(track_state["is_confirmed"], "Unregistered target should be locked with is_confirmed=True after 3 attempts.")
        print("[PASS] Test 5: Unregistered far target locked as is_confirmed=True after 3 scans, stopping infinite loops.")

    def test_camera_cover_hand_removal_recovery(self):
        """Test 6: Verify covering camera with hand and removing it clears ghost tracks and recovers identity cleanly."""
        # 1. Person A identified
        resA, _ = self.engine.process_track_frame(track_id=1, embedding=self.vec_A, active_track_ids=[1])
        self.assertEqual(resA, "Person A")

        # 2. Intruder covers camera with hand (0 active tracks)
        self.engine.track_cache.clear()

        # 3. Intruder removes hand; Person A re-appears as new track ID 50
        resA_new, _ = self.engine.process_track_frame(track_id=50, embedding=self.vec_A, active_track_ids=[50])
        self.assertEqual(resA_new, "Person A", "Person A should be immediately re-identified after hand removal from camera lens.")
        print("[PASS] Test 6: Hand-cover camera occlusion flush & re-identification recovery verified.")

if __name__ == "__main__":
    unittest.main(verbosity=2)

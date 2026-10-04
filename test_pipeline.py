import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.inference.live_cctv as live_cctv
import src.database.enroll as enroll
import src.database.db_setup as db_setup
import src.utils.jitter_filter as jitter_filter
import src.utils.tune_tracker as tune_tracker
import src.utils.face_align as face_align
import src.utils.fsrcnn_upscaler as fsrcnn_upscaler

class TestCCTVFAISSPipeline(unittest.TestCase):
    
    def setUp(self):
        # Reset state dictionaries
        live_cctv.active_tracks = {}
        live_cctv.track_last_seen = {}
        live_cctv.track_votes = {}
        live_cctv.track_consensus_name = {}
        live_cctv.identified_cooldowns = {}
        
    def test_l2_normalize_unit_length(self):
        v = [3.0, 4.0]  # norm is 5.0
        norm_v = live_cctv.l2_normalize(v)
        expected = np.array([0.6, 0.8], dtype=np.float32)
        np.testing.assert_array_almost_equal(norm_v, expected, decimal=5)
        self.assertAlmostEqual(np.linalg.norm(norm_v), 1.0, places=5)
        
    def test_l2_normalize_zero_vector(self):
        v = [0.0, 0.0]
        norm_v = live_cctv.l2_normalize(v)
        np.testing.assert_array_almost_equal(norm_v, np.array([0.0, 0.0], dtype=np.float32))

    def test_query_vector_db_match_gated(self):
        mock_index = MagicMock()
        mock_index.ntotal = 2
        mock_index.search.return_value = (
            np.array([[0.2, 0.5]], dtype=np.float32), 
            np.array([[0, 1]], dtype=np.int64)
        )
        
        metadata = {"0": "Ceaser", "1": "Alice"}
        query_emb = [0.1] * 512
        
        matched_name, dist = live_cctv.query_vector_db(query_emb, mock_index, metadata)
        self.assertEqual(matched_name, "Ceaser")
        self.assertAlmostEqual(dist, 0.2, places=4)
        
    def test_query_vector_db_unknown_distance(self):
        mock_index = MagicMock()
        mock_index.ntotal = 1
        # Match 1: dist = 1.5 >= L2_MATCH_THRESHOLD (Rejected)
        mock_index.search.return_value = (
            np.array([[1.5, 2.0]], dtype=np.float32), 
            np.array([[0, -1]], dtype=np.int64)
        )
        
        metadata = {"0": "Ceaser"}
        query_emb = [0.1] * 512
        
        matched_name, dist = live_cctv.query_vector_db(query_emb, mock_index, metadata)
        self.assertEqual(matched_name, "Unknown")
        self.assertAlmostEqual(dist, 1.5, places=4)

    def test_bufferless_video_capture(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, "mock_frame")
        
        with patch('cv2.VideoCapture', return_value=mock_cap):
            bvc = live_cctv.BufferlessVideoCapture("test.mp4")
            self.assertTrue(bvc.isOpened())
            bvc.release()

    def test_jitter_filter_engine_cbkf_scaling(self):
        engine = jitter_filter.JitterFilterEngine()
        box1 = (10, 10, 50, 50)
        filtered1 = engine.filter_box(track_id=1, target_box=box1, confidence=0.9)
        self.assertEqual(filtered1, (10, 10, 50, 50))
        
        # Low confidence measurement (conf=0.1) should increase R and rely more on Kalman prediction
        box2 = (11, 11, 51, 51)
        filtered2 = engine.filter_box(track_id=1, target_box=box2, confidence=0.1)
        self.assertEqual(filtered2, (10, 10, 50, 50))

    def test_track_lifecycle_cleanup_60_frames(self):
        live_cctv.active_tracks[42] = "Ceaser"
        live_cctv.track_last_seen[42] = 100
        live_cctv.track_votes[42] = ["Ceaser"]
        live_cctv.track_consensus_name[42] = "Ceaser"
        
        # Frame 155 is within 60 frames (155 - 100 = 55 <= 60) -> Not expired!
        frame_count = 155
        expired_ids = [tid for tid, last_seen in live_cctv.track_last_seen.items() if frame_count - last_seen > 60]
        self.assertEqual(len(expired_ids), 0)
        
        # Frame 161 is >60 frames (161 - 100 = 61 > 60) -> Expired!
        frame_count = 161
        expired_ids = [tid for tid, last_seen in live_cctv.track_last_seen.items() if frame_count - last_seen > 60]
        self.assertEqual(len(expired_ids), 1)

    def test_umeyama_face_alignment(self):
        aligner = face_align.YuNetFaceAligner()
        mock_crop = np.zeros((100, 100, 3), dtype=np.uint8) + 128
        # Non-strict mode resizes pre-cropped faces
        aligned_non_strict = aligner.align_face(mock_crop, (112, 112), strict=False)
        self.assertEqual(aligned_non_strict.shape, (112, 112, 3))
        # Strict mode rejects non-face synthetic noise
        aligned_strict = aligner.align_face(mock_crop, (112, 112), strict=True)
        self.assertIsNone(aligned_strict)

    def test_fsrcnn_upscaler(self):
        upscaler = fsrcnn_upscaler.FSRCNNUpscaler(target_min_dim=112)
        small_crop = np.zeros((40, 40, 3), dtype=np.uint8) + 100
        upscaled = upscaler.upscale(small_crop)
        self.assertGreaterEqual(min(upscaled.shape[:2]), 112)

    def test_enrollment_name_parsing(self):
        name1 = enroll.extract_subject_name("known_faces/Ceaser.jpg")
        self.assertEqual(name1, "Ceaser")
        
        name2 = enroll.extract_subject_name("known_faces/frontal/Ceaser_1.jpg")
        self.assertEqual(name2, "Ceaser")
        
        name3 = enroll.extract_subject_name("known_faces/left_profile/John_Doe_left.png")
        self.assertEqual(name3, "John Doe")
        
        name4 = enroll.extract_subject_name("known_faces/Jane_Smith/left.jpg")
        self.assertEqual(name4, "Jane Smith")
        
        name5 = enroll.extract_subject_name("known_faces/Alice_Bob/frontal/frontal_2.jpg")
        self.assertEqual(name5, "Alice Bob")

if __name__ == '__main__':
    unittest.main()

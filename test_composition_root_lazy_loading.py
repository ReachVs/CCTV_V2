"""
Unit test suite verifying Candidate 6: Composition Root & Lazy Loading.
Validates zero-cost module import, dynamic PEP 562 attribute resolution,
lazy component initialization, setter overrides, and isolated teardown.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.inference.live_cctv as live_cctv
from src.inference.engine import BiometricTrackingEngine
from src.inference.person_detector import ScriptedPersonDetector, DetectedBox
from src.inference.face_embedder import ScriptedFaceEmbedder
from src.database.audit_repository import ScriptedAuditLogger
from src.database.vector_store import InMemoryIndexAdapter


class TestCompositionRootLazyLoading(unittest.TestCase):
    def setUp(self):
        live_cctv.reset_engine()

    def tearDown(self):
        live_cctv.reset_engine()

    def test_lazy_engine_singleton_lifecycle(self):
        """Verify _DEFAULT_ENGINE is None on start, created on first call, and reset cleanly."""
        self.assertIsNone(live_cctv._DEFAULT_ENGINE)
        
        # Access through get_engine()
        eng1 = live_cctv.get_engine()
        self.assertIsNotNone(eng1)
        self.assertIs(live_cctv._DEFAULT_ENGINE, eng1)
        
        # Access through engine property
        eng2 = live_cctv.engine
        self.assertIs(eng1, eng2)
        
        # Reset engine
        live_cctv.reset_engine()
        self.assertIsNone(live_cctv._DEFAULT_ENGINE)

    def test_module_pep562_attribute_resolution(self):
        """Verify dynamic module attributes resolve to the underlying engine."""
        eng = live_cctv.get_engine()
        
        self.assertIs(live_cctv.state_lock, eng.lock)
        self.assertIs(live_cctv.active_tracks, eng.active_tracks)
        self.assertIs(live_cctv.track_last_seen, eng.track_last_seen)
        self.assertIs(live_cctv.track_first_seen, eng.track_first_seen)
        self.assertIs(live_cctv.track_keypoints, eng.track_keypoints)
        self.assertIs(live_cctv.limb_cooldowns, eng.limb_cooldowns)
        self.assertIs(live_cctv.identified_cooldowns, eng.identified_cooldowns)
        self.assertIs(live_cctv.encrypted_audit_logger, eng.encrypted_audit_logger)
        self.assertIs(live_cctv.audit_logger, eng.encrypted_audit_logger)
        self.assertEqual(live_cctv.POSE_ONLY_MODE, eng.is_pose_only)

    def test_module_setattr_engine_override(self):
        """Verify custom engine injection via live_cctv.engine = custom_engine and set_engine."""
        custom_detector = ScriptedPersonDetector()
        custom_embedder = ScriptedFaceEmbedder()
        custom_audit = ScriptedAuditLogger()
        
        custom_engine = BiometricTrackingEngine(
            index_adapter=InMemoryIndexAdapter(np.zeros((1, 512), dtype=np.float32)),
            detector=custom_detector,
            embedder=custom_embedder,
            audit_logger=custom_audit
        )
        
        # Injected via set_engine
        live_cctv.set_engine(custom_engine)
        self.assertIs(live_cctv.engine, custom_engine)
        self.assertIs(live_cctv.get_engine(), custom_engine)
        
        # Injected via live_cctv.engine assignment
        another_engine = BiometricTrackingEngine(
            index_adapter=InMemoryIndexAdapter(np.zeros((1, 512), dtype=np.float32)),
            detector=custom_detector,
            embedder=custom_embedder,
            audit_logger=custom_audit
        )
        live_cctv.engine = another_engine
        self.assertIs(live_cctv.engine, another_engine)
        self.assertIs(live_cctv.get_engine(), another_engine)
        
        custom_engine.stop()
        another_engine.stop()

    def test_lazy_subcomponent_properties(self):
        """Verify detector, embedder, and pose_estimator initialize lazily and support setters."""
        engine = BiometricTrackingEngine(
            index_adapter=InMemoryIndexAdapter(np.zeros((1, 512), dtype=np.float32)),
            audit_logger=ScriptedAuditLogger()
        )
        
        # Verify internal backing fields are None before access
        self.assertIsNone(engine._detector)
        self.assertIsNone(engine._embedder)
        self.assertIsNone(engine._pose_estimator)
        
        # Test setter injection for detector
        test_detector = ScriptedPersonDetector()
        engine.detector = test_detector
        self.assertIs(engine.detector, test_detector)
        self.assertIs(engine._detector, test_detector)
        
        # Test setter injection for embedder
        test_embedder = ScriptedFaceEmbedder()
        engine.embedder = test_embedder
        self.assertIs(engine.embedder, test_embedder)
        self.assertIs(engine._embedder, test_embedder)
        
        # Test pose_estimator setter
        engine.pose_estimator = None
        self.assertIsNone(engine._pose_estimator)
        
        engine.stop()

    def test_state_mutation_through_module_delegates(self):
        """Verify legacy scripts modifying live_cctv.active_tracks update the engine state."""
        live_cctv.active_tracks[77] = "Candidate Six"
        self.assertIn(77, live_cctv.get_engine().active_tracks)
        self.assertEqual(live_cctv.get_engine().active_tracks[77], "Candidate Six")
        
        # Override active_tracks dictionary directly
        live_cctv.active_tracks = {88: "New Dict Track"}
        self.assertIn(88, live_cctv.get_engine().active_tracks)
        self.assertEqual(live_cctv.get_engine().active_tracks[88], "New Dict Track")

    def test_helpers_delegate_to_engine(self):
        """Verify helper functions (set_pose_only_mode, process_single_frame, etc.) route correctly."""
        custom_detector = ScriptedPersonDetector(detections=[
            DetectedBox(box=[10.0, 10.0, 100.0, 100.0], conf=0.95, track_id=1)
        ])
        custom_embedder = ScriptedFaceEmbedder()
        custom_engine = BiometricTrackingEngine(
            index_adapter=InMemoryIndexAdapter(np.zeros((1, 512), dtype=np.float32)),
            detector=custom_detector,
            embedder=custom_embedder,
            audit_logger=ScriptedAuditLogger()
        )
        live_cctv.set_engine(custom_engine)
        
        live_cctv.set_pose_only_mode(True)
        self.assertTrue(live_cctv.get_pose_only_mode())
        self.assertTrue(custom_engine.is_pose_only)
        
        live_cctv.set_pose_only_mode(False)
        self.assertFalse(live_cctv.get_pose_only_mode())
        
        dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        annotated = live_cctv.process_single_frame(dummy_frame)
        self.assertEqual(annotated.shape, dummy_frame.shape)
        
        custom_engine.stop()


if __name__ == "__main__":
    unittest.main()

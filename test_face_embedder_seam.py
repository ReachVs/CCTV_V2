"""
Unit tests for the FaceEmbedder seam and adapters (Candidate 3).
Verifies:
1. ScriptedFaceEmbedder provides fast, hermetic, model-free vector extraction.
2. Dynamic callable generators and L2 normalization in FaceEmbedder.
3. BiometricTrackingEngine delegates extract_face_embedding and flush() through the seam.
4. Backward compatibility with legacy engine._keras_model attribute mutations.
"""
import unittest
import numpy as np

from src.database.vector_store import InMemoryIndexAdapter
from src.inference.engine import BiometricTrackingEngine
from src.inference.face_embedder import (
    FaceEmbedder,
    ArcFaceEmbedder,
    ScriptedFaceEmbedder,
)
from src.inference.person_detector import ScriptedPersonDetector, DetectedBox


class TestFaceEmbedderSeam(unittest.TestCase):

    def setUp(self):
        self.adapter = InMemoryIndexAdapter(dimension=512)
        self.dummy_crop = np.full((112, 112, 3), 128, dtype=np.uint8)

    def test_scripted_embedder_unit_normalization(self):
        """Verify ScriptedFaceEmbedder correctly normalizes unnormalized vectors to unit L2 norm."""
        raw_vec = np.zeros(512, dtype=np.float32)
        raw_vec[0] = 3.0
        raw_vec[1] = 4.0  # norm = 5.0

        embedder = ScriptedFaceEmbedder(vector=raw_vec)
        result = embedder.embed(self.dummy_crop)

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)
        self.assertAlmostEqual(float(result[0]), 0.6, places=5)
        self.assertAlmostEqual(float(result[1]), 0.8, places=5)
        self.assertEqual(embedder.call_count, 1)

    def test_scripted_embedder_handles_empty_inputs(self):
        """Verify embedder safely returns None on empty or None crops."""
        embedder = ScriptedFaceEmbedder(vector=np.ones(512))
        self.assertIsNone(embedder.embed(None))
        self.assertIsNone(embedder.embed(np.array([])))

    def test_engine_injection_and_delegation(self):
        """Verify BiometricTrackingEngine extracts embeddings and flushes through the injected seam."""
        target_vec = np.zeros(512, dtype=np.float32)
        target_vec[42] = 1.0

        scripted_embedder = ScriptedFaceEmbedder(vector=target_vec)
        scripted_detector = ScriptedPersonDetector([
            DetectedBox(box=[50.0, 50.0, 150.0, 250.0], conf=0.90, track_id=7)
        ])

        engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            detector=scripted_detector,
            embedder=scripted_embedder,
            metadata_path=None,
        )

        # 1. Direct call to extract_face_embedding
        emb = engine.extract_face_embedding(self.dummy_crop)
        self.assertIsNotNone(emb)
        self.assertEqual(emb[42], 1.0)
        self.assertEqual(scripted_embedder.call_count, 1)

        # 2. Queue and flush
        engine._add_crop_to_embedder(track_id=7, crop=self.dummy_crop, frame_num=1)
        engine.flush()
        # flush() calls embedder.embed()
        self.assertEqual(scripted_embedder.call_count, 2)

    def test_backward_compatibility_keras_model(self):
        """Verify engine._keras_model getter and setter proxy to embedder."""
        embedder = ScriptedFaceEmbedder(vector=np.zeros(512))
        engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            embedder=embedder,
            metadata_path=None,
        )

        dummy_model = object()
        engine._keras_model = dummy_model
        self.assertIs(engine._keras_model, dummy_model)
        self.assertIs(embedder.keras_model, dummy_model)


if __name__ == "__main__":
    unittest.main()

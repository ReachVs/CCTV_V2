import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2
import os
import shutil
import tempfile
from typing import Dict, Any, List, Optional

from src.services.biometric_enrollment_manager import (
    BiometricEnrollmentManager,
    CuratedFacePreview,
    EnrollmentResult
)


class DummyFaceAligner:
    def __init__(self, return_face: bool = True):
        self.return_face = return_face

    def align_face(self, img, output_size=(112, 112), strict=False):
        if self.return_face and img is not None and img.size > 0:
            return cv2.resize(img, output_size)
        return None


class DummyEmbedder:
    def extract_face_embedding(self, img_or_crop, align=True):
        if img_or_crop is not None and img_or_crop.size > 0:
            # Deterministic mock 512D unit vector
            vec = np.ones((512,), dtype=np.float32)
            return vec / np.linalg.norm(vec)
        return None


class DummyBiometricRepo:
    def __init__(self):
        self.subjects = {}

    def add_subject(self, name, embeddings, profile_data=None, trigger_reload=False):
        self.subjects[name] = {
            "embeddings": embeddings,
            "profile_data": profile_data
        }
        return len(self.subjects)


class DummyEngine:
    def __init__(self):
        self.reloaded = False
        self.ntotal = 17
        self.lock = MagicMock()
        self.lock.__enter__ = MagicMock(return_value=None)
        self.lock.__exit__ = MagicMock(return_value=None)

    def reload_index(self):
        self.reloaded = True
        self.ntotal += 1


class TestBiometricEnrollmentManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.face_aligner = DummyFaceAligner(return_face=True)
        self.embedder = DummyEmbedder()
        self.repo = DummyBiometricRepo()
        self.engine = DummyEngine()

        self.manager = BiometricEnrollmentManager(
            known_faces_dir=self.test_dir,
            face_aligner=self.face_aligner,
            embedder=self.embedder,
            biometric_repo=self.repo,
            engine=self.engine
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_preview_and_curate_with_valid_image(self):
        # Create a synthetic test image with varying pixels (sharpness)
        img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        raw_bytes = buf.tobytes()

        previews = self.manager.preview_and_curate([raw_bytes])
        self.assertEqual(len(previews), 1)
        self.assertTrue(isinstance(previews[0], CuratedFacePreview))
        self.assertGreater(previews[0].sharpness_score, 0.0)
        self.assertTrue(len(previews[0].image_b64) > 0)
        self.assertTrue(previews[0].is_recommended)

    def test_preview_and_curate_no_face_detected(self):
        self.face_aligner.return_face = False
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        raw_bytes = buf.tobytes()

        previews = self.manager.preview_and_curate([raw_bytes])
        self.assertEqual(len(previews), 0)

    def test_commit_enrollment_multi_exemplar_and_storage(self):
        # Prepare 3 synthetic face crops
        crops = []
        for _ in range(3):
            crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", crop)
            crops.append(buf.tobytes())

        profile_data = {
            "occupation": "Cyber Security",
            "clearance": "Top Secret",
            "threat_level": "LOW"
        }

        result = self.manager.commit_enrollment(
            name="John Doe",
            profile_data=profile_data,
            approved_crops=crops
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.exemplar_count, 3)
        self.assertEqual(result.name, "John Doe")
        self.assertTrue(self.engine.reloaded)

        # Verify subject files were saved to disk in safe directory
        subject_dir = os.path.join(self.test_dir, "John_Doe")
        self.assertTrue(os.path.exists(subject_dir))
        saved_files = os.listdir(subject_dir)
        self.assertEqual(len(saved_files), 3)

        # Verify repo received all 3 multi-exemplar embeddings
        self.assertIn("John Doe", self.repo.subjects)
        enrolled_embs = self.repo.subjects["John Doe"]["embeddings"]
        self.assertEqual(len(enrolled_embs), 3)

    def test_commit_enrollment_path_traversal_protection(self):
        crop = np.ones((112, 112, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", crop)

        result = self.manager.commit_enrollment(
            name="../../etc/passwd",
            profile_data={},
            approved_crops=[buf.tobytes()]
        )
        self.assertEqual(result.status, "error")
        self.assertIn("Invalid", result.message)


if __name__ == "__main__":
    unittest.main()

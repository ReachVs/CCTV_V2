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
    def __init__(self, mode: str = "constant"):
        self.mode = mode
        self._call_count = 0

    def extract_face_embedding(self, img_or_crop, align=True):
        if img_or_crop is not None and img_or_crop.size > 0:
            self._call_count += 1
            if self.mode == "distinct":
                # Create distinct 512D vectors by setting different coordinates
                vec = np.zeros((512,), dtype=np.float32)
                vec[self._call_count % 512] = 1.0
                vec[0] = 0.5
                return vec / np.linalg.norm(vec)
            else:
                # Deterministic mock 512D unit vector
                vec = np.ones((512,), dtype=np.float32)
                return vec / np.linalg.norm(vec)
        return None


class DummyBiometricRepo:
    def __init__(self):
        self.subjects = {}

    def get_subject_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for n, data in self.subjects.items():
            if n.lower() == name.lower():
                return {"id": 1, "name": n}
        return None

    def get_subject_vectors(self, name: str) -> List[np.ndarray]:
        for n, data in self.subjects.items():
            if n.lower() == name.lower():
                return data.get("embeddings", [])
        return []

    def add_subject(self, name, embeddings, profile_data=None, trigger_reload=False):
        self.subjects[name] = {
            "embeddings": embeddings,
            "profile_data": profile_data
        }
        return len(self.subjects)


class DummyIndexAdapter:
    def __init__(self):
        self.ntotal = 0
        self._mock_dist = 999.0
        self._mock_idx = 0

    def search(self, query_vector, k=1):
        if self.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
        return np.array([[self._mock_dist]], dtype=np.float32), np.array([[self._mock_idx]], dtype=np.int64)


class DummyEngine:
    def __init__(self):
        self.reloaded = False
        self.reset_called = False
        self.index_adapter = DummyIndexAdapter()
        self.metadata = {}
        self.lock = MagicMock()
        self.lock.__enter__ = MagicMock(return_value=None)
        self.lock.__exit__ = MagicMock(return_value=None)

    @property
    def ntotal(self):
        return self.index_adapter.ntotal

    def reload_index(self):
        self.reloaded = True

    def reset_state(self):
        self.reset_called = True

    def get_identity(self, idx: int) -> str:
        return self.metadata.get(str(idx), "Unknown")


class DummyAuditLogger:
    def __init__(self):
        self.logged_events = []

    def log_event(self, timestamp, track_id, person_name, match_distance=0.0, extra_meta=None, **kwargs):
        self.logged_events.append({
            "timestamp": timestamp,
            "track_id": track_id,
            "person_name": person_name,
            "match_distance": match_distance,
            "extra_meta": extra_meta or {}
        })


class TestBiometricEnrollmentManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.face_aligner = DummyFaceAligner(return_face=True)
        self.embedder = DummyEmbedder()
        self.repo = DummyBiometricRepo()
        self.engine = DummyEngine()
        self.audit_logger = DummyAuditLogger()

        self.manager = BiometricEnrollmentManager(
            known_faces_dir=self.test_dir,
            face_aligner=self.face_aligner,
            embedder=self.embedder,
            biometric_repo=self.repo,
            engine=self.engine,
            audit_logger=self.audit_logger,
            max_exemplars=6
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_preview_and_curate_with_valid_image(self):
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
        self.assertTrue(self.engine.reset_called)

        # Verify subject files were saved to disk in safe directory
        subject_dir = os.path.join(self.test_dir, "John_Doe")
        self.assertTrue(os.path.exists(subject_dir))
        saved_files = os.listdir(subject_dir)
        self.assertEqual(len(saved_files), 3)

        # Verify repo received all 3 multi-exemplar embeddings
        self.assertIn("John Doe", self.repo.subjects)
        enrolled_embs = self.repo.subjects["John Doe"]["embeddings"]
        self.assertEqual(len(enrolled_embs), 3)

        # Verify audit log recorded commitment
        committed_events = [
            e for e in self.audit_logger.logged_events
            if e["extra_meta"].get("event_type") == "BIOMETRIC_ENROLLMENT_COMMITTED"
        ]
        self.assertEqual(len(committed_events), 1)
        self.assertEqual(committed_events[0]["person_name"], "John Doe")

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

    def test_commit_enrollment_angular_diversity_cap(self):
        # Embedder generates distinct vectors
        distinct_embedder = DummyEmbedder(mode="distinct")
        manager = BiometricEnrollmentManager(
            known_faces_dir=self.test_dir,
            face_aligner=self.face_aligner,
            embedder=distinct_embedder,
            biometric_repo=self.repo,
            engine=self.engine,
            audit_logger=self.audit_logger,
            max_exemplars=6
        )

        # Upload 10 distinct crops
        crops = []
        for i in range(10):
            crop = np.random.randint(10, 200, (112, 112, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", crop)
            crops.append(buf.tobytes())

        result = manager.commit_enrollment(
            name="Alice Smith",
            profile_data={},
            approved_crops=crops
        )
        self.assertEqual(result.status, "success")
        self.assertEqual(result.exemplar_count, 6)
        self.assertEqual(len(self.repo.subjects["Alice Smith"]["embeddings"]), 6)

    def test_commit_enrollment_smart_merging_existing_subject(self):
        # Pre-populate existing subject with 3 vectors
        existing_v1 = np.zeros((512,), dtype=np.float32)
        existing_v1[1] = 1.0
        existing_v2 = np.zeros((512,), dtype=np.float32)
        existing_v2[2] = 1.0

        self.repo.add_subject("Agent Cooper", [existing_v1, existing_v2])

        distinct_embedder = DummyEmbedder(mode="distinct")
        manager = BiometricEnrollmentManager(
            known_faces_dir=self.test_dir,
            face_aligner=self.face_aligner,
            embedder=distinct_embedder,
            biometric_repo=self.repo,
            engine=self.engine,
            audit_logger=self.audit_logger,
            max_exemplars=6
        )

        crops = []
        for _ in range(2):
            crop = np.random.randint(10, 200, (112, 112, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", crop)
            crops.append(buf.tobytes())

        result = manager.commit_enrollment(
            name="Agent Cooper",
            profile_data={},
            approved_crops=crops
        )
        self.assertEqual(result.status, "success")
        # 2 existing + 2 new = 4 total (< 6 cap)
        self.assertEqual(result.exemplar_count, 4)

    def test_commit_enrollment_biometric_collision_security_block(self):
        # Setup engine index to simulate existing subject "Target Suspect"
        self.engine.index_adapter.ntotal = 5
        self.engine.index_adapter._mock_dist = 0.20  # squared L2 < 0.36 = High match
        self.engine.index_adapter._mock_idx = 42
        self.engine.metadata = {"42": "Target Suspect"}

        crop = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", crop)

        result = self.manager.commit_enrollment(
            name="Malicious Actor",
            profile_data={},
            approved_crops=[buf.tobytes()]
        )

        # Must trigger 409 conflict
        self.assertEqual(result.status, "conflict")
        self.assertEqual(result.conflicting_subject, "Target Suspect")
        self.assertIn("Biometric collision detected", result.message)

        # Subject must NOT be added to database
        self.assertNotIn("Malicious Actor", self.repo.subjects)

        # Collision must be logged to encrypted WAL audit log
        collision_events = [
            e for e in self.audit_logger.logged_events
            if e["extra_meta"].get("event_type") == "ALERT_BIOMETRIC_COLLISION"
        ]
        self.assertEqual(len(collision_events), 1)
        self.assertEqual(collision_events[0]["extra_meta"]["conflicting_subject"], "Target Suspect")


if __name__ == "__main__":
    unittest.main()

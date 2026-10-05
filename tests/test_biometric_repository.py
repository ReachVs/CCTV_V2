import os
import sys
import unittest
import tempfile
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.biometric_repository import BiometricRepository, _l2_normalize


class TestBiometricRepository(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_faces.db")
        self.index_path = os.path.join(self.temp_dir.name, "test_faces.index")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")
        self.repo = BiometricRepository(
            db_path=self.db_path,
            index_path=self.index_path,
            metadata_path=self.metadata_path
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_initialization(self):
        """Verify tables and columns are created cleanly."""
        subjects = self.repo.get_all_subjects()
        self.assertEqual(len(subjects), 0)

    def test_add_and_query_subject(self):
        """Verify enrolling a subject with multiple vectors populates SQLite, FAISS, and metadata."""
        v1 = np.random.randn(512).astype(np.float32)
        v2 = np.random.randn(512).astype(np.float32)
        embeddings = [v1, v2]

        profile = {
            "occupation": "CHIEF ENGINEER",
            "clearance": "LEVEL 3 TOP SECRET",
            "threat_level": "MINIMAL"
        }

        person_id = self.repo.add_subject(
            name="Alice Walker",
            embeddings=embeddings,
            profile_data=profile,
            poses=["frontal", "tilt_up"],
            trigger_reload=False
        )

        self.assertGreater(person_id, 0)

        # Query by name
        sub = self.repo.get_subject_by_name("alice walker")
        self.assertIsNotNone(sub)
        self.assertEqual(sub["name"], "Alice Walker")
        self.assertEqual(sub["occupation"], "CHIEF ENGINEER")
        self.assertEqual(sub["clearance"], "LEVEL 3 TOP SECRET")

        # Query all subjects
        all_subs = self.repo.get_all_subjects()
        self.assertEqual(len(all_subs), 1)
        self.assertEqual(all_subs[0]["vector_count"], 2)

        # Verify metadata.json was generated and has 2 vectors
        self.assertTrue(os.path.exists(self.metadata_path))
        with open(self.metadata_path) as f:
            meta = json.load(f)
        self.assertEqual(len(meta), 2)
        self.assertEqual(meta["0"]["name"], "Alice Walker")
        self.assertEqual(meta["1"]["name"], "Alice Walker")
        self.assertEqual(meta["0"]["pose"], "frontal")
        self.assertEqual(meta["1"]["pose"], "tilt_up")

    def test_multiple_subjects_and_removal(self):
        """Verify adding multiple subjects and removing one cleanly re-indexes FAISS and cache."""
        v_alice = [np.random.randn(512).astype(np.float32) for _ in range(3)]
        v_bob = [np.random.randn(512).astype(np.float32) for _ in range(2)]

        self.repo.add_subject("Alice", v_alice, trigger_reload=False)
        self.repo.add_subject("Bob", v_bob, trigger_reload=False)

        subs = self.repo.get_all_subjects()
        self.assertEqual(len(subs), 2)

        with open(self.metadata_path) as f:
            meta = json.load(f)
        self.assertEqual(len(meta), 5)  # 3 + 2 = 5 vectors

        # Remove Bob
        removed = self.repo.remove_subject("bob", trigger_reload=False)
        self.assertTrue(removed)

        subs_after = self.repo.get_all_subjects()
        self.assertEqual(len(subs_after), 1)
        self.assertEqual(subs_after[0]["name"], "Alice")
        self.assertEqual(subs_after[0]["vector_count"], 3)

        with open(self.metadata_path) as f:
            meta_after = json.load(f)
        self.assertEqual(len(meta_after), 3)
        self.assertEqual(list(meta_after.keys()), ["0", "1", "2"])
        for v in meta_after.values():
            self.assertEqual(v["name"], "Alice")


if __name__ == "__main__":
    unittest.main()

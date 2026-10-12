"""
Unit tests for the deep BiometricMatcher module.
Validates input verification, pose-aware L2 thresholds, inter-subject margin gating,
and metadata resolution.
"""
import unittest
import numpy as np

from src.database.vector_store import InMemoryIndexAdapter
from src.inference.biometric_matcher import BiometricMatcher
from src.inference.consensus import MatchCandidate


class TestBiometricMatcherSeam(unittest.TestCase):

    def setUp(self):
        # 2 distinct subject vectors: Alice (idx 0), Bob (idx 1)
        self.v_alice = np.zeros(512, dtype=np.float32)
        self.v_alice[0] = 1.0

        self.v_bob = np.zeros(512, dtype=np.float32)
        self.v_bob[1] = 1.0

        matrix = np.vstack([self.v_alice, self.v_bob])
        self.adapter = InMemoryIndexAdapter(matrix, dimension=512)
        self.metadata = {
            "0": {"name": "Alice"},
            "1": {"name": "Bob"}
        }
        self.matcher = BiometricMatcher(
            index_adapter=self.adapter,
            metadata=self.metadata,
            l2_threshold=0.68,
            margin_threshold=0.15
        )

    def test_none_or_empty_input(self):
        cand1 = self.matcher.match(None)
        self.assertEqual(cand1.name, "Unknown")
        self.assertEqual(cand1.dist1, 999.0)

        cand2 = self.matcher.match(np.array([1.0, 2.0]))  # wrong shape
        self.assertEqual(cand2.name, "Unknown")

    def test_face_center_ratio_bounds(self):
        q = np.copy(self.v_alice)
        cand_low = self.matcher.match(q, face_center_ratio=0.10)
        self.assertEqual(cand_low.name, "Ambiguous Match")

        cand_high = self.matcher.match(q, face_center_ratio=0.90)
        self.assertEqual(cand_high.name, "Ambiguous Match")

        cand_valid = self.matcher.match(q, face_center_ratio=0.50)
        self.assertEqual(cand_valid.name, "Alice")

    def test_low_norm_quality(self):
        q = self.v_alice * 0.1  # norm 0.1 < 0.3
        cand = self.matcher.match(q)
        self.assertEqual(cand.name, "Unknown")
        self.assertAlmostEqual(cand.norm, 0.1, places=2)

    def test_pose_aware_side_profile_threshold(self):
        # Create a vector at distance ~0.45 from Alice
        # Alice is [1, 0, ...], Bob is [0, 1, ...]
        # Query: [0.9, 0.1, ...] -> L2 dist from Alice < 0.68 but > 0.30
        q = np.copy(self.v_alice)
        q[0] = 0.85
        q[10] = 0.65
        q = q / np.linalg.norm(q)

        # Frontal: matches Alice (< 0.68)
        cand_frontal = self.matcher.match(q, is_side_profile=False)
        self.assertEqual(cand_frontal.name, "Alice")

        # Side profile: rejected (> 0.30)
        cand_side = self.matcher.match(q, is_side_profile=True)
        self.assertEqual(cand_side.name, "Unknown")

    def test_inter_subject_margin_gating(self):
        # Query equidistant between Alice and Bob: [0.707, 0.707, 0, ...]
        # L2 distances to Alice and Bob are nearly identical (< 0.15 margin)
        q = (self.v_alice + self.v_bob) / np.linalg.norm(self.v_alice + self.v_bob)
        cand = self.matcher.match(q)
        self.assertEqual(cand.name, "Ambiguous Match")

    def test_metadata_resolution(self):
        self.assertEqual(self.matcher.get_identity(0), "Alice")
        self.assertEqual(self.matcher.get_identity(1), "Bob")
        self.assertEqual(self.matcher.get_identity(99), "Unknown")


if __name__ == "__main__":
    unittest.main()

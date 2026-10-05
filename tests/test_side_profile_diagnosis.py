import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import numpy as np

from src.database.vector_store import InMemoryIndexAdapter, FAISSIndexAdapter
from src.inference.engine import BiometricTrackingEngine
import faiss

class TestSideProfileDiagnosis(unittest.TestCase):
    def setUp(self):
        self.dim = 512
        np.random.seed(42)

    def _normalize(self, v):
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def test_faiss_index_adapter_normalizes_inner_product_to_l2(self):
        """
        Phase 1 Red-Test:
        When an index uses METRIC_INNER_PRODUCT (e.g. from IndexFlatIP),
        FAISSIndexAdapter MUST convert the inner product to Euclidean L2 distance.
        Otherwise, an identical match (IP=1.0) is treated as dist=1.0 > 0.68 (Unknown),
        while a poor match (IP=0.50) is treated as dist=0.50 < 0.68 (Accepted match).
        """
        # Create an IndexFlatIP with 1 normalized vector
        vec = self._normalize(np.random.randn(self.dim))
        ip_index = faiss.IndexFlatIP(self.dim)
        ip_index.add(np.expand_dims(vec, axis=0))

        adapter = FAISSIndexAdapter(ip_index)
        # Search with the exact same vector
        dists, idxs = adapter.search(vec, k=1)
        
        # In L2 space, distance to self must be 0.0, NOT 1.0!
        self.assertAlmostEqual(
            float(dists[0][0]), 0.0, places=3,
            msg=f"FAISSIndexAdapter failed to convert inner product to L2! Got {dists[0][0]} instead of ~0.0"
        )

    def test_side_profile_guardrail_prevents_false_lock_in(self):
        """
        Phase 1 Red-Test:
        Side-profile query at L2 distance 0.60 (which is < default 0.68 threshold):
        1. When pose is indicated as 'profile' / steep side angle, the engine must enforce
           a stricter cutoff (e.g. 0.52), rejecting a loose 0.60 match.
        2. Even if distance is close, a track must NOT lock 'is_confirmed=True' solely
           from side-profile frames without at least one frontal/quarter confirmation.
        """
        # Subject 1 (Ceaser)
        v_ceaser = self._normalize(np.random.randn(self.dim))
        # Subject 2 (Kouch Liza)
        v_liza = self._normalize(np.random.randn(self.dim))
        v_liza = self._normalize(v_liza - np.dot(v_ceaser, v_liza) * v_ceaser)

        matrix = np.vstack([v_ceaser, v_liza])
        adapter = InMemoryIndexAdapter(matrix, dimension=self.dim)
        engine = BiometricTrackingEngine(index_adapter=adapter)
        engine.metadata = {
            "0": {"name": "Ceaser", "pose": "frontal"},
            "1": {"name": "Kouch Liza", "pose": "side_left_angle"}
        }

        # Query that is distance 0.60 from Liza's template
        # (loose match that passes default 0.68 L2 threshold)
        diff = self._normalize(np.random.randn(self.dim))
        diff = self._normalize(diff - np.dot(v_liza, diff) * v_liza)
        # For unit vectors: ||v - q||^2 = 2 - 2*(v.q) = 0.60^2 = 0.36 => v.q = 1 - 0.18 = 0.82
        cos_theta = 1.0 - (0.60 ** 2) / 2.0
        sin_theta = np.sqrt(1.0 - cos_theta ** 2)
        side_query = self._normalize(cos_theta * v_liza + sin_theta * diff)

        actual_dist = float(np.linalg.norm(side_query - v_liza))
        self.assertAlmostEqual(actual_dist, 0.60, places=2)

        # verify_face with is_side_profile=True should reject distance 0.60 (stricter 0.52 cutoff)
        matched_name, d1, d2, q = engine.verify_face(side_query, is_side_profile=True)
        self.assertEqual(
            matched_name, "Unknown",
            f"Expected 'Unknown' under side profile strict cutoff (<0.52), got '{matched_name}' with dist {d1:.3f}"
        )

        # Test 3: Even if side profile matches with close distance (d1 < 0.52),
        # side-profile votes alone must NOT lock 'is_confirmed = True' without frontal corroboration!
        cos_theta_close = 1.0 - (0.45 ** 2) / 2.0
        sin_theta_close = np.sqrt(1.0 - cos_theta_close ** 2)
        close_side_query = self._normalize(cos_theta_close * v_liza + sin_theta_close * diff)

        track_id = 99
        res1, info1 = engine.process_track_frame(track_id, close_side_query, track_age=3, is_side_profile=True)
        res2, info2 = engine.process_track_frame(track_id, close_side_query, track_age=4, is_side_profile=True)
        track = engine.track_cache.get(track_id, {})
        self.assertFalse(
            track.get("is_confirmed", False),
            "Side profile votes alone must NEVER lock track as confirmed!"
        )

if __name__ == "__main__":
    unittest.main()

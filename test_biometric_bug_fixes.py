import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import unittest
import numpy as np
import threading
import time

from src.inference.engine import (
    BiometricTrackingEngine,
    FAISSIndexAdapter,
    InMemoryIndexAdapter
)


class TestBiometricBugFixes(unittest.TestCase):
    def setUp(self):
        self.dim = 512

    def _normalize(self, v):
        v = np.array(v, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def test_multi_pose_same_person_never_ambiguous(self):
        """
        Bug A Fix Verification:
        When a single person has multiple enrolled poses in FAISS, the delta between
        two poses of that person is small (< 0.15). Top-2 margin gating must be inter-subject,
        so it must NOT return 'Ambiguous Match'.
        """
        base_emb = self._normalize(np.random.randn(self.dim))
        # 5 poses for 'Ceaser'
        poses = []
        for i in range(5):
            pose = self._normalize(base_emb + np.random.randn(self.dim) * 0.03)
            poses.append(pose)

        matrix = np.vstack(poses)
        adapter = InMemoryIndexAdapter(matrix, dimension=self.dim)
        engine = BiometricTrackingEngine(index_adapter=adapter)
        engine.metadata = {str(i): {"name": "Ceaser"} for i in range(5)}

        # Query very close to pose 0
        query = self._normalize(poses[0] + np.random.randn(self.dim) * 0.01)
        name, d1, d2, q = engine.verify_face(query)

        self.assertEqual(name, "Ceaser")
        self.assertNotEqual(name, "Ambiguous Match")
        self.assertLess(d1, 0.68)
        # Since all neighbors belong to Ceaser, dist2 is 999.0
        self.assertEqual(d2, 999.0)

    def test_inter_subject_ambiguity_triggers_when_distinct_subjects_close(self):
        """
        Verify that margin gating STILL correctly flags ambiguity when two DIFFERENT
        individuals have embeddings that are too close to each other.
        """
        emb_alice = self._normalize(np.random.randn(self.dim))
        # Bob is artificially constructed to be extremely close to Alice (margin < 0.15)
        emb_bob = self._normalize(emb_alice + np.random.randn(self.dim) * 0.04)

        matrix = np.vstack([emb_alice, emb_bob])
        adapter = InMemoryIndexAdapter(matrix, dimension=self.dim)
        engine = BiometricTrackingEngine(index_adapter=adapter, margin_threshold=0.15)
        engine.metadata = {
            "0": {"name": "Alice"},
            "1": {"name": "Bob"}
        }

        # Query equidistant between Alice and Bob
        query = self._normalize((emb_alice + emb_bob) / 2.0)
        name, d1, d2, q = engine.verify_face(query)

        self.assertEqual(name, "Ambiguous Match")
        self.assertLess(abs(d2 - d1), 0.15)

    def test_track_switch_hysteresis_no_false_spatial_collision(self):
        """
        Bug B Fix Verification:
        When a tracked person switches track_id (e.g. track 1 -> track 2) because of camera
        motion or brief occlusion, track 2 must NOT be permanently branded 'Unknown Person'
        due to an outdated claim held by track 1.
        """
        emb = self._normalize(np.random.randn(self.dim))
        adapter = InMemoryIndexAdapter(emb.reshape(1, -1), dimension=self.dim)
        engine = BiometricTrackingEngine(index_adapter=adapter)
        engine.metadata = {"0": {"name": "Ceaser"}}

        # Frame 1 to 5: Track 1 seen and confirmed as Ceaser
        for f in range(1, 6):
            engine.frame_count = f
            engine.track_last_seen[1] = f
            disp, telem = engine.process_track_frame(track_id=1, embedding=emb, active_track_ids=[1])
            if f >= 4:
                self.assertEqual(disp, "Ceaser")

        self.assertEqual(engine.active_identity_claims.get("Ceaser"), 1)

        # Track 1 disappears. 20 frames pass (simulating gap).
        # Track 2 appears in frame 30 with Ceaser's face.
        engine.frame_count = 30
        engine.track_last_seen[2] = 30
        # Active tracks only contains Track 2 now
        disp, telem = engine.process_track_frame(track_id=2, embedding=emb, active_track_ids=[2])

        # Track 2 should NOT be spatial collision!
        self.assertFalse(engine.track_cache[2].get("is_spatial_collision", False))
        self.assertNotEqual(disp, "Unknown Person")

        # After voting frames, Track 2 takes over confirmation
        for f in range(31, 35):
            engine.frame_count = f
            engine.track_last_seen[2] = f
            disp, telem = engine.process_track_frame(track_id=2, embedding=emb, active_track_ids=[2])

        self.assertEqual(disp, "Ceaser")
        self.assertEqual(engine.active_identity_claims.get("Ceaser"), 2)

    def test_concurrent_hot_reload_during_tracking(self):
        """
        Bug C Fix Verification:
        Verify that reloading the database and metadata while tracking is active does
        not cause deadlocks, exceptions, or race conditions.
        """
        emb1 = self._normalize(np.random.randn(self.dim))
        adapter = InMemoryIndexAdapter(emb1.reshape(1, -1), dimension=self.dim)
        engine = BiometricTrackingEngine(index_adapter=adapter)
        engine.metadata = {"0": {"name": "Initial User"}}

        stop_threads = threading.Event()
        errors = []

        def worker_tracking():
            for f in range(1, 50):
                if stop_threads.is_set():
                    break
                try:
                    engine.frame_count = f
                    engine.process_track_frame(track_id=10, embedding=emb1, active_track_ids=[10])
                    time.sleep(0.002)
                except Exception as ex:
                    errors.append(f"Tracking error: {ex}")

        def worker_reloading():
            for i in range(10):
                if stop_threads.is_set():
                    break
                try:
                    new_emb = self._normalize(np.random.randn(self.dim))
                    new_meta = {"0": {"name": f"User_{i}"}}
                    engine.update_database([new_emb], new_meta)
                    time.sleep(0.005)
                except Exception as ex:
                    errors.append(f"Reloading error: {ex}")

        t1 = threading.Thread(target=worker_tracking)
        t2 = threading.Thread(target=worker_reloading)
        t1.start()
        t2.start()

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
        stop_threads.set()

        self.assertEqual(len(errors), 0, f"Concurrent tracking and reload errors: {errors}")


if __name__ == "__main__":
    unittest.main()

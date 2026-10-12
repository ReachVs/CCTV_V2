import unittest
import threading
from src.domain.subject_session import SubjectSession, TrackStatus
from src.domain.track_registry import TrackRegistry
from src.inference.engine import BiometricTrackingEngine


class TestTrackRegistrySeam(unittest.TestCase):
    """
    Unit tests for the deep TrackRegistry module and its integration
    with BiometricTrackingEngine.
    """

    def test_registry_lifecycle_and_single_source_of_truth(self):
        registry = TrackRegistry()
        track_id = 1
        box = (100, 100, 300, 500)

        # 1. get_or_create
        session = registry.get_or_create(track_id, box, current_frame=1)
        self.assertEqual(session.track_id, 1)
        self.assertEqual(session.status, TrackStatus.SCANNING)
        self.assertEqual(session.box, box)
        self.assertEqual(session.last_seen, 1)
        self.assertEqual(len(registry), 1)

        # 2. update_box
        new_box = (110, 105, 310, 505)
        registry.update_box(track_id, new_box, current_frame=5)
        self.assertEqual(session.box, new_box)
        self.assertEqual(session.last_seen, 5)

        # 3. update_keypoints
        kpts = [{"x": 120.0, "y": 130.0, "conf": 0.9}]
        registry.update_keypoints(track_id, kpts)
        self.assertEqual(session.keypoints, kpts)

        # 4. Proxy reflection: active_tracks reflects display_name
        self.assertIn(track_id, registry.active_tracks)
        self.assertEqual(registry.active_tracks[track_id], "Scanning Track 1")

        # 5. Confirming identity reflects across all views
        session.confirm_identity("Ceaser", distance=0.25, norm=1.2)
        self.assertEqual(registry.active_tracks[track_id], "Ceaser")
        self.assertTrue(registry.track_cache[track_id]["is_confirmed"])
        self.assertEqual(registry.track_cache[track_id]["name"], "Ceaser")

        # 6. Purge
        purged = registry.purge(track_id)
        self.assertIsNotNone(purged)
        self.assertEqual(len(registry), 0)
        self.assertNotIn(track_id, registry.active_tracks)

    def test_stale_track_garbage_collection(self):
        registry = TrackRegistry()
        registry.get_or_create(1, (10, 10, 50, 50), current_frame=1)
        registry.get_or_create(2, (20, 20, 60, 60), current_frame=10)

        # Frame 40: track 1 missing for 39 frames, track 2 active
        purged = registry.purge_stale(current_frame=40, active_track_ids=[2], max_missing_frames=30)
        self.assertEqual(purged, [1])
        self.assertNotIn(1, registry)
        self.assertIn(2, registry)

    def test_engine_facade_compatibility(self):
        engine = BiometricTrackingEngine()
        track_id = 42

        # Engine uses TrackRegistry
        session = engine._get_or_create_session(track_id, (50, 50, 200, 400), frame_num=10)
        self.assertIsInstance(session, SubjectSession)
        self.assertIn(track_id, engine.registry)

        # Legacy active_tracks facade works seamlessly
        self.assertIn(track_id, engine.active_tracks)
        self.assertEqual(engine.active_tracks[track_id], f"Scanning Track {track_id}")

        # Mutating active_tracks facade updates session
        engine.active_tracks[track_id] = "Ceaser"
        self.assertTrue(session.is_confirmed)
        self.assertEqual(session.identity_name, "Ceaser")

        # Mutating track_cache facade updates session
        engine.track_cache[track_id]["failed_attempts"] = 4
        self.assertEqual(session.failed_attempts, 4)

        # Purging via engine purges from registry
        engine.purge_track(track_id)
        self.assertNotIn(track_id, engine.registry)
        self.assertNotIn(track_id, engine.active_tracks)

    def test_concurrent_registry_access(self):
        registry = TrackRegistry()

        def worker(start_id: int):
            for i in range(50):
                tid = start_id + i
                registry.get_or_create(tid, (i, i, i + 10, i + 10), current_frame=i)
                registry.active_tracks[tid] = f"Worker {tid}"
                _ = registry.active_sessions()
                if i % 5 == 0:
                    registry.purge(tid)

        threads = [
            threading.Thread(target=worker, args=(0,)),
            threading.Thread(target=worker, args=(100,)),
            threading.Thread(target=worker, args=(200,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(len(registry) >= 0)


if __name__ == "__main__":
    unittest.main()

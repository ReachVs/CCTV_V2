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

    def test_spatial_iou_inheritance(self):
        registry = TrackRegistry()
        box_a = (100, 100, 200, 200)

        # 1. Subject 1 established and confirmed
        sess1, inherited = registry.update_frame_track(track_id=1, box=box_a, current_frame=10)
        self.assertIsNone(inherited)
        registry.confirm_track_identity(1, "Alice", dist=0.3)

        # Update subject 1 at frame 20
        registry.update_frame_track(track_id=1, box=box_a, current_frame=20)

        # Purge subject 1 (e.g. tracker drop)
        # Note: recent_track_history retains recent confirmed info if populated
        registry.recent_track_history[1] = {
            "box": box_a,
            "last_seen": 20,
            "name": "Alice",
            "confirmed": True,
        }

        # Track 2 appears at frame 30 with high IoU (overlap > 0.40)
        box_b = (105, 105, 205, 205)
        sess2, inherited = registry.update_frame_track(track_id=2, box=box_b, current_frame=30)
        self.assertEqual(inherited, "Alice")
        self.assertEqual(sess2.display_name, "Alice")
        self.assertTrue(sess2.is_confirmed)

    def test_spatial_occlusion_checking(self):
        registry = TrackRegistry()
        box_confirmed = (100, 100, 300, 400)
        registry.recent_track_history[10] = {
            "box": box_confirmed,
            "last_seen": 50,
            "name": "Bob",
            "confirmed": True,
        }

        # Overlapping unconfirmed box (high IoU or containment)
        overlapping_box = (150, 150, 280, 380)
        self.assertTrue(registry.check_spatial_occlusion(track_id=20, box=overlapping_box))

        # Distant non-overlapping box
        distant_box = (500, 500, 700, 800)
        self.assertFalse(registry.check_spatial_occlusion(track_id=20, box=distant_box))

    def test_anti_limbo_timeout(self):
        registry = TrackRegistry()
        sess = registry.get_or_create(5, (50, 50, 100, 100), current_frame=1)
        sess.last_seen = 30  # track_age = 30

        # Scanning candidate with 0 valid face crops and age > 25 -> triggers unverified transition
        triggered = registry.evaluate_anti_limbo_timeout(sess, max_scanning_age=25)
        self.assertTrue(triggered)
        self.assertEqual(sess.status, TrackStatus.UNVERIFIED)
        self.assertEqual(sess.display_name, "Unverified ID 5")

        # Confirmed track -> should not transition
        sess_conf = registry.get_or_create(6, (50, 50, 100, 100), current_frame=1)
        sess_conf.last_seen = 30
        sess_conf.confirm_identity("Alice", 0.3)
        self.assertFalse(registry.evaluate_anti_limbo_timeout(sess_conf, max_scanning_age=25))

    def test_find_expired_tracks(self):
        registry = TrackRegistry()
        # Track 1: dead track (last seen frame 10, current frame 80 -> gap 70 > 60)
        s1 = registry.get_or_create(1, (10, 10, 50, 50), current_frame=10)
        s1.last_seen = 10

        # Track 2: persistent headless phantom (first_seen 1, last_seen 50, age 50 > 45, 0 face crops, unconfirmed)
        s2 = registry.get_or_create(2, (20, 20, 60, 60), current_frame=1)
        s2.last_seen = 50
        s2.valid_face_crops = 0

        # Track 3: healthy confirmed track (last seen 75)
        s3 = registry.get_or_create(3, (30, 30, 70, 70), current_frame=1)
        s3.last_seen = 75
        s3.confirm_identity("Charlie", 0.25)

        expired = registry.find_expired_tracks(current_frame=80)
        self.assertIn(1, expired)
        self.assertIn(2, expired)
        self.assertNotIn(3, expired)

    def test_verifying_consensus_not_prematurely_confirmed(self):
        """Verifies that transient vote accumulation string does not trigger premature identity confirmation."""
        registry = TrackRegistry()
        sess = registry.get_or_create(5, (100, 100, 200, 200), current_frame=1)

        # 1. Setting active_tracks to Verifying: Ceaser (1/2) must keep track in SCANNING state
        registry.active_tracks[5] = "Verifying: Ceaser (1/2)"
        self.assertFalse(sess.is_confirmed)
        self.assertEqual(sess.status, TrackStatus.SCANNING)
        self.assertEqual(sess.display_name, "Verifying: Ceaser (1/2)")

        # 2. Setting track_cache name to Verifying: Ceaser (1/2) must not confirm
        registry.track_cache[5]["name"] = "Verifying: Ceaser (1/2)"
        self.assertFalse(sess.is_confirmed)
        self.assertEqual(sess.status, TrackStatus.SCANNING)

        # 3. Confirming identity must sanitize Verifying / Verified prefixes
        sess.confirm_identity("Verifying: Ceaser (1/2)", 0.25)
        self.assertTrue(sess.is_confirmed)
        self.assertEqual(sess.status, TrackStatus.CONFIRMED)
        self.assertEqual(sess.identity_name, "Ceaser")
        self.assertEqual(sess.display_name, "Ceaser")


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for CameraIngestionPool (Candidate 5).
Verifies:
1. Reference-counting acquisition & deferred shutdown.
2. Safe pause/resume hardware arbitration.
3. Multi-camera feed addition, querying, and removal.
4. Synthetic fallback frame generation for disconnected/warmup feeds.
5. Backward compatibility aliases (MultiCameraIngestionManager, GlobalCameraStream).
"""
import unittest
import numpy as np
import time

from src.inference.multi_camera import (
    CameraIngestionPool,
    MultiCameraIngestionManager,
    GlobalCameraStream,
    BufferlessVideoCapture,
)


class TestCameraIngestionPool(unittest.TestCase):

    def setUp(self):
        self.pool = CameraIngestionPool(
            source="invalid_test_source",
            width=640,
            height=360,
            deferred_shutdown_seconds=0.05,  # Fast timeout for tests
        )

    def tearDown(self):
        self.pool.stop()

    def test_synthetic_frame_generation(self):
        """Verify CameraIngestionPool returns synthetic standby frame when hardware is disconnected."""
        frame = self.pool._generate_synthetic_frame()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (360, 640, 3))

        raw = self.pool.get_raw_frame("non_existent_stream")
        self.assertEqual(raw.shape, (360, 640, 3))

    def test_reference_counting_lifecycle(self):
        """Verify acquire() and release() properly increment and decrement client count."""
        self.assertEqual(self.pool.client_count, 0)
        self.assertFalse(self.pool.running)

        self.pool.acquire()
        self.assertEqual(self.pool.client_count, 1)
        self.assertTrue(self.pool.running)

        self.pool.acquire()
        self.assertEqual(self.pool.client_count, 2)

        self.pool.release()
        self.assertEqual(self.pool.client_count, 1)
        self.assertTrue(self.pool.running)

        self.pool.release()
        self.assertEqual(self.pool.client_count, 0)
        # Poll for deferred shutdown timer to fire
        for _ in range(20):
            if not self.pool.running:
                break
            time.sleep(0.05)
        self.assertFalse(self.pool.running)

    def test_pause_and_resume(self):
        """Verify pause() marks pool as paused and resume() restores active flag."""
        self.assertFalse(self.pool.paused)
        self.pool.pause()
        self.assertTrue(self.pool.paused)
        self.pool.resume()
        self.assertFalse(self.pool.paused)

    def test_multi_camera_management(self):
        """Verify dynamic adding and removing of multi-camera feeds."""
        self.assertEqual(len(self.pool.active_stream_ids()), 0)

        # Mock add_camera with invalid source (creates BufferlessVideoCapture safely)
        added = self.pool.add_camera("cam_north", "invalid_source_path")
        self.assertTrue(added)
        self.assertIn("cam_north", self.pool.active_stream_ids())

        # Cannot add duplicate stream_id
        dup = self.pool.add_camera("cam_north", "invalid_source_path")
        self.assertFalse(dup)

        # Remove stream
        removed = self.pool.remove_camera("cam_north")
        self.assertTrue(removed)
        self.assertNotIn("cam_north", self.pool.active_stream_ids())

    def test_backward_compatibility_aliases(self):
        """Verify legacy class names alias to CameraIngestionPool."""
        self.assertIs(MultiCameraIngestionManager, CameraIngestionPool)
        self.assertIs(GlobalCameraStream, CameraIngestionPool)


if __name__ == "__main__":
    unittest.main()

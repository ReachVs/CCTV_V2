import asyncio
import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.services.stream_broadcaster import StreamBroadcaster


class MockProcessResult:
    def __init__(self, frame):
        self.annotated_frame = frame


class MockEngine:
    def __init__(self):
        self.frame_count = 0
        self.is_pose_only = False
        self.processed = False

    def process_frame(self, frame, annotate=True):
        self.frame_count += 1
        self.processed = True
        return MockProcessResult(frame)

    def get_active_tracks(self):
        return []

    def set_pose_only(self, val):
        self.is_pose_only = val


class TestStreamBroadcaster(unittest.TestCase):
    """
    Unit tests for the deep StreamBroadcaster service.
    """

    def setUp(self):
        self.mock_engine = MockEngine()
        self.broadcaster = StreamBroadcaster(
            engine=self.mock_engine,
            default_source=0,
            width=320,
            height=240,
        )

    def tearDown(self):
        self.broadcaster.shutdown()

    def test_mode_transitions(self):
        """Verify mode transitions and stream pause/resume."""
        self.assertEqual(self.broadcaster.system_mode, "SURVEILLANCE")

        # Switch to ENROLLMENT
        mode = self.broadcaster.set_mode("ENROLLMENT", pose_only=True)
        self.assertEqual(mode, "ENROLLMENT")
        self.assertEqual(self.broadcaster.system_mode, "ENROLLMENT")
        self.assertTrue(self.mock_engine.is_pose_only)

        # Switch back to SURVEILLANCE
        mode = self.broadcaster.set_mode("SURVEILLANCE", pose_only=False)
        self.assertEqual(mode, "SURVEILLANCE")
        self.assertFalse(self.mock_engine.is_pose_only)

    def test_process_frame_orchestration(self):
        """Verify frame processing routes correctly based on system mode."""
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

        # Enrollment mode: passes through raw frame without engine processing
        self.broadcaster.system_mode = "ENROLLMENT"
        self.mock_engine.processed = False
        out = self.broadcaster.process_frame(dummy_frame)
        self.assertFalse(self.mock_engine.processed)
        self.assertEqual(out.shape, dummy_frame.shape)

        # Surveillance mode: executes engine inference
        self.broadcaster.system_mode = "SURVEILLANCE"
        out = self.broadcaster.process_frame(dummy_frame)
        self.assertTrue(self.mock_engine.processed)
        self.assertEqual(self.mock_engine.frame_count, 1)

    def test_raw_frame_base64_encoding(self):
        """Verify get_raw_frame_b64 returns valid data URI."""
        b64 = self.broadcaster.get_raw_frame_b64()
        self.assertIsNotNone(b64)
        self.assertTrue(b64.startswith("data:image/jpeg;base64,"))

    def test_mjpeg_frame_generator(self):
        """Verify mjpeg generator yields valid multipart/x-mixed-replace frame chunks."""
        gen = self.broadcaster.mjpeg_frame_generator(quality=70, delay=0.001)
        chunk = next(gen)
        self.assertIn(b"--frame", chunk)
        self.assertIn(b"Content-Type: image/jpeg", chunk)

    def test_camera_management_facade(self):
        """Verify camera stream adding and querying."""
        streams = self.broadcaster.active_stream_ids()
        self.assertIsInstance(streams, list)
        self.assertTrue(self.broadcaster.is_camera_ready())

    def test_identity_dispatch_filters_transient_tracks(self):
        """Verify unconfirmed or transient tracks are never dispatched as confirmed identities."""
        dispatched_events = []

        class MockDispatcher:
            def dispatch(self, event):
                dispatched_events.append(event)

        class MockTrack:
            def __init__(self, name, is_confirmed, conf):
                self.subject_name = name
                self.is_confirmed = is_confirmed
                self.match_confidence = conf
                self.track_id = 1

        self.mock_engine.identity_dispatcher = MockDispatcher()

        # 1. Track is accumulating votes / unconfirmed: must NOT dispatch
        self.mock_engine.get_active_tracks = lambda: [
            MockTrack("Verifying: Ceaser (1/2)", is_confirmed=False, conf=0.60)
        ]
        self.broadcaster.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(len(dispatched_events), 0)

        # 2. Track is confirmed: MUST dispatch with clean name
        self.mock_engine.get_active_tracks = lambda: [
            MockTrack("Verified: Ceaser", is_confirmed=True, conf=0.85)
        ]
        self.broadcaster.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertEqual(len(dispatched_events), 1)
        self.assertEqual(dispatched_events[0].person_name, "Ceaser")


if __name__ == "__main__":
    unittest.main()

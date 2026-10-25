"""
Hermetic Unit Tests for UnusualPersonService.
Verifies alert recording, snapshot persistence, dismissal, quick enrollment,
and retrieval of active alerts for the top pinned banner.
"""
import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

from src.domain.identity_observer import UnusualPersonDetectedEvent
from src.services.unusual_person_service import UnusualPersonService


class TestUnusualPersonService(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage_dir = os.path.join(self.test_dir, "unusual_snapshots")
        self.service = UnusualPersonService(storage_dir=self.storage_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_record_and_get_active_alerts(self):
        """Verifies recording an unusual person alert and querying active alerts."""
        event = UnusualPersonDetectedEvent(
            track_id=101,
            first_seen_frame=10,
            frame_num=25,
            timestamp_str="2026-10-22 10:15:00",
            snapshot_path=""
        )
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        # Draw some pixels
        dummy_crop[20:80, 20:80] = (255, 255, 255)

        alert = self.service.record_alert(
            track_id=event.track_id,
            snapshot_img=dummy_crop,
            timestamp_str=event.timestamp_str
        )
        self.assertEqual(alert["track_id"], 101)
        self.assertEqual(alert["status"], "ACTIVE")
        self.assertTrue(os.path.exists(alert["snapshot_path"]))

        active = self.service.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["track_id"], 101)
        self.assertEqual(active[0]["status"], "ACTIVE")

    def test_dismiss_alert(self):
        """Verifies dismissing an alert removes it from active list."""
        self.service.record_alert(track_id=202, timestamp_str="2026-10-22 10:20:00")
        self.assertEqual(len(self.service.get_active_alerts()), 1)

        success = self.service.dismiss_alert(202)
        self.assertTrue(success)
        self.assertEqual(len(self.service.get_active_alerts()), 0)

    def test_observer_seam_integration(self):
        """Verifies that on_unusual_person_detected automatically records an alert."""
        event = UnusualPersonDetectedEvent(
            track_id=303,
            first_seen_frame=1,
            frame_num=16,
            timestamp_str="2026-10-22 10:25:00",
            snapshot_path="",
            metadata={"valid_face_crops": 15}
        )
        self.service.on_unusual_person_detected(event)
        active = self.service.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["track_id"], 303)


if __name__ == "__main__":
    unittest.main()

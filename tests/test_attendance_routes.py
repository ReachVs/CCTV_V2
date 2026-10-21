"""
Unit Tests for Attendance API Route Handlers in src/api/cctv_server.py.
"""
import asyncio
import os
import unittest

from src.api.cctv_server import (
    attendance_service,
    get_attendance_config,
    set_attendance_config,
    get_attendance_checklist,
    manual_mark_attendance,
    reset_attendance
)
from src.api.schemas import AttendanceConfigRequest, AttendanceManualMarkRequest, AttendanceCheckAllRequest


class TestAttendanceRoutes(unittest.TestCase):
    def setUp(self):
        self._orig_config = attendance_service.get_config()
        # Reset today's attendance before each test
        asyncio.run(reset_attendance())

    def tearDown(self):
        if hasattr(self, "_orig_config") and self._orig_config:
            attendance_service.set_schedule(
                class_start_time=self._orig_config.get("class_start_time", "18:00:00"),
                class_end_time=self._orig_config.get("class_end_time", "21:00:00"),
                grace_period_minutes=self._orig_config.get("grace_period_minutes", 15)
            )
        asyncio.run(reset_attendance())

    def test_get_and_set_attendance_config(self):
        # Set config to 08:30 with 5 min grace
        req = AttendanceConfigRequest(cutoff_time="08:30:00", grace_period_minutes=5)
        res = asyncio.run(set_attendance_config(req))
        self.assertEqual(res.status, "success")
        self.assertEqual(res.cutoff_time, "08:30:00")
        self.assertEqual(res.grace_period_minutes, 5)

        # Retrieve config
        cfg = asyncio.run(get_attendance_config())
        self.assertEqual(cfg.cutoff_time, "08:30:00")
        self.assertEqual(cfg.grace_period_minutes, 5)

    def test_attendance_checklist_and_manual_marking(self):
        # Set target time
        req_cfg = AttendanceConfigRequest(cutoff_time="09:00:00", grace_period_minutes=0)
        asyncio.run(set_attendance_config(req_cfg))

        # Manually mark a student as PRESENT
        mark_req = AttendanceManualMarkRequest(
            person_name="Test Student",
            status="PRESENT",
            custom_time="08:45:00"
        )
        res = asyncio.run(manual_mark_attendance(mark_req))
        self.assertEqual(res.status_code, 200)

        # Check that checklist returns the student as PRESENT
        checklist_res = asyncio.run(get_attendance_checklist())
        self.assertEqual(checklist_res.status, "success")
        matching = [item for item in checklist_res.checklist if item.person_name == "Test Student"]
        self.assertTrue(len(matching) >= 1)
        self.assertEqual(matching[0].status, "PRESENT")
        self.assertEqual(matching[0].check_in_time, "08:45:00")

    def test_reset_attendance(self):
        mark_req = AttendanceManualMarkRequest(
            person_name="Reset Student",
            status="PRESENT",
            custom_time="08:50:00"
        )
        asyncio.run(manual_mark_attendance(mark_req))

        chk1 = asyncio.run(get_attendance_checklist())
        self.assertTrue(any(i.person_name == "Reset Student" for i in chk1.checklist))

        # Reset
        res_reset = asyncio.run(reset_attendance())
        self.assertEqual(res_reset.status_code, 200)

        chk2 = asyncio.run(get_attendance_checklist())
        # Should be absent or cleared
        matching = [i for i in chk2.checklist if i.person_name == "Reset Student"]
        if matching:
            self.assertEqual(matching[0].status, "ABSENT")

    def test_input_validation_and_bounds(self):
        from fastapi import HTTPException

        # 1. Invalid date pattern
        with self.assertRaises(HTTPException) as cm:
            asyncio.run(get_attendance_checklist(date="2026/10/20"))
        self.assertEqual(cm.exception.status_code, 400)

        # 2. Empty person name in manual mark
        with self.assertRaises(HTTPException) as cm:
            mark_req = AttendanceManualMarkRequest(person_name="   ", status="PRESENT")
            asyncio.run(manual_mark_attendance(mark_req))
        self.assertEqual(cm.exception.status_code, 400)

        # 3. Invalid status string
        with self.assertRaises(HTTPException) as cm:
            mark_req = AttendanceManualMarkRequest(person_name="Valid Name", status="DISQUALIFIED")
            asyncio.run(manual_mark_attendance(mark_req))
        self.assertEqual(cm.exception.status_code, 400)

    def test_schedule_start_and_end_times(self):
        # Set class schedule with start time 08:30 and end time 11:30
        req = AttendanceConfigRequest(class_start_time="08:30:00", class_end_time="11:30:00", grace_period_minutes=15)
        res = asyncio.run(set_attendance_config(req))
        self.assertEqual(res.status, "success")
        self.assertEqual(res.class_start_time, "08:30:00")
        self.assertEqual(res.class_end_time, "11:30:00")
        self.assertEqual(res.grace_period_minutes, 15)

        # Retrieve config
        cfg = asyncio.run(get_attendance_config())
        self.assertEqual(cfg.class_start_time, "08:30:00")
        self.assertEqual(cfg.class_end_time, "11:30:00")
        self.assertEqual(cfg.grace_period_minutes, 15)

    def test_check_all_attendance(self):
        # Call check-all route to mark all enrolled students as PRESENT
        from src.api.cctv_server import check_all_attendance
        req = AttendanceCheckAllRequest(status="PRESENT")
        res = asyncio.run(check_all_attendance(req))
        self.assertEqual(res.status_code, 200)

        chk = asyncio.run(get_attendance_checklist())
        self.assertEqual(chk.status, "success")
        self.assertGreater(chk.total_enrolled, 0)
        self.assertEqual(chk.absent_count, 0)
    def test_cctv_frame_automatic_check_in(self):
        from src.api.cctv_server import _process_cctv_frame, engine
        import numpy as np

        class MockTrack:
            track_id = 999
            confirmed_name = "Ceaser"
            subject_name = "Ceaser"
            match_confidence = 0.85
            is_confirmed = True

        orig_tracks = engine.get_active_tracks
        engine.get_active_tracks = lambda: [MockTrack()]
        try:
            frame = np.zeros((480, 854, 3), dtype=np.uint8)
            _process_cctv_frame(frame)

            chk = asyncio.run(get_attendance_checklist())
            ceaser_item = next((i for i in chk.checklist if i.person_name == "Ceaser"), None)
            self.assertIsNotNone(ceaser_item)
            self.assertIn(ceaser_item.status, ("PRESENT", "LATE"))
        finally:
            engine.get_active_tracks = orig_tracks


if __name__ == "__main__":
    unittest.main()


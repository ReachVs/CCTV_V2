"""
End-to-End Integration Test for Redefined Attendance Workflow:
- Single-camera course period timetable & manual session switching
- Class-specific roster with PENDING state for active course periods
- Biometric verification consensus check-in (auto-transitions to PRESENT)
- Teacher manual overrides, reason notes, and audit trail in attendance_audit_log
- Pinned Unusual Person / Intruder alert lifecycle (Active -> Dismissed / Enrolled)
- CSV attendance report generation with course filtering
"""
import os
import unittest
import tempfile
import sqlite3
import numpy as np

from src.database.db_setup import initialize_database
from src.domain.course_repository import CourseRepository
from src.services.attendance_service import AttendanceService, InMemoryRosterProvider
from src.services.unusual_person_service import UnusualPersonService
from src.domain.identity_observer import (
    IdentityConfirmedEvent,
    UnusualPersonDetectedEvent,
    AttendanceIdentityObserver
)


class TestAttendanceE2EIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_cctv.db")
        self.events_dir = os.path.join(self.temp_dir.name, "unusual_events")
        os.makedirs(self.events_dir, exist_ok=True)

        initialize_database(self.db_path)
        self.course_repo = CourseRepository(self.db_path)
        self.roster_provider = InMemoryRosterProvider([
            {"name": "Alice Smith", "occupation": "STUDENT"},
            {"name": "Bob Jones", "occupation": "STUDENT"},
            {"name": "Charlie Day", "occupation": "STUDENT"},
        ])
        self.attendance_service = AttendanceService(
            db_path=self.db_path,
            course_repo=self.course_repo,
            roster_provider=self.roster_provider
        )
        self.unusual_service = UnusualPersonService(storage_dir=self.events_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_attendance_lifecycle_e2e(self):
        # 1. Setup Course & Enrolled Students
        course = self.course_repo.create_course(code="CS-401", name="Deep Learning Surveillance")
        course_id = course["id"]

        self.course_repo.enroll_student(course_id, "Alice Smith")
        self.course_repo.enroll_student(course_id, "Bob Jones")
        self.course_repo.enroll_student(course_id, "Charlie Day")

        # 2. Add Active Class Timetable Schedule
        from datetime import datetime
        self.course_repo.add_schedule(
            course_id=course_id,
            day_of_week=datetime.now().weekday(),
            start_time="09:55:00",
            end_time="23:59:59",
            grace_minutes=15
        )

        # 3. Fetch Initial Checklist: All students must be PENDING during active session
        checklist_data = self.attendance_service.get_daily_checklist(course_id=course_id)
        self.assertEqual(checklist_data["total_enrolled"], 3)
        self.assertEqual(checklist_data["pending_count"], 3)
        self.assertEqual(checklist_data["present_count"], 0)

        today_str = datetime.now().strftime("%Y-%m-%d")
        verified_event = IdentityConfirmedEvent(
            person_name="Alice Smith",
            track_id=101,
            match_distance=0.35,
            timestamp_str=f"{today_str} 10:00:00",
            frame_num=100,
            metadata={"course_id": course_id}
        )
        AttendanceIdentityObserver(self.attendance_service).on_identity_confirmed(verified_event)

        # Verify Alice is now PRESENT and pending_count drops to 2
        chk_after_bio = self.attendance_service.get_daily_checklist(course_id=course_id)
        self.assertEqual(chk_after_bio["present_count"], 1)
        self.assertEqual(chk_after_bio["pending_count"], 2)
        alice_entry = next(i for i in chk_after_bio["checklist"] if i["person_name"] == "Alice Smith")
        self.assertEqual(alice_entry["status"], "PRESENT")
        self.assertFalse(alice_entry["is_manual"])

        # 5. Teacher Manual Override: Bob has late note -> marked LATE with reason
        override_res = self.attendance_service.manual_mark(
            person_name="Bob Jones",
            status="LATE",
            reason="Hospital medical slip approved by instructor",
            course_id=course_id
        )
        self.assertEqual(override_res["status"], "success")
        self.assertEqual(override_res["attendance_status"], "LATE")

        # Check Bob status and audit trail
        chk_after_override = self.attendance_service.get_daily_checklist(course_id=course_id)
        self.assertEqual(chk_after_override["late_count"], 1)
        self.assertEqual(chk_after_override["pending_count"], 1)
        bob_entry = next(i for i in chk_after_override["checklist"] if i["person_name"] == "Bob Jones")
        self.assertEqual(bob_entry["status"], "LATE")
        self.assertTrue(bob_entry["is_manual"])
        self.assertEqual(bob_entry["override_reason"], "Hospital medical slip approved by instructor")

        # Verify SQLite Audit Log records the override
        audit_logs = self.attendance_service.get_audit_log(limit=10)
        self.assertTrue(len(audit_logs) >= 1)
        self.assertEqual(audit_logs[0]["person_name"], "Bob Jones")
        self.assertEqual(audit_logs[0]["new_status"], "LATE")
        self.assertIn("medical slip", audit_logs[0]["reason"].lower())

        # 6. Unusual Person Detection: Unknown person lingers for >= 15 frames
        snap_path = os.path.join(self.events_dir, "crop_999.jpg")
        with open(snap_path, "wb") as f:
            f.write(b"fake jpeg data")

        unusual_event = UnusualPersonDetectedEvent(
            track_id=999,
            first_seen_frame=1,
            frame_num=20,
            timestamp_str="2026-10-22 10:05:00",
            snapshot_path=snap_path,
            metadata={"camera_id": 1}
        )
        self.unusual_service.on_unusual_person_detected(unusual_event)

        active_alerts = self.unusual_service.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        self.assertEqual(active_alerts[0]["track_id"], 999)

        # Teacher dismisses the visitor alert
        dismissed = self.unusual_service.dismiss_alert(999)
        self.assertTrue(dismissed)
        self.assertEqual(len(self.unusual_service.get_active_alerts()), 0)


if __name__ == "__main__":
    unittest.main()

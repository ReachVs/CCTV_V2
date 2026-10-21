"""
Comprehensive Unit Tests for AttendanceService.
Validates arrival time evaluation, PRESENT vs LATE classification,
idempotency, checklist generation across enrolled rosters, and manual overrides.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from src.services.attendance_service import AttendanceService, get_cambodia_now


class MockBiometricRepo:
    """Mock repository providing enrolled subjects for checklist testing."""
    def __init__(self, subjects):
        self.subjects = subjects

    def get_all_subjects(self):
        return self.subjects


class TestAttendanceService(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_faces.db")
        self.config_path = os.path.join(self.test_dir, "test_attendance_config.json")

        self.mock_roster = [
            {"id": 1, "name": "Alice Smith", "occupation": "Student", "clearance": "CLASS_A"},
            {"id": 2, "name": "Bob Jones", "occupation": "Student", "clearance": "CLASS_A"},
            {"id": 3, "name": "Charlie Brown", "occupation": "Teaching Assistant", "clearance": "CLASS_B"},
        ]
        self.mock_repo = MockBiometricRepo(self.mock_roster)

        self.service = AttendanceService(
            db_path=self.db_path,
            config_path=self.config_path,
            default_cutoff_time="09:00:00",
            biometric_repo=self.mock_repo
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initial_config(self):
        cfg = self.service.get_config()
        self.assertEqual(cfg["cutoff_time"], "09:00:00")
        self.assertEqual(cfg["grace_period_minutes"], 0)

    def test_set_cutoff_time_valid_formats(self):
        # 24-hour HH:MM
        res = self.service.set_cutoff_time("08:45", grace_period_minutes=5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(self.service.get_config()["cutoff_time"], "08:45:00")
        self.assertEqual(self.service.get_config()["grace_period_minutes"], 5)

        # 12-hour AM/PM
        res = self.service.set_cutoff_time("10:15 AM")
        self.assertEqual(res["status"], "success")
        self.assertEqual(self.service.get_config()["cutoff_time"], "10:15:00")

    def test_set_cutoff_time_invalid_format(self):
        with self.assertRaises(ValueError):
            self.service.set_cutoff_time("invalid_time_string")

    def test_evaluate_status(self):
        self.service.set_cutoff_time("09:00:00", grace_period_minutes=0)

        # 08:59:59 -> PRESENT
        dt_on_time = datetime(2026, 10, 20, 8, 59, 59)
        self.assertEqual(self.service.evaluate_status(dt_on_time), "PRESENT")

        # Exactly 09:00:00 -> PRESENT
        dt_exact = datetime(2026, 10, 20, 9, 0, 0)
        self.assertEqual(self.service.evaluate_status(dt_exact), "PRESENT")

        # 09:00:01 -> LATE
        dt_late = datetime(2026, 10, 20, 9, 0, 1)
        self.assertEqual(self.service.evaluate_status(dt_late), "LATE")

        # With 10 min grace period: 09:05:00 is PRESENT
        self.service.set_cutoff_time("09:00:00", grace_period_minutes=10)
        dt_in_grace = datetime(2026, 10, 20, 9, 5, 0)
        self.assertEqual(self.service.evaluate_status(dt_in_grace), "PRESENT")

        # 09:10:01 -> LATE
        dt_past_grace = datetime(2026, 10, 20, 9, 10, 1)
        self.assertEqual(self.service.evaluate_status(dt_past_grace), "LATE")

    def test_record_check_in_present_and_late(self):
        self.service.set_cutoff_time("09:00:00")

        # Alice arrives at 08:50 (PRESENT)
        res1 = self.service.record_check_in(
            person_name="Alice Smith",
            timestamp_str="2026-10-20 08:50:00",
            track_id=101,
            match_distance=0.32
        )
        self.assertEqual(res1["status"], "new_check_in")
        self.assertEqual(res1["attendance_status"], "PRESENT")
        self.assertEqual(res1["check_in_time"], "08:50:00")

        # Bob arrives at 09:15 (LATE)
        res2 = self.service.record_check_in(
            person_name="Bob Jones",
            timestamp_str="2026-10-20 09:15:22",
            track_id=102,
            match_distance=0.29
        )
        self.assertEqual(res2["status"], "new_check_in")
        self.assertEqual(res2["attendance_status"], "LATE")
        self.assertEqual(res2["check_in_time"], "09:15:22")

    def test_record_check_in_idempotency(self):
        self.service.set_cutoff_time("09:00:00")

        # Initial check-in at 08:45 (PRESENT)
        first_res = self.service.record_check_in(
            person_name="Alice Smith",
            timestamp_str="2026-10-20 08:45:00"
        )
        self.assertEqual(first_res["status"], "new_check_in")
        self.assertEqual(first_res["attendance_status"], "PRESENT")
        self.assertEqual(first_res["check_in_time"], "08:45:00")

        # Subsequent detection at 10:30 on same day
        second_res = self.service.record_check_in(
            person_name="Alice Smith",
            timestamp_str="2026-10-20 10:30:00"
        )
        self.assertEqual(second_res["status"], "already_recorded")
        # Ensure initial check-in time and status were preserved
        self.assertEqual(second_res["attendance_status"], "PRESENT")
        self.assertEqual(second_res["check_in_time"], "08:45:00")

    def test_anonymous_and_empty_names_ignored(self):
        res = self.service.record_check_in("Unknown Person")
        self.assertEqual(res["status"], "ignored")

        res_empty = self.service.record_check_in("")
        self.assertEqual(res_empty["status"], "ignored")

    def test_get_daily_checklist(self):
        self.service.set_cutoff_time("09:00:00")

        # Alice checked in on time
        self.service.record_check_in("Alice Smith", "2026-10-20 08:55:00")
        # Bob checked in late
        self.service.record_check_in("Bob Jones", "2026-10-20 09:20:00")
        # Charlie Brown did not check in (ABSENT)

        checklist_data = self.service.get_daily_checklist("2026-10-20")

        self.assertEqual(checklist_data["total_enrolled"], 3)
        self.assertEqual(checklist_data["present_count"], 1)
        self.assertEqual(checklist_data["late_count"], 1)
        self.assertEqual(checklist_data["absent_count"], 1)
        self.assertEqual(checklist_data["attended_count"], 2)
        self.assertAlmostEqual(checklist_data["attendance_rate"], 66.7, places=1)

        # Inspect items
        items_by_name = {item["person_name"]: item for item in checklist_data["checklist"]}
        self.assertEqual(items_by_name["Alice Smith"]["status"], "PRESENT")
        self.assertEqual(items_by_name["Alice Smith"]["check_in_time"], "08:55:00")

        self.assertEqual(items_by_name["Bob Jones"]["status"], "LATE")
        self.assertEqual(items_by_name["Bob Jones"]["check_in_time"], "09:20:00")

        self.assertEqual(items_by_name["Charlie Brown"]["status"], "ABSENT")
        self.assertIsNone(items_by_name["Charlie Brown"]["check_in_time"])

    def test_manual_mark_and_override(self):
        self.service.set_cutoff_time("09:00:00")
        date_str = "2026-10-20"

        # Initially Bob is absent
        # Teacher manually marks Bob Jones as PRESENT with note
        res = self.service.manual_mark("Bob Jones", "PRESENT", date_str=date_str, custom_time="08:58:00")
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["attendance_status"], "PRESENT")

        chk = self.service.get_daily_checklist(date_str)
        items = {i["person_name"]: i for i in chk["checklist"]}
        self.assertEqual(items["Bob Jones"]["status"], "PRESENT")

        # Now mark as ABSENT (clears check-in)
        res_abs = self.service.manual_mark("Bob Jones", "ABSENT", date_str=date_str)
        self.assertEqual(res_abs["status"], "success")

        chk_after = self.service.get_daily_checklist(date_str)
        items_after = {i["person_name"]: i for i in chk_after["checklist"]}
        self.assertEqual(items_after["Bob Jones"]["status"], "ABSENT")

    def test_reset_daily_attendance(self):
        date_str = "2026-10-20"
        self.service.record_check_in("Alice Smith", f"{date_str} 08:50:00")
        self.service.record_check_in("Bob Jones", f"{date_str} 09:10:00")

        chk = self.service.get_daily_checklist(date_str)
        self.assertEqual(chk["attended_count"], 2)

        reset_res = self.service.reset_daily_attendance(date_str)
        self.assertEqual(reset_res["status"], "success")
        self.assertEqual(reset_res["deleted_records"], 2)

        chk_empty = self.service.get_daily_checklist(date_str)
        self.assertEqual(chk_empty["attended_count"], 0)
        self.assertEqual(chk_empty["absent_count"], 3)

    def test_set_schedule_and_end_time_evaluation(self):
        # Schedule: 08:30 to 11:30 with 10 min grace
        self.service.set_schedule(class_start_time="08:30:00", class_end_time="11:30:00", grace_period_minutes=10)
        cfg = self.service.get_config()
        self.assertEqual(cfg["class_start_time"], "08:30:00")
        self.assertEqual(cfg["class_end_time"], "11:30:00")
        self.assertEqual(cfg["grace_period_minutes"], 10)

        # On time (<= 08:40) -> PRESENT
        dt_on_time = datetime(2026, 10, 20, 8, 35, 0)
        self.assertEqual(self.service.evaluate_status(dt_on_time), "PRESENT")

        # In grace window (08:40) -> PRESENT
        dt_grace = datetime(2026, 10, 20, 8, 40, 0)
        self.assertEqual(self.service.evaluate_status(dt_grace), "PRESENT")

        # Late (08:41 to 11:30) -> LATE
        dt_late = datetime(2026, 10, 20, 9, 15, 0)
        self.assertEqual(self.service.evaluate_status(dt_late), "LATE")

        # Exactly at class end -> LATE
        dt_end = datetime(2026, 10, 20, 11, 30, 0)
        self.assertEqual(self.service.evaluate_status(dt_end), "LATE")

        # After class end (> 11:30) -> ABSENT (missed class)
        dt_after_class = datetime(2026, 10, 20, 11, 30, 1)
        self.assertEqual(self.service.evaluate_status(dt_after_class), "ABSENT")

    def test_check_all(self):
        date_str = "2026-10-20"
        # Initially all 3 enrolled are ABSENT
        chk0 = self.service.get_daily_checklist(date_str)
        self.assertEqual(chk0["total_enrolled"], 3)
        self.assertEqual(chk0["absent_count"], 3)

        # Check all enrolled students as PRESENT
        res = self.service.check_all(status="PRESENT", date_str=date_str)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_updated"], 3)

        chk1 = self.service.get_daily_checklist(date_str)
        self.assertEqual(chk1["present_count"], 3)
        self.assertEqual(chk1["absent_count"], 0)
        self.assertEqual(chk1["attended_count"], 3)

    def test_roster_provider_adapter(self):
        from src.services.attendance_service import InMemoryRosterProvider, BiometricRepositoryRosterProvider

        custom_roster = [
            {"id": 101, "name": "Custom Student A", "occupation": "Graduate", "clearance": "LAB_1"},
            {"id": 102, "name": "Custom Student B", "occupation": "Undergrad", "clearance": "LAB_2"}
        ]
        in_memory_provider = InMemoryRosterProvider(custom_roster)
        self.service.set_roster_provider(in_memory_provider)

        chk = self.service.get_daily_checklist("2026-10-20")
        self.assertEqual(chk["total_enrolled"], 2)
        names = {i["person_name"] for i in chk["checklist"]}
        self.assertEqual(names, {"Custom Student A", "Custom Student B"})


    def test_early_attendance_and_schedule_update(self):
        # Schedule: 18:43 to 18:47 with 15m grace
        self.service.set_schedule(class_start_time="18:43:00", class_end_time="18:47:00", grace_period_minutes=15)

        # Student arrives early at 18:42:20 (before class start 18:43)
        res = self.service.record_check_in(
            person_name="Ceaser",
            timestamp_str="2026-10-21 18:42:20",
            track_id=53,
            match_distance=0.22
        )
        self.assertEqual(res["status"], "new_check_in")
        self.assertEqual(res["attendance_status"], "PRESENT")

        chk = self.service.get_daily_checklist("2026-10-21")
        ceaser_chk = next(i for i in chk["checklist"] if i["person_name"] == "Ceaser")
        self.assertEqual(ceaser_chk["status"], "PRESENT")
        self.assertEqual(ceaser_chk["check_in_time"], "18:42:20")

    def test_schedule_update_re_evaluates_status(self):
        today = get_cambodia_now().strftime("%Y-%m-%d")
        # Schedule originally 18:00 to 18:30 (grace 0)
        self.service.set_schedule(class_start_time="18:00:00", class_end_time="18:30:00", grace_period_minutes=0)
        # Check-in at 18:15 (LATE under 18:00 start with 0 grace)
        res = self.service.record_check_in("Student X", f"{today} 18:15:00")
        self.assertEqual(res["attendance_status"], "LATE")

        # Now schedule is updated to 18:15 with 15m grace (18:15 is on time)
        self.service.set_schedule(class_start_time="18:15:00", class_end_time="20:00:00", grace_period_minutes=15)
        chk = self.service.get_daily_checklist(today)
        item = next(i for i in chk["checklist"] if i["person_name"] == "Student X")
        self.assertEqual(item["status"], "PRESENT")


if __name__ == "__main__":
    unittest.main()


"""
Hermetic Unit Tests for Course & Timetable Repository.
Verifies course creation, schedule slot management, enrollment roster queries,
and active session resolution based on day-of-week and time.
"""
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, time

from src.domain.course_repository import CourseRepository


class TestCourseRepository(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_school.db")
        self.repo = CourseRepository(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_initialization(self):
        """Verifies that courses, schedules, and enrollments tables are created."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()
        self.assertIn("courses", tables)
        self.assertIn("schedules", tables)
        self.assertIn("course_enrollments", tables)

    def test_create_and_list_courses(self):
        """Verifies course creation and retrieval."""
        c1 = self.repo.create_course(code="CS101", name="Computer Vision", department="CS")
        self.assertEqual(c1["code"], "CS101")
        self.assertEqual(c1["name"], "Computer Vision")

        courses = self.repo.list_courses()
        self.assertEqual(len(courses), 1)
        self.assertEqual(courses[0]["code"], "CS101")

        # Duplicate code should raise or be handled
        with self.assertRaises(ValueError):
            self.repo.create_course(code="CS101", name="Duplicate Code")

    def test_enroll_and_query_students(self):
        """Verifies student enrollment and roster listing."""
        c = self.repo.create_course(code="MATH201", name="Linear Algebra")
        cid = c["id"]

        self.assertTrue(self.repo.enroll_student(cid, "Alice Smith"))
        self.assertTrue(self.repo.enroll_student(cid, "Bob Jones"))
        # Enrolling duplicate student is idempotent
        self.assertTrue(self.repo.enroll_student(cid, "Alice Smith"))

        roster = self.repo.get_enrolled_students(cid)
        self.assertEqual(sorted(roster), ["Alice Smith", "Bob Jones"])

        # Unenroll
        self.assertTrue(self.repo.unenroll_student(cid, "Alice Smith"))
        roster_after = self.repo.get_enrolled_students(cid)
        self.assertEqual(roster_after, ["Bob Jones"])

    def test_add_schedule_and_get_active_session(self):
        """Verifies schedule creation and current active session resolution."""
        c = self.repo.create_course(code="CCTV100", name="Video Analytics")
        cid = c["id"]

        # Monday is weekday 0, Wednesday is 2
        # Add schedule for Wednesday 14:00 - 16:00 (grace: 15)
        sched = self.repo.add_schedule(
            course_id=cid,
            day_of_week=2,
            start_time="14:00:00",
            end_time="16:00:00",
            grace_minutes=15
        )
        self.assertEqual(sched["day_of_week"], 2)

        # 1. Target time is Wednesday 14:30 -> Should resolve to CCTV100
        wed_active_dt = datetime(2026, 10, 21, 14, 30, 0) # 2026-10-21 is Wednesday
        active = self.repo.get_active_session(wed_active_dt)
        self.assertIsNotNone(active)
        self.assertEqual(active["course"]["code"], "CCTV100")
        self.assertEqual(active["schedule"]["grace_minutes"], 15)

        # 2. Target time is Wednesday 16:30 -> After class, no active session
        wed_past_dt = datetime(2026, 10, 21, 16, 30, 0)
        self.assertIsNone(self.repo.get_active_session(wed_past_dt))

        # 3. Target time is Thursday 14:30 -> Wrong day, no active session
        thu_dt = datetime(2026, 10, 22, 14, 30, 0) # 2026-10-22 is Thursday
        self.assertIsNone(self.repo.get_active_session(thu_dt))


if __name__ == "__main__":
    unittest.main()

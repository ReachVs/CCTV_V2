"""
Course & Timetable Repository
=============================
Authoritative data access module for academic courses, recurring schedules,
student course enrollments, and real-time active session resolution.
"""
import os
import sqlite3
import threading
from datetime import datetime, time as dtime
from typing import Dict, Any, List, Optional, Tuple


class CourseRepository:
    """
    Deep Module: Manages courses, weekly schedules, student enrollments,
    and resolves the currently active academic session.
    """

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Creates courses, schedules, and course_enrollments tables if they don't exist."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS courses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        code TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        department TEXT DEFAULT 'GENERAL',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS schedules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                        day_of_week INTEGER NOT NULL, -- 0=Monday .. 6=Sunday
                        start_time TEXT NOT NULL,     -- "HH:MM:SS"
                        end_time TEXT NOT NULL,       -- "HH:MM:SS"
                        grace_minutes INTEGER DEFAULT 15,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS course_enrollments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                        person_name TEXT NOT NULL,
                        enrolled_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(course_id, person_name)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sched_course ON schedules(course_id);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_sched_day ON schedules(day_of_week);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_enroll_course ON course_enrollments(course_id);")
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _normalize_time(time_str: str) -> str:
        """Parses and formats time string into HH:MM:SS."""
        clean = time_str.strip().upper()
        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
            try:
                parsed = datetime.strptime(clean, fmt)
                return parsed.strftime("%H:%M:%S")
            except ValueError:
                continue
        raise ValueError(f"Invalid time format: '{time_str}'. Expected HH:MM or HH:MM:SS.")

    # ---------------------------------------------------------
    # Course CRUD
    # ---------------------------------------------------------

    def create_course(self, code: str, name: str, department: str = "GENERAL") -> Dict[str, Any]:
        """Creates a new course with a unique code."""
        clean_code = code.strip().upper()
        clean_name = name.strip()
        clean_dept = department.strip().upper()

        if not clean_code or not clean_name:
            raise ValueError("Course code and name must not be empty.")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO courses (code, name, department) VALUES (?, ?, ?);",
                    (clean_code, clean_name, clean_dept)
                )
                course_id = cur.lastrowid
                conn.commit()
                return {
                    "id": course_id,
                    "code": clean_code,
                    "name": clean_name,
                    "department": clean_dept
                }
            except sqlite3.IntegrityError:
                raise ValueError(f"Course with code '{clean_code}' already exists.")
            finally:
                conn.close()

    def get_course(self, course_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a course by its primary key ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT id, code, name, department, created_at FROM courses WHERE id = ?;", (course_id,))
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_course_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Retrieves a course by its unique course code."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT id, code, name, department, created_at FROM courses WHERE code = ?;", (code.strip().upper(),))
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def list_courses(self) -> List[Dict[str, Any]]:
        """Lists all registered courses sorted by code."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT id, code, name, department, created_at FROM courses ORDER BY code ASC;")
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def delete_course(self, course_id: int) -> bool:
        """Deletes a course and cascades deletion of its schedules and enrollments."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM courses WHERE id = ?;", (course_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ---------------------------------------------------------
    # Schedule Management
    # ---------------------------------------------------------

    def add_schedule(
        self,
        course_id: int,
        day_of_week: int,
        start_time: str,
        end_time: str,
        grace_minutes: int = 15,
        is_active: int = 1
    ) -> Dict[str, Any]:
        """
        Adds a recurring weekly schedule slot for a course.
        day_of_week: 0=Monday, 1=Tuesday, 2=Wednesday, 3=Thursday, 4=Friday, 5=Saturday, 6=Sunday.
        """
        if not (0 <= day_of_week <= 6):
            raise ValueError("day_of_week must be an integer between 0 (Monday) and 6 (Sunday).")

        norm_start = self._normalize_time(start_time)
        norm_end = self._normalize_time(end_time)
        grace = max(0, min(120, int(grace_minutes)))

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                # Verify course exists
                cur.execute("SELECT id FROM courses WHERE id = ?;", (course_id,))
                if not cur.fetchone():
                    raise ValueError(f"Course with ID {course_id} does not exist.")

                cur.execute("""
                    INSERT INTO schedules (course_id, day_of_week, start_time, end_time, grace_minutes, is_active)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (course_id, day_of_week, norm_start, norm_end, grace, 1 if is_active else 0))
                sched_id = cur.lastrowid
                conn.commit()
                return {
                    "id": sched_id,
                    "course_id": course_id,
                    "day_of_week": day_of_week,
                    "start_time": norm_start,
                    "end_time": norm_end,
                    "grace_minutes": grace,
                    "is_active": is_active
                }
            finally:
                conn.close()

    def list_schedules(self, course_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Lists schedules, optionally filtered by course_id."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if course_id is not None:
                    cur.execute("""
                        SELECT s.id, s.course_id, c.code AS course_code, c.name AS course_name,
                               s.day_of_week, s.start_time, s.end_time, s.grace_minutes, s.is_active
                        FROM schedules s
                        JOIN courses c ON s.course_id = c.id
                        WHERE s.course_id = ?
                        ORDER BY s.day_of_week ASC, s.start_time ASC;
                    """, (course_id,))
                else:
                    cur.execute("""
                        SELECT s.id, s.course_id, c.code AS course_code, c.name AS course_name,
                               s.day_of_week, s.start_time, s.end_time, s.grace_minutes, s.is_active
                        FROM schedules s
                        JOIN courses c ON s.course_id = c.id
                        ORDER BY s.day_of_week ASC, s.start_time ASC;
                    """)
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def delete_schedule(self, schedule_id: int) -> bool:
        """Deletes a schedule slot."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM schedules WHERE id = ?;", (schedule_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    # ---------------------------------------------------------
    # Enrollment Management
    # ---------------------------------------------------------

    def enroll_student(self, course_id: int, person_name: str) -> bool:
        """Enrolls a student identity into a course (idempotent)."""
        clean_name = person_name.strip()
        if not clean_name:
            raise ValueError("Student name must not be empty.")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR IGNORE INTO course_enrollments (course_id, person_name)
                    VALUES (?, ?);
                """, (course_id, clean_name))
                conn.commit()
                return True
            finally:
                conn.close()

    def unenroll_student(self, course_id: int, person_name: str) -> bool:
        """Removes a student identity from a course roster."""
        clean_name = person_name.strip()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    DELETE FROM course_enrollments
                    WHERE course_id = ? AND person_name = ?;
                """, (course_id, clean_name))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get_enrolled_students(self, course_id: int) -> List[str]:
        """Returns the list of enrolled student names for a course."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT person_name
                    FROM course_enrollments
                    WHERE course_id = ?
                    ORDER BY person_name ASC;
                """, (course_id,))
                return [row["person_name"] for row in cur.fetchall()]
            finally:
                conn.close()

    # ---------------------------------------------------------
    # Active Session Resolution
    # ---------------------------------------------------------

    def get_active_session(self, target_dt: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Resolves the currently active academic course session for a given datetime.
        Matches active schedules where:
          - schedule.day_of_week == target_dt.weekday() (0=Monday .. 6=Sunday)
          - schedule.start_time <= target_time <= schedule.end_time
        Returns dict with course details and schedule details, or None if outside any scheduled class.
        """
        now = target_dt if target_dt is not None else datetime.now()
        day_of_week = now.weekday()
        current_time_str = now.strftime("%H:%M:%S")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT s.id AS schedule_id, s.start_time, s.end_time, s.grace_minutes,
                           c.id AS course_id, c.code AS course_code, c.name AS course_name, c.department
                    FROM schedules s
                    JOIN courses c ON s.course_id = c.id
                    WHERE s.is_active = 1
                      AND s.day_of_week = ?
                      AND s.start_time <= ?
                      AND s.end_time >= ?
                    ORDER BY s.start_time ASC
                    LIMIT 1;
                """, (day_of_week, current_time_str, current_time_str))
                row = cur.fetchone()
                if not row:
                    return None

                return {
                    "course": {
                        "id": row["course_id"],
                        "code": row["course_code"],
                        "name": row["course_name"],
                        "department": row["department"]
                    },
                    "schedule": {
                        "id": row["schedule_id"],
                        "start_time": row["start_time"],
                        "end_time": row["end_time"],
                        "grace_minutes": row["grace_minutes"]
                    }
                }
            finally:
                conn.close()

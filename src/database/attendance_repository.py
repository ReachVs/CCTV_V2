"""
Attendance Database Repository
==============================
Authoritative data access module for attendance records and audit logs.
Encapsulates SQLite connection pooling, transactions, schema migrations, and queries.
"""
import os
import sqlite3
import threading
from typing import Dict, Any, List, Optional


class AttendanceRepository:
    """
    Data Access Layer managing attendance records and forensic audit logs.
    Thread-safe via internal RLock.
    """

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        self._lock = threading.RLock()
        self._ensure_schema()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensures the attendance_records and attendance_audit_log tables exist with safe column migrations."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        person_name TEXT NOT NULL,
                        check_in_time TEXT NOT NULL,
                        status TEXT NOT NULL,
                        track_id INTEGER DEFAULT 0,
                        match_distance REAL DEFAULT 0.0,
                        snapshot_path TEXT DEFAULT '',
                        course_id INTEGER DEFAULT NULL,
                        is_manual INTEGER DEFAULT 0,
                        override_reason TEXT DEFAULT '',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, person_name)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(date);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance_records(person_name);")

                # Schema migrations for existing databases
                cur.execute("PRAGMA table_info(attendance_records);")
                existing_cols = {col["name"] for col in cur.fetchall()}
                if "course_id" not in existing_cols:
                    cur.execute("ALTER TABLE attendance_records ADD COLUMN course_id INTEGER DEFAULT NULL;")
                if "is_manual" not in existing_cols:
                    cur.execute("ALTER TABLE attendance_records ADD COLUMN is_manual INTEGER DEFAULT 0;")
                if "override_reason" not in existing_cols:
                    cur.execute("ALTER TABLE attendance_records ADD COLUMN override_reason TEXT DEFAULT '';")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        attendance_id INTEGER,
                        person_name TEXT NOT NULL,
                        old_status TEXT NOT NULL,
                        new_status TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        changed_by TEXT DEFAULT 'TEACHER',
                        changed_at TEXT DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                conn.commit()
            finally:
                conn.close()

    def get_records_by_date(
        self,
        date_str: str,
        course_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieves attendance records for a specific date and optional course."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                if course_id is not None:
                    cur.execute("""
                        SELECT id, date, person_name, check_in_time, status, track_id, match_distance, snapshot_path,
                               course_id, is_manual, override_reason
                        FROM attendance_records
                        WHERE date = ? AND (course_id = ? OR course_id IS NULL)
                        ORDER BY check_in_time ASC;
                    """, (date_str, course_id))
                else:
                    cur.execute("""
                        SELECT id, date, person_name, check_in_time, status, track_id, match_distance, snapshot_path,
                               course_id, is_manual, override_reason
                        FROM attendance_records
                        WHERE date = ?
                        ORDER BY check_in_time ASC;
                    """, (date_str,))
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

    def get_record(
        self,
        date_str: str,
        person_name: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieves a single person's attendance record for a specific date."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, date, person_name, check_in_time, status, track_id, match_distance, snapshot_path,
                           course_id, is_manual, override_reason
                    FROM attendance_records
                    WHERE date = ? AND LOWER(person_name) = ?;
                """, (date_str, person_name.lower()))
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def insert_check_in(
        self,
        date_str: str,
        person_name: str,
        check_in_time: str,
        status: str,
        track_id: int = 0,
        match_distance: float = 0.0,
        snapshot_path: str = "",
        course_id: Optional[int] = None
    ) -> Optional[int]:
        """Inserts a new attendance record. Returns record ID if inserted, None if duplicate."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT OR IGNORE INTO attendance_records (
                        date, person_name, check_in_time, status, track_id, match_distance, snapshot_path, course_id, is_manual
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0);
                """, (date_str, person_name, check_in_time, status, track_id, match_distance, snapshot_path, course_id))
                inserted = (cur.rowcount > 0)
                record_id = cur.lastrowid if inserted else None
                conn.commit()
                return record_id
            finally:
                conn.close()

    def update_record_status(
        self,
        record_id: int,
        new_status: str
    ) -> bool:
        """Updates status of an existing attendance record."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute("UPDATE attendance_records SET status = ? WHERE id = ?;", (new_status, record_id))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def delete_record(self, record_id: int) -> bool:
        """Deletes an attendance record by ID."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM attendance_records WHERE id = ?;", (record_id,))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def upsert_manual_mark(
        self,
        date_str: str,
        person_name: str,
        status: str,
        check_in_time: str,
        reason: str,
        course_id: Optional[int] = None,
        changed_by: str = "TEACHER"
    ) -> Dict[str, Any]:
        """Upserts a manual attendance record and writes an audit log entry."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, status FROM attendance_records WHERE date = ? AND LOWER(person_name) = ?;",
                    (date_str, person_name.lower())
                )
                existing = cur.fetchone()

                if existing:
                    rec_id = existing["id"]
                    old_status = existing["status"]
                    cur.execute("""
                        UPDATE attendance_records
                        SET status = ?, is_manual = 1, override_reason = ?, check_in_time = ?, course_id = COALESCE(?, course_id)
                        WHERE id = ?;
                    """, (status, reason, check_in_time, course_id, rec_id))
                else:
                    old_status = "ABSENT"
                    cur.execute("""
                        INSERT INTO attendance_records (date, person_name, check_in_time, status, course_id, is_manual, override_reason)
                        VALUES (?, ?, ?, ?, ?, 1, ?);
                    """, (date_str, person_name, check_in_time, status, course_id, reason))
                    rec_id = cur.lastrowid

                # Write forensic audit log entry
                cur.execute("""
                    INSERT INTO attendance_audit_log (attendance_id, person_name, old_status, new_status, reason, changed_by)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (rec_id, person_name, old_status, status, reason, changed_by))

                conn.commit()
                return {
                    "id": rec_id,
                    "old_status": old_status,
                    "new_status": status,
                    "check_in_time": check_in_time
                }
            finally:
                conn.close()

    def get_audit_log(
        self,
        date_str: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Retrieves audit log entries for overrides."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                if date_str:
                    cur.execute("""
                        SELECT a.id, a.attendance_id, a.person_name, a.old_status, a.new_status, a.reason, a.changed_by, a.changed_at,
                               r.date
                        FROM attendance_audit_log a
                        LEFT JOIN attendance_records r ON a.attendance_id = r.id
                        WHERE r.date = ? OR DATE(a.changed_at) = ?
                        ORDER BY a.id DESC
                        LIMIT ?;
                    """, (date_str, date_str, max(1, min(200, limit))))
                else:
                    cur.execute("""
                        SELECT a.id, a.attendance_id, a.person_name, a.old_status, a.new_status, a.reason, a.changed_by, a.changed_at,
                               r.date
                        FROM attendance_audit_log a
                        LEFT JOIN attendance_records r ON a.attendance_id = r.id
                        ORDER BY a.id DESC
                        LIMIT ?;
                    """, (max(1, min(200, limit)),))
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    def reset_daily_attendance(
        self,
        date_str: str,
        course_id: Optional[int] = None
    ) -> int:
        """Deletes attendance records for the given date. Returns count of deleted records."""
        with self._lock:
            conn = self.get_connection()
            try:
                cur = conn.cursor()
                if course_id is not None:
                    cur.execute("DELETE FROM attendance_records WHERE date = ? AND course_id = ?;", (date_str, course_id))
                else:
                    cur.execute("DELETE FROM attendance_records WHERE date = ?;", (date_str,))
                deleted_count = cur.rowcount
                conn.commit()
                return deleted_count
            finally:
                conn.close()

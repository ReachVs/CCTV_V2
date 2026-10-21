"""
Attendance Verification & Checklist Service.
Manages automated biometric attendance recording, daily checklists,
and arrival status classification (PRESENT vs LATE) based on a configurable cutoff time.
"""
import os
import re
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, time as dtime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple

try:
    from zoneinfo import ZoneInfo
    CAMBODIA_TZ = ZoneInfo("Asia/Phnom_Penh")
except Exception:
    from datetime import timezone
    CAMBODIA_TZ = timezone(timedelta(hours=7))


def get_cambodia_now() -> datetime:
    """Returns the current datetime in Cambodia timezone (UTC+7 / ICT) as a naive datetime for seamless arithmetic."""
    return datetime.now(CAMBODIA_TZ).replace(tzinfo=None)


# ISO 8601 YYYY-MM-DD Date Pattern
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")



class RosterProvider(ABC):
    """
    Abstract Seam for attendee roster sources.
    Decouples daily checklist generation from biometric vector storage,
    enabling pluggable university SIS, CSV roster files, or LDAP registries.
    """
    @abstractmethod
    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        """Returns list of enrolled attendee dicts: [{'name': str, 'occupation': str, 'clearance': str, ...}]"""
        pass


class BiometricRepositoryRosterProvider(RosterProvider):
    """Default adapter fetching enrolled subjects from BiometricRepository."""
    def __init__(self, biometric_repo: Any):
        self.biometric_repo = biometric_repo

    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        if self.biometric_repo is not None and hasattr(self.biometric_repo, "get_all_subjects"):
            try:
                return self.biometric_repo.get_all_subjects()
            except Exception as e:
                print(f"[RosterProvider Warning] Could not fetch enrolled subjects: {e}")
        return []


class InMemoryRosterProvider(RosterProvider):
    """In-memory roster test double for deterministic unit/integration testing."""
    def __init__(self, roster: Optional[List[Dict[str, Any]]] = None):
        self.roster = list(roster) if roster else []

    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        return list(self.roster)

    def set_roster(self, roster: List[Dict[str, Any]]) -> None:
        self.roster = list(roster)


class AttendanceService:
    """
    Deep Module: Manages automated attendance tracking for enrolled biometric subjects.

    Key Architectural Principles:
    - Maintains target attendance cutoff time (e.g. 09:00:00).
    - Classifies arrival as 'PRESENT' (on time) or 'LATE' (after cutoff + grace).
    - In-Memory Fast-Path Cache: Avoids redundant per-frame SQLite transactions once a subject is recorded.
    - Strict Idempotency: First verified check-in for the day is permanently locked via UNIQUE(date, person_name).
    - RosterProvider Seam: Decouples daily checklist generation from biometric vector storage.
    - Thread-safe SQLite WAL persistence with safe connection lifecycle management.
    """

    def __init__(
        self,
        db_path: str,
        config_path: Optional[str] = None,
        default_cutoff_time: str = "18:00:00",
        biometric_repo: Optional[Any] = None,
        roster_provider: Optional[RosterProvider] = None,
        default_grace_period_minutes: int = 0
    ):
        self.db_path = os.path.abspath(db_path)
        self.config_path = config_path or os.path.join(os.path.dirname(self.db_path), "attendance_config.json")
        self.biometric_repo = biometric_repo
        if roster_provider is not None:
            self.roster_provider = roster_provider
        elif biometric_repo is not None:
            self.roster_provider = BiometricRepositoryRosterProvider(biometric_repo)
        else:
            self.roster_provider = InMemoryRosterProvider([])
        self._lock = threading.RLock()

        # In-memory fast lookup cache: date_str -> {name_lower: {"id": id, "person_name": name, "check_in_time": time, "status": status}}
        self._checked_in_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # Load or initialize configuration (Defaults to Cambodia Evening Session: 18:00 - 21:00)
        self._class_start_time_str = default_cutoff_time
        self._class_end_time_str = "21:00:00"
        self._cutoff_time_str = self._class_start_time_str
        self._grace_period_minutes = default_grace_period_minutes
        self._load_config()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensures the attendance_records table and indexes exist without dropping data."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS attendance_records (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        person_name TEXT NOT NULL,
                        check_in_time TEXT NOT NULL,
                        status TEXT NOT NULL, -- 'PRESENT' or 'LATE'
                        track_id INTEGER DEFAULT 0,
                        match_distance REAL DEFAULT 0.0,
                        snapshot_path TEXT DEFAULT '',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, person_name)
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(date);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_attendance_person ON attendance_records(person_name);")
                conn.commit()

                # Pre-populate today's in-memory checked-in cache (Cambodia local date)
                today = get_cambodia_now().strftime("%Y-%m-%d")
                cur.execute("SELECT id, person_name, check_in_time, status FROM attendance_records WHERE date = ?;", (today,))
                self._checked_in_cache[today] = {
                    row["person_name"].lower(): {
                        "id": row["id"],
                        "person_name": row["person_name"],
                        "check_in_time": row["check_in_time"],
                        "status": row["status"]
                    }
                    for row in cur.fetchall()
                }
            finally:
                conn.close()

    def _load_config(self) -> None:
        """Loads configuration from persistent JSON file if available."""
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._class_start_time_str = data.get("class_start_time", data.get("cutoff_time", self._class_start_time_str))
                        self._class_end_time_str = data.get("class_end_time", self._class_end_time_str)
                        self._cutoff_time_str = self._class_start_time_str
                        self._grace_period_minutes = max(0, min(120, int(data.get("grace_period_minutes", 0))))
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not load config: {e}")

    def _save_config(self) -> None:
        """Persists current configuration to JSON file."""
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "class_start_time": self._class_start_time_str,
                        "class_end_time": self._class_end_time_str,
                        "cutoff_time": self._cutoff_time_str,
                        "grace_period_minutes": self._grace_period_minutes,
                        "updated_at": get_cambodia_now().isoformat()
                    }, f, indent=2)
            except Exception as e:
                print(f"[AttendanceService Warning] Could not save config: {e}")

    @staticmethod
    def _parse_time(time_str: str) -> dtime:
        """
        Parses time string in formats like '09:00', '09:00:00', '9:30', or '9:30 AM'.
        Raises ValueError if invalid.
        """
        if not time_str or not isinstance(time_str, str):
            raise ValueError("Time string must be a non-empty string.")

        clean = time_str.strip().upper()
        if len(clean) > 20:
            raise ValueError("Time string exceeds maximum allowed length.")

        for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
            try:
                parsed = datetime.strptime(clean, fmt)
                return parsed.time()
            except ValueError:
                continue
        raise ValueError(f"Invalid time format: '{time_str}'. Expected HH:MM or HH:MM:SS (24-hour).")

    @staticmethod
    def _validate_date(date_str: str) -> str:
        """Validates that a date string is in strict ISO 8601 YYYY-MM-DD format."""
        clean = date_str.strip()
        if not DATE_PATTERN.match(clean):
            raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")
        try:
            datetime.strptime(clean, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid calendar date: '{date_str}'.")
        return clean

    def set_roster_provider(self, provider: RosterProvider) -> None:
        """Hot-swaps the attendee roster provider adapter."""
        with self._lock:
            self.roster_provider = provider

    def set_schedule(
        self,
        class_start_time: str,
        class_end_time: Optional[str] = None,
        grace_period_minutes: int = 0
    ) -> Dict[str, Any]:
        """
        Sets and persists the class start time, class end time, and grace period.
        - Arrival <= class_start_time + grace_period -> 'PRESENT'
        - class_start_time + grace_period < Arrival <= class_end_time -> 'LATE'
        - Arrival > class_end_time -> 'ABSENT'
        """
        with self._lock:
            parsed_start = self._parse_time(class_start_time)
            norm_start = parsed_start.strftime("%H:%M:%S")

            if class_end_time:
                parsed_end = self._parse_time(class_end_time)
                norm_end = parsed_end.strftime("%H:%M:%S")
            else:
                norm_end = self._class_end_time_str

            self._class_start_time_str = norm_start
            self._class_end_time_str = norm_end
            self._cutoff_time_str = norm_start
            self._grace_period_minutes = max(0, min(120, int(grace_period_minutes)))
            self._save_config()

            # Dynamically re-evaluate existing check-in records for today under the updated schedule
            try:
                today = get_cambodia_now().strftime("%Y-%m-%d")
                conn = self._get_connection()
                try:
                    cur = conn.cursor()
                    cur.execute("SELECT id, date, person_name, check_in_time FROM attendance_records WHERE date = ?;", (today,))
                    rows = cur.fetchall()
                    cached_day = self._checked_in_cache.setdefault(today, {})
                    for row in rows:
                        rec_id = row["id"]
                        p_name = row["person_name"]
                        c_time = row["check_in_time"]
                        try:
                            dt = datetime.strptime(f"{today} {c_time}", "%Y-%m-%d %H:%M:%S")
                            new_status = self.evaluate_status(dt)
                            cur.execute("UPDATE attendance_records SET status = ? WHERE id = ?;", (new_status, rec_id))
                            cached_day[p_name.lower()] = {
                                "id": rec_id,
                                "person_name": p_name,
                                "check_in_time": c_time,
                                "status": new_status
                            }
                        except Exception:
                            pass
                    conn.commit()
                finally:
                    conn.close()
            except Exception as e:
                print(f"[AttendanceService Warning] Could not re-evaluate existing records: {e}")

            return {
                "status": "success",
                "class_start_time": self._class_start_time_str,
                "class_end_time": self._class_end_time_str,
                "cutoff_time": self._cutoff_time_str,
                "grace_period_minutes": self._grace_period_minutes,
                "message": f"Class schedule set: {self._class_start_time_str} to {self._class_end_time_str} (Grace: {self._grace_period_minutes}m)"
            }

    def set_cutoff_time(self, cutoff_time: str, grace_period_minutes: int = 0) -> Dict[str, Any]:
        """
        Backward-compatible setter for attendance cutoff (maps to class_start_time).
        """
        return self.set_schedule(class_start_time=cutoff_time, class_end_time=self._class_end_time_str, grace_period_minutes=grace_period_minutes)

    def get_config(self) -> Dict[str, Any]:
        """Returns the active class attendance schedule configuration."""
        with self._lock:
            return {
                "class_start_time": self._class_start_time_str,
                "class_end_time": self._class_end_time_str,
                "cutoff_time": self._cutoff_time_str,
                "grace_period_minutes": self._grace_period_minutes,
                "current_time": get_cambodia_now().strftime("%H:%M:%S"),
                "today": get_cambodia_now().strftime("%Y-%m-%d")
            }

    def evaluate_status(self, check_in_dt: datetime) -> str:
        """
        Evaluates arrival status against class start and end times:
        - Arrival on or before class_start_time + grace_period -> 'PRESENT'
        - Arrival after class_start_time + grace_period but on or before class_end_time -> 'LATE'
        - Arrival after class_end_time -> 'ABSENT' (missed class)
        """
        with self._lock:
            start_t = self._parse_time(self._class_start_time_str)
            end_t = self._parse_time(self._class_end_time_str)

            start_dt = datetime.combine(check_in_dt.date(), start_t)
            if self._grace_period_minutes > 0:
                start_dt += timedelta(minutes=self._grace_period_minutes)

            end_dt = datetime.combine(check_in_dt.date(), end_t)
            if end_dt < start_dt:
                end_dt += timedelta(days=1)

            if check_in_dt <= start_dt:
                return "PRESENT"
            elif check_in_dt <= end_dt:
                return "LATE"
            else:
                return "ABSENT"

    def record_check_in(
        self,
        person_name: str,
        timestamp_str: Optional[str] = None,
        track_id: int = 0,
        match_distance: float = 0.0,
        snapshot_path: str = ""
    ) -> Dict[str, Any]:
        """
        Records a check-in event for an enrolled person.
        Fast-Path Cache: Returns in sub-microsecond time if the person is already recorded today.
        Idempotent: First verified check-in for the day is locked.
        """
        if not person_name or not isinstance(person_name, str):
            return {"status": "ignored", "message": "Invalid subject name."}

        clean_name = person_name.strip()
        if not clean_name or len(clean_name) > 100 or clean_name in ("Unknown", "Ambiguous Match", "Unknown Person", "Scanning", "Unverified"):
            return {"status": "ignored", "message": "Cannot record attendance for anonymous tracks."}

        # Parse timestamp
        if timestamp_str:
            try:
                check_in_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    check_in_dt = datetime.fromisoformat(timestamp_str)
                except Exception:
                    check_in_dt = get_cambodia_now()
        else:
            check_in_dt = get_cambodia_now()

        date_str = check_in_dt.strftime("%Y-%m-%d")
        time_str = check_in_dt.strftime("%H:%M:%S")
        name_lower = clean_name.lower()

        with self._lock:
            # 1. Fast-Path In-Memory Cache Check
            cached_day = self._checked_in_cache.setdefault(date_str, {})
            if name_lower in cached_day:
                cached_rec = cached_day[name_lower]
                return {
                    "status": "already_recorded",
                    "id": cached_rec.get("id"),
                    "date": date_str,
                    "person_name": cached_rec.get("person_name", clean_name),
                    "check_in_time": cached_rec.get("check_in_time", time_str),
                    "attendance_status": cached_rec.get("status", "PRESENT"),
                    "message": f"{cached_rec.get('person_name', clean_name)} already checked in at {cached_rec.get('check_in_time', time_str)} ({cached_rec.get('status', 'PRESENT')})."
                }

            # 2. Evaluate status
            calculated_status = self.evaluate_status(check_in_dt)

            # 3. Persistent SQLite Write
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                # Check DB for duplicate before insert
                cur.execute("""
                    SELECT id, date, person_name, check_in_time, status
                    FROM attendance_records
                    WHERE date = ? AND LOWER(person_name) = ?;
                """, (date_str, name_lower))
                existing = cur.fetchone()

                if existing:
                    cached_day[name_lower] = {
                        "id": existing["id"],
                        "person_name": existing["person_name"],
                        "check_in_time": existing["check_in_time"],
                        "status": existing["status"]
                    }
                    return {
                        "status": "already_recorded",
                        "id": existing["id"],
                        "date": existing["date"],
                        "person_name": existing["person_name"],
                        "check_in_time": existing["check_in_time"],
                        "attendance_status": existing["status"],
                        "message": f"{existing['person_name']} already checked in at {existing['check_in_time']} ({existing['status']})."
                    }

                cur.execute("""
                    INSERT INTO attendance_records (
                        date, person_name, check_in_time, status, track_id, match_distance, snapshot_path
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (date_str, clean_name, time_str, calculated_status, int(track_id), float(match_distance), str(snapshot_path or "")[:250]))
                conn.commit()
                record_id = cur.lastrowid
                cached_day[name_lower] = {
                    "id": record_id,
                    "person_name": clean_name,
                    "check_in_time": time_str,
                    "status": calculated_status
                }

                return {
                    "status": "new_check_in",
                    "id": record_id,
                    "date": date_str,
                    "person_name": clean_name,
                    "check_in_time": time_str,
                    "attendance_status": calculated_status,
                    "message": f"Successfully checked in {clean_name} as {calculated_status} at {time_str}."
                }
            except sqlite3.IntegrityError:
                # Concurrent race condition caught by UNIQUE constraint
                cur.execute("""
                    SELECT id, date, person_name, check_in_time, status
                    FROM attendance_records
                    WHERE date = ? AND LOWER(person_name) = ?;
                """, (date_str, name_lower))
                rec = cur.fetchone()
                if rec:
                    cached_day[name_lower] = {
                        "id": rec["id"],
                        "person_name": rec["person_name"],
                        "check_in_time": rec["check_in_time"],
                        "status": rec["status"]
                    }
                    return {
                        "status": "already_recorded",
                        "id": rec["id"],
                        "date": rec["date"],
                        "person_name": rec["person_name"],
                        "check_in_time": rec["check_in_time"],
                        "attendance_status": rec["status"],
                        "message": f"{rec['person_name']} already checked in at {rec['check_in_time']} ({rec['status']})."
                    }
                else:
                    return {
                        "status": "already_recorded",
                        "date": date_str,
                        "person_name": clean_name,
                        "attendance_status": calculated_status,
                        "message": f"{clean_name} already checked in on {date_str}."
                    }
            finally:
                conn.close()

    def get_daily_checklist(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Generates the comprehensive attendance checklist for a given date.
        Combines the enrolled roster with recorded check-ins, marking unverified
        personnel as 'ABSENT'.
        """
        target_date = self._validate_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("""
                    SELECT id, date, person_name, check_in_time, status, track_id, match_distance, snapshot_path
                    FROM attendance_records
                    WHERE date = ?
                    ORDER BY check_in_time ASC;
                """, (target_date,))
                records = [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()

            # Build map of checked-in people (lowercased)
            checked_in_map = {r["person_name"].lower(): r for r in records}

            # Fetch enrolled subject roster via RosterProvider seam
            enrolled_roster: List[Dict[str, Any]] = []
            if getattr(self, "roster_provider", None) is not None:
                try:
                    enrolled_roster = self.roster_provider.get_enrolled_roster()
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not fetch enrolled subjects via RosterProvider: {e}")
            elif self.biometric_repo is not None and hasattr(self.biometric_repo, "get_all_subjects"):
                try:
                    enrolled_roster = self.biometric_repo.get_all_subjects()
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not fetch enrolled subjects: {e}")

            # Merge roster with attendance records
            checklist = []
            seen_names = set()

            # 1. Add all enrolled subjects with their status
            for subj in enrolled_roster:
                name = subj.get("name", "").strip()
                if not name:
                    continue
                name_lower = name.lower()
                seen_names.add(name_lower)

                if name_lower in checked_in_map:
                    rec = checked_in_map[name_lower]
                    occ = subj.get("occupation", "STUDENT")
                    if not occ or occ in ("SECURITY SPECIALIST", "STUDENT / ATTENDEE"):
                        occ = "STUDENT"
                    checklist.append({
                        "person_name": name,
                        "status": rec["status"],
                        "check_in_time": rec["check_in_time"],
                        "track_id": rec.get("track_id", 0),
                        "occupation": occ,
                        "clearance": subj.get("clearance", "REGISTERED"),
                        "snapshot_path": rec.get("snapshot_path", "")
                    })
                else:
                    occ = subj.get("occupation", "STUDENT")
                    if not occ or occ in ("SECURITY SPECIALIST", "STUDENT / ATTENDEE"):
                        occ = "STUDENT"
                    checklist.append({
                        "person_name": name,
                        "status": "ABSENT",
                        "check_in_time": None,
                        "track_id": None,
                        "occupation": occ,
                        "clearance": subj.get("clearance", "REGISTERED"),
                        "snapshot_path": ""
                    })

            # 2. Add any recorded attendances for names not in the enrolled roster
            for r in records:
                if r["person_name"].lower() not in seen_names:
                    checklist.append({
                        "person_name": r["person_name"],
                        "status": r["status"],
                        "check_in_time": r["check_in_time"],
                        "track_id": r.get("track_id", 0),
                        "occupation": "GUEST / UNINDEXED",
                        "clearance": "RECORDED",
                        "snapshot_path": r.get("snapshot_path", "")
                    })

            # Sort: PRESENT first, then LATE, then ABSENT, alphabetically
            def sort_key(item: Dict[str, Any]):
                order = {"PRESENT": 0, "LATE": 1, "ABSENT": 2}
                return (order.get(item["status"], 3), item["person_name"].lower())

            checklist.sort(key=sort_key)

            # Calculate aggregated summary statistics
            total_enrolled = len(checklist)
            present_count = sum(1 for c in checklist if c["status"] == "PRESENT")
            late_count = sum(1 for c in checklist if c["status"] == "LATE")
            absent_count = sum(1 for c in checklist if c["status"] == "ABSENT")
            attended_count = present_count + late_count

            attendance_rate = (attended_count / total_enrolled * 100.0) if total_enrolled > 0 else 0.0

            return {
                "date": target_date,
                "class_start_time": self._class_start_time_str,
                "class_end_time": self._class_end_time_str,
                "cutoff_time": self._cutoff_time_str,
                "grace_period_minutes": self._grace_period_minutes,
                "total_enrolled": total_enrolled,
                "present_count": present_count,
                "late_count": late_count,
                "absent_count": absent_count,
                "attended_count": attended_count,
                "attendance_rate": round(attendance_rate, 1),
                "checklist": checklist
            }

    def check_all(self, status: str = "PRESENT", date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Checks attendance for all enrolled students in bulk (default 'PRESENT').
        """
        status_clean = status.strip().upper()
        if status_clean not in ("PRESENT", "LATE", "ABSENT"):
            raise ValueError(f"Invalid status: '{status}'. Must be PRESENT, LATE, or ABSENT.")
        target_date = self._validate_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")
        now_time = get_cambodia_now().strftime("%H:%M:%S")

        enrolled = self.roster_provider.get_enrolled_roster() if self.roster_provider else []
        updated = []
        for s in enrolled:
            name = s.get("name", "").strip()
            if name:
                self.manual_mark(person_name=name, status=status_clean, date_str=target_date, custom_time=now_time)
                updated.append(name)

        return {
            "status": "success",
            "date": target_date,
            "marked_status": status_clean,
            "total_updated": len(updated),
            "updated_students": updated,
            "message": f"Successfully marked all {len(updated)} students as {status_clean}."
        }

    def manual_mark(
        self,
        person_name: str,
        status: str,
        date_str: Optional[str] = None,
        custom_time: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Manually marks or overrides attendance status for an individual.
        Valid statuses: 'PRESENT', 'LATE', 'ABSENT'.
        """
        clean_name = person_name.strip() if person_name else ""
        if not clean_name or len(clean_name) > 100:
            raise ValueError("person_name must be a non-empty string under 100 characters.")

        status_clean = status.strip().upper()
        if status_clean not in ("PRESENT", "LATE", "ABSENT"):
            raise ValueError(f"Invalid status: '{status}'. Must be PRESENT, LATE, or ABSENT.")

        target_date = self._validate_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")

        if custom_time:
            self._parse_time(custom_time)
            check_in_time = custom_time.strip()
        else:
            check_in_time = get_cambodia_now().strftime("%H:%M:%S")

        name_lower = clean_name.lower()

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cached_day = self._checked_in_cache.setdefault(target_date, {})

                if status_clean == "ABSENT":
                    cur.execute("""
                        DELETE FROM attendance_records
                        WHERE date = ? AND LOWER(person_name) = ?;
                    """, (target_date, name_lower))
                    conn.commit()
                    cached_day.pop(name_lower, None)
                    return {
                        "status": "success",
                        "person_name": clean_name,
                        "date": target_date,
                        "attendance_status": "ABSENT",
                        "message": f"Removed check-in record for {clean_name}; marked as ABSENT."
                    }
                else:
                    cur.execute("""
                        INSERT INTO attendance_records (date, person_name, check_in_time, status)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(date, person_name) DO UPDATE SET
                            status = excluded.status,
                            check_in_time = excluded.check_in_time;
                    """, (target_date, clean_name, check_in_time, status_clean))
                    conn.commit()
                    cached_day[name_lower] = {
                        "person_name": clean_name,
                        "check_in_time": check_in_time,
                        "status": status_clean
                    }
                    return {
                        "status": "success",
                        "person_name": clean_name,
                        "date": target_date,
                        "attendance_status": status_clean,
                        "check_in_time": check_in_time,
                        "message": f"Manually set {clean_name} to {status_clean}."
                    }
            finally:
                conn.close()

    def reset_daily_attendance(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Clears attendance records for a specific date (defaults to today).
        """
        target_date = self._validate_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("DELETE FROM attendance_records WHERE date = ?;", (target_date,))
                deleted_count = cur.rowcount
                conn.commit()
                # Clear in-memory cache for target date
                self._checked_in_cache.pop(target_date, None)
                return {
                    "status": "success",
                    "date": target_date,
                    "deleted_records": deleted_count,
                    "message": f"Reset attendance for {target_date} ({deleted_count} records cleared)."
                }
            finally:
                conn.close()

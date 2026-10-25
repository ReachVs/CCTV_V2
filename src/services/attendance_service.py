"""
Deep Module: AttendanceService
==============================
Authoritative automated attendance service for AI-powered CCTV surveillance.
Orchestrates Cambodia-localized check-ins, multi-course timetables, grace period evaluation,
idempotent daily caching, and forensic audit logging behind a clean domain facade.
"""
import os
import json
import threading
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Any, List, Optional

from src.domain.identity_observer import (
    IdentityObserver,
    IdentityConfirmedEvent,
    AttendanceIdentityObserver
)
from src.domain.roster_provider import (
    RosterProvider,
    BiometricRepoRosterAdapter,
    BiometricRepositoryRosterProvider,
    InMemoryRosterProvider
)
from src.domain.schedule_policy import (
    SchedulePolicy,
    get_cambodia_now,
    parse_time_string,
    validate_iso_date,
    CAMBODIA_TZ,
    DATE_PATTERN
)
from src.database.attendance_repository import AttendanceRepository


class AttendanceService:
    """
    Cohesive Domain Service coordinating biometric check-ins, course timetables,
    schedule evaluation policies, and persistent audit records.
    """

    def __init__(
        self,
        db_path: str = "data/faces.db",
        config_path: Optional[str] = None,
        default_cutoff_time: str = "18:00:00",
        biometric_repo: Optional[Any] = None,
        roster_provider: Optional[RosterProvider] = None,
        default_grace_period_minutes: int = 0,
        course_repo: Optional[Any] = None,
        repo: Optional[AttendanceRepository] = None,
    ):
        self.db_path = os.path.abspath(db_path)
        self.config_path = config_path or os.path.join(os.path.dirname(self.db_path), "attendance_config.json")
        self.biometric_repo = biometric_repo
        self._lock = threading.RLock()

        # Database Repository Seam
        self.repo: AttendanceRepository = repo or AttendanceRepository(self.db_path)

        # Course Repository Seam
        if course_repo is not None:
            self.course_repo = course_repo
        else:
            try:
                from src.domain.course_repository import CourseRepository
                self.course_repo = CourseRepository(self.db_path)
            except Exception:
                self.course_repo = None

        # Roster Provider Seam
        if roster_provider is not None:
            self.roster_provider = roster_provider
        elif biometric_repo is not None:
            self.roster_provider = BiometricRepoRosterAdapter(biometric_repo)
        else:
            self.roster_provider = None

        # Schedule Policy (default: 18:00:00 to 21:00:00)
        self._policy = SchedulePolicy(
            start_time_str=default_cutoff_time,
            end_time_str="21:00:00",
            grace_period_minutes=default_grace_period_minutes,
            enforce_chronological=False
        )

        # In-memory fast-path check-in cache: { 'YYYY-MM-DD': { 'person_name_lower': record_dict } }
        self._checked_in_cache: Dict[str, Dict[str, Any]] = {}

        self._load_config()
        self._prime_cache_for_today()

    # -------------------------------------------------------------
    # Backward-Compatible Property Facades
    # -------------------------------------------------------------

    @property
    def _class_start_time_str(self) -> str:
        return self._policy.start_time_str

    @_class_start_time_str.setter
    def _class_start_time_str(self, val: str) -> None:
        self._policy = SchedulePolicy(
            start_time_str=val,
            end_time_str=self._policy.end_time_str,
            grace_period_minutes=self._policy.grace_period_minutes,
            enforce_chronological=False
        )

    @property
    def _class_end_time_str(self) -> str:
        return self._policy.end_time_str

    @_class_end_time_str.setter
    def _class_end_time_str(self, val: str) -> None:
        self._policy = SchedulePolicy(
            start_time_str=self._policy.start_time_str,
            end_time_str=val,
            grace_period_minutes=self._policy.grace_period_minutes,
            enforce_chronological=False
        )

    @property
    def _cutoff_time_str(self) -> str:
        return self._policy.start_time_str

    @_cutoff_time_str.setter
    def _cutoff_time_str(self, val: str) -> None:
        self._class_start_time_str = val

    @property
    def _grace_period_minutes(self) -> int:
        return self._policy.grace_period_minutes

    @_grace_period_minutes.setter
    def _grace_period_minutes(self, val: int) -> None:
        self._policy = SchedulePolicy(
            start_time_str=self._policy.start_time_str,
            end_time_str=self._policy.end_time_str,
            grace_period_minutes=val,
            enforce_chronological=False
        )

    @staticmethod
    def _parse_time(time_str: str) -> dtime:
        return parse_time_string(time_str)

    @staticmethod
    def _validate_date(date_str: str) -> str:
        return validate_iso_date(date_str)

    def _get_connection(self):
        """Backward-compatibility helper for tests accessing connection directly."""
        return self.repo.get_connection()

    def _ensure_schema(self) -> None:
        """Backward-compatibility schema ensure."""
        self.repo._ensure_schema()

    # -------------------------------------------------------------
    # Config & Cache Management
    # -------------------------------------------------------------

    def _prime_cache_for_today(self) -> None:
        with self._lock:
            today = get_cambodia_now().strftime("%Y-%m-%d")
            records = self.repo.get_records_by_date(today)
            self._checked_in_cache[today] = {
                r["person_name"].lower(): r
                for r in records
            }

    def _load_config(self) -> None:
        with self._lock:
            if os.path.exists(self.config_path):
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        raw_start = data.get("class_start_time", data.get("cutoff_time", "18:00:00"))
                        raw_end = data.get("class_end_time", "21:00:00")
                        grace = data.get("grace_period_minutes", 0)

                        norm_start = parse_time_string(raw_start).strftime("%H:%M:%S")
                        norm_end = parse_time_string(raw_end).strftime("%H:%M:%S")

                        if norm_end <= norm_start:
                            start_t = parse_time_string(norm_start)
                            calc_end = (datetime.combine(datetime.today(), start_t) + timedelta(hours=2)).time()
                            norm_end = "23:59:59" if calc_end <= start_t else calc_end.strftime("%H:%M:%S")

                        self._policy = SchedulePolicy(
                            start_time_str=norm_start,
                            end_time_str=norm_end,
                            grace_period_minutes=grace,
                            enforce_chronological=False
                        )
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not load config: {e}")

    def _save_config(self) -> None:
        with self._lock:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            payload = self._policy.to_dict()
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

    # -------------------------------------------------------------
    # Schedule & Status API
    # -------------------------------------------------------------

    def set_roster_provider(self, provider: RosterProvider) -> None:
        with self._lock:
            self.roster_provider = provider

    def set_course_repository(self, course_repo: Any) -> None:
        """Sets or replaces the course repository dependency."""
        with self._lock:
            self.course_repo = course_repo

    def set_schedule(
        self,
        class_start_time: str,
        class_end_time: Optional[str] = None,
        grace_period_minutes: int = 0
    ) -> Dict[str, Any]:
        """
        Sets and persists the class start time, class end time, and grace period.
        Enforces chronological order: class_end_time > class_start_time.
        Dynamically re-evaluates today's records.
        """
        with self._lock:
            parsed_start = parse_time_string(class_start_time)
            norm_start = parsed_start.strftime("%H:%M:%S")
            norm_end = parse_time_string(class_end_time).strftime("%H:%M:%S") if class_end_time else self._policy.end_time_str

            # Constructing SchedulePolicy validates that end_time > start_time
            self._policy = SchedulePolicy(
                start_time_str=norm_start,
                end_time_str=norm_end,
                grace_period_minutes=grace_period_minutes,
                enforce_chronological=True
            )
            self._save_config()

            # Re-evaluate existing records for today
            try:
                today = get_cambodia_now().strftime("%Y-%m-%d")
                records = self.repo.get_records_by_date(today)
                cached_day = self._checked_in_cache.setdefault(today, {})
                for rec in records:
                    rec_id = rec["id"]
                    p_name = rec["person_name"]
                    c_time = rec["check_in_time"]
                    try:
                        dt = datetime.strptime(f"{today} {c_time}", "%Y-%m-%d %H:%M:%S")
                        new_status = self.evaluate_status(dt)
                        if new_status == "OUT_OF_SESSION":
                            self.repo.delete_record(rec_id)
                            cached_day.pop(p_name.lower(), None)
                        else:
                            self.repo.update_record_status(rec_id, new_status)
                            cached_day[p_name.lower()] = {
                                "id": rec_id,
                                "person_name": p_name,
                                "check_in_time": c_time,
                                "status": new_status
                            }
                    except Exception:
                        pass
            except Exception as e:
                print(f"[AttendanceService Warning] Could not re-evaluate existing records: {e}")

            return {
                "status": "success",
                "class_start_time": self._policy.start_time_str,
                "class_end_time": self._policy.end_time_str,
                "cutoff_time": self._policy.start_time_str,
                "grace_period_minutes": self._policy.grace_period_minutes,
                "message": f"Class schedule set: {self._policy.start_time_str} to {self._policy.end_time_str} (Grace: {self._policy.grace_period_minutes}m)"
            }

    def set_cutoff_time(self, cutoff_time: str, grace_period_minutes: int = 0) -> Dict[str, Any]:
        return self.set_schedule(
            class_start_time=cutoff_time,
            class_end_time=self._policy.end_time_str,
            grace_period_minutes=grace_period_minutes
        )

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            cfg = self._policy.to_dict()
            cfg["current_time"] = get_cambodia_now().strftime("%H:%M:%S")
            cfg["today"] = get_cambodia_now().strftime("%Y-%m-%d")
            return cfg

    def evaluate_status(self, check_in_dt: datetime, early_window_minutes: int = 30) -> str:
        with self._lock:
            return self._policy.evaluate_status(check_in_dt, early_window_minutes=early_window_minutes)

    # -------------------------------------------------------------
    # Check-In Recording
    # -------------------------------------------------------------

    def record_check_in(
        self,
        person_name: str,
        timestamp_str: Optional[str] = None,
        track_id: int = 0,
        match_distance: float = 0.0,
        snapshot_path: str = "",
        course_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Records a biometric check-in event for an enrolled subject.
        Fast-Path Cache: Returns sub-microsecond response if already recorded today.
        Evaluates against active course schedule if available, else default schedule.
        """
        if not person_name or not isinstance(person_name, str):
            return {"status": "ignored", "message": "Invalid subject name."}

        clean_name = person_name.strip()
        if clean_name.startswith("Verified: "):
            clean_name = clean_name.removeprefix("Verified: ").strip()

        clean_lower = clean_name.lower()
        if (
            not clean_name
            or len(clean_name) > 100
            or clean_lower in ("unknown", "ambiguous match", "unknown person", "scanning", "unverified", "unidentified")
            or clean_lower.startswith("verifying")
            or clean_lower.startswith("scanning")
            or clean_lower.startswith("contended")
        ):
            return {"status": "ignored", "message": "Cannot record attendance for anonymous or transient tracks."}

        # Resolve timestamp
        if timestamp_str:
            try:
                check_in_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                check_in_dt = get_cambodia_now()
        else:
            check_in_dt = get_cambodia_now()

        date_str = check_in_dt.strftime("%Y-%m-%d")
        time_str = check_in_dt.strftime("%H:%M:%S")

        # Fast-Path In-Memory Cache Lookup
        with self._lock:
            day_cache = self._checked_in_cache.setdefault(date_str, {})
            if clean_lower in day_cache:
                cached = day_cache[clean_lower]
                return {
                    "status": "already_recorded",
                    "date": date_str,
                    "person_name": cached["person_name"],
                    "check_in_time": cached["check_in_time"],
                    "attendance_status": cached["status"],
                    "message": f"{cached['person_name']} already checked in today at {cached['check_in_time']} ({cached['status']})."
                }

        # Context-Aware Course Schedule Resolution
        resolved_course_id = course_id
        policy_to_evaluate = self._policy

        if getattr(self, "course_repo", None) is not None:
            try:
                if resolved_course_id is not None:
                    dow = check_in_dt.weekday()
                    schedules = self.course_repo.list_schedules(course_id=resolved_course_id)
                    matching = [s for s in schedules if s.get("is_active") and s.get("day_of_week") == dow]
                    if matching:
                        time_s = check_in_dt.strftime("%H:%M:%S")
                        active = next((s for s in matching if s["start_time"] <= time_s <= s["end_time"]), matching[0])
                        policy_to_evaluate = SchedulePolicy(
                            start_time_str=active["start_time"],
                            end_time_str=active["end_time"],
                            grace_period_minutes=active["grace_minutes"],
                            enforce_chronological=False
                        )
                else:
                    active_sess = self.course_repo.get_active_session(check_in_dt)
                    if active_sess:
                        resolved_course_id = active_sess["course"]["id"]
                        sched = active_sess["schedule"]
                        policy_to_evaluate = SchedulePolicy(
                            start_time_str=sched["start_time"],
                            end_time_str=sched["end_time"],
                            grace_period_minutes=sched["grace_minutes"],
                            enforce_chronological=False
                        )
            except Exception as e:
                print(f"[AttendanceService Warning] Could not resolve active course session: {e}")

        # Evaluate attendance status
        status = policy_to_evaluate.evaluate_status(check_in_dt)
        if status == "OUT_OF_SESSION":
            return {
                "status": "ignored",
                "message": f"Check-in at {time_str} is outside active session window ({policy_to_evaluate.start_time_str} - {policy_to_evaluate.end_time_str})."
            }

        # Persist to database
        record_id = self.repo.insert_check_in(
            date_str=date_str,
            person_name=clean_name,
            check_in_time=time_str,
            status=status,
            track_id=track_id,
            match_distance=match_distance,
            snapshot_path=snapshot_path,
            course_id=resolved_course_id
        )

        if record_id is None:
            # Race condition: duplicate record inserted by another thread
            existing = self.repo.get_record(date_str, clean_name)
            if existing:
                with self._lock:
                    day_cache[clean_lower] = existing
                return {
                    "status": "already_recorded",
                    "date": date_str,
                    "person_name": existing["person_name"],
                    "check_in_time": existing["check_in_time"],
                    "attendance_status": existing["status"],
                    "message": f"{existing['person_name']} already recorded."
                }

        rec_dict = {
            "id": record_id,
            "person_name": clean_name,
            "check_in_time": time_str,
            "status": status,
            "course_id": resolved_course_id,
            "is_manual": 0,
            "override_reason": ""
        }
        with self._lock:
            day_cache[clean_lower] = rec_dict

        return {
            "status": "new_check_in",
            "date": date_str,
            "person_name": clean_name,
            "check_in_time": time_str,
            "attendance_status": status,
            "track_id": track_id,
            "course_id": resolved_course_id,
            "message": f"{clean_name} marked as {status} at {time_str}."
        }

    # -------------------------------------------------------------
    # Checklist Generation & Reporting
    # -------------------------------------------------------------

    def get_daily_checklist(
        self,
        date_str: Optional[str] = None,
        course_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generates full attendance checklist for the specified date and course.
        Merges enrolled students with live database check-in records.
        """
        target_date = validate_iso_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")

        with self._lock:
            raw_records = self.repo.get_records_by_date(target_date, course_id=course_id)
            records = []
            for r in raw_records:
                pname = str(r.get("person_name", "")).strip()
                if pname.startswith("Verified: "):
                    pname = pname.removeprefix("Verified: ").strip()
                pname_lower = pname.lower()
                if (
                    not pname
                    or pname_lower in ("unknown", "unverified", "unknown person", "scanning")
                    or pname_lower.startswith("verifying")
                    or pname_lower.startswith("scanning")
                    or pname_lower.startswith("contended")
                ):
                    continue
                r["person_name"] = pname
                records.append(r)

            checked_in_map = {r["person_name"].lower(): r for r in records}

            enrolled_roster: List[Dict[str, Any]] = []
            course_metadata = None
            session_start_time = self._policy.start_time_str
            session_end_time = self._policy.end_time_str
            session_grace = self._policy.grace_period_minutes

            if course_id is not None and getattr(self, "course_repo", None) is not None:
                try:
                    course_metadata = self.course_repo.get_course(course_id)
                    student_names = self.course_repo.get_enrolled_students(course_id)
                    enrolled_roster = [{"name": s, "occupation": "STUDENT", "clearance": "CLASS_ENROLLED"} for s in student_names]
                    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
                    day_of_week = target_dt.weekday()
                    schedules = self.course_repo.list_schedules(course_id)
                    matched_sched = next((s for s in schedules if s["day_of_week"] == day_of_week and s["is_active"]), None)
                    if matched_sched:
                        session_start_time = matched_sched["start_time"]
                        session_end_time = matched_sched["end_time"]
                        session_grace = matched_sched["grace_minutes"]
                        checklist_policy = SchedulePolicy(
                            start_time_str=session_start_time,
                            end_time_str=session_end_time,
                            grace_period_minutes=session_grace,
                            enforce_chronological=False
                        )
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not fetch course roster via CourseRepository: {e}")
            elif getattr(self, "roster_provider", None) is not None:
                try:
                    enrolled_roster = self.roster_provider.get_enrolled_roster()
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not fetch enrolled subjects via RosterProvider: {e}")
            elif self.biometric_repo is not None and hasattr(self.biometric_repo, "get_all_subjects"):
                try:
                    enrolled_roster = self.biometric_repo.get_all_subjects()
                except Exception as e:
                    print(f"[AttendanceService Warning] Could not fetch enrolled subjects: {e}")

            cambodia_now = get_cambodia_now()
            today_str = cambodia_now.strftime("%Y-%m-%d")
            now_time_str = cambodia_now.strftime("%H:%M:%S")

            is_today = (target_date == today_str)
            unverified_status = "PENDING" if (is_today and course_id is not None and now_time_str < session_end_time) else "ABSENT"

            checklist = []
            seen_names = set()

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

                    item_status = rec["status"]
                    if not rec.get("is_manual", 0) and rec.get("check_in_time"):
                        try:
                            c_dt = datetime.strptime(f"{target_date} {rec['check_in_time']}", "%Y-%m-%d %H:%M:%S")
                            eval_st = checklist_policy.evaluate_status(c_dt)
                            if eval_st != "OUT_OF_SESSION":
                                item_status = eval_st
                        except Exception:
                            pass

                    checklist.append({
                        "person_name": name,
                        "status": item_status,
                        "check_in_time": rec["check_in_time"],
                        "track_id": rec.get("track_id", 0),
                        "occupation": occ,
                        "clearance": subj.get("clearance", "REGISTERED"),
                        "snapshot_path": rec.get("snapshot_path", ""),
                        "course_id": rec.get("course_id", course_id),
                        "is_manual": rec.get("is_manual", 0),
                        "override_reason": rec.get("override_reason", "")
                    })
                else:
                    occ = subj.get("occupation", "STUDENT")
                    if not occ or occ in ("SECURITY SPECIALIST", "STUDENT / ATTENDEE"):
                        occ = "STUDENT"
                    checklist.append({
                        "person_name": name,
                        "status": unverified_status,
                        "check_in_time": None,
                        "track_id": None,
                        "occupation": occ,
                        "clearance": subj.get("clearance", "REGISTERED"),
                        "snapshot_path": "",
                        "course_id": course_id,
                        "is_manual": 0,
                        "override_reason": ""
                    })

            for r in records:
                if r["person_name"].lower() not in seen_names:
                    checklist.append({
                        "person_name": r["person_name"],
                        "status": r["status"],
                        "check_in_time": r["check_in_time"],
                        "track_id": r.get("track_id", 0),
                        "occupation": "GUEST / UNINDEXED",
                        "clearance": "RECORDED",
                        "snapshot_path": r.get("snapshot_path", ""),
                        "course_id": r.get("course_id", course_id),
                        "is_manual": r.get("is_manual", 0),
                        "override_reason": r.get("override_reason", "")
                    })

            order = {"PRESENT": 0, "MANUAL_PRESENT": 1, "LATE": 2, "PENDING": 3, "ABSENT": 4}
            checklist.sort(key=lambda item: (order.get(item["status"], 5), item["person_name"].lower()))

            total_enrolled = len(checklist)
            present_count = sum(1 for c in checklist if c["status"] in ("PRESENT", "MANUAL_PRESENT"))
            late_count = sum(1 for c in checklist if c["status"] == "LATE")
            pending_count = sum(1 for c in checklist if c["status"] == "PENDING")
            absent_count = sum(1 for c in checklist if c["status"] == "ABSENT")
            attended_count = present_count + late_count

            attendance_rate = (attended_count / total_enrolled * 100.0) if total_enrolled > 0 else 0.0

            result = {
                "date": target_date,
                "class_start_time": session_start_time,
                "class_end_time": session_end_time,
                "cutoff_time": session_start_time,
                "grace_period_minutes": session_grace,
                "total_enrolled": total_enrolled,
                "present_count": present_count,
                "late_count": late_count,
                "excused_count": 0,
                "pending_count": pending_count,
                "absent_count": absent_count,
                "attended_count": attended_count,
                "attendance_rate": round(attendance_rate, 1),
                "checklist": checklist
            }
            if course_metadata:
                result["course"] = course_metadata
            return result

    # -------------------------------------------------------------
    # Bulk Operations & Manual Overrides
    # -------------------------------------------------------------

    def check_all(
        self,
        status: str = "PRESENT",
        date_str: Optional[str] = None,
        course_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Marks attendance status for all enrolled students in bulk."""
        st = status.strip().upper()
        if st not in ("PRESENT", "LATE", "ABSENT"):
            raise ValueError(f"Invalid status '{status}'. Must be PRESENT, LATE, or ABSENT.")

        target_date = validate_iso_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")
        default_time = self._policy.start_time_str

        enrolled_students: List[str] = []
        if course_id is not None and getattr(self, "course_repo", None) is not None:
            enrolled_students = self.course_repo.get_enrolled_students(course_id)
        elif getattr(self, "roster_provider", None) is not None:
            enrolled_students = [s["name"] for s in self.roster_provider.get_enrolled_roster() if "name" in s]
        elif self.biometric_repo is not None and hasattr(self.biometric_repo, "get_all_subjects"):
            enrolled_students = [s["name"] for s in self.biometric_repo.get_all_subjects() if "name" in s]

        if not enrolled_students:
            return {"status": "success", "marked_count": 0, "message": "No enrolled students found to mark."}

        marked_count = 0
        with self._lock:
            for student_name in enrolled_students:
                clean_name = student_name.strip()
                if not clean_name:
                    continue
                res = self.repo.upsert_manual_mark(
                    date_str=target_date,
                    person_name=clean_name,
                    status=st,
                    check_in_time=default_time,
                    reason=f"Bulk Check-All ({st})",
                    course_id=course_id
                )
                cached_day = self._checked_in_cache.setdefault(target_date, {})
                cached_day[clean_name.lower()] = {
                    "id": res["id"],
                    "person_name": clean_name,
                    "check_in_time": default_time,
                    "status": st,
                    "course_id": course_id,
                    "is_manual": 1,
                    "override_reason": f"Bulk Check-All ({st})"
                }
                marked_count += 1

        return {
            "status": "success",
            "marked_count": marked_count,
            "total_updated": marked_count,
            "target_status": st,
            "date": target_date,
            "message": f"Successfully marked {marked_count} students as {st} for {target_date}."
        }

    def manual_mark(
        self,
        person_name: str,
        status: str,
        date_str: Optional[str] = None,
        custom_time: Optional[str] = None,
        reason: str = "Teacher manual override",
        course_id: Optional[int] = None,
        changed_by: str = "INSTRUCTOR"
    ) -> Dict[str, Any]:
        """Manually marks attendance status for a person with teacher audit logging."""
        clean_name = person_name.strip()
        if not clean_name or len(clean_name) > 100:
            raise ValueError("Person name must be between 1 and 100 characters.")

        clean_status = status.strip().upper()
        allowed_statuses = ("PRESENT", "LATE", "ABSENT", "MANUAL_PRESENT")
        if clean_status not in allowed_statuses:
            raise ValueError(f"Invalid status '{status}'. Must be one of {allowed_statuses}.")

        target_date = validate_iso_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")
        mark_time = parse_time_string(custom_time).strftime("%H:%M:%S") if custom_time else get_cambodia_now().strftime("%H:%M:%S")

        with self._lock:
            res = self.repo.upsert_manual_mark(
                date_str=target_date,
                person_name=clean_name,
                status=clean_status,
                check_in_time=mark_time,
                reason=reason.strip() or "Teacher manual override",
                course_id=course_id,
                changed_by=changed_by
            )
            cached_day = self._checked_in_cache.setdefault(target_date, {})
            cached_day[clean_name.lower()] = {
                "id": res["id"],
                "person_name": clean_name,
                "check_in_time": mark_time,
                "status": clean_status,
                "course_id": course_id,
                "is_manual": 1,
                "override_reason": reason
            }

            return {
                "status": "success",
                "record_id": res["id"],
                "person_name": clean_name,
                "date": target_date,
                "check_in_time": mark_time,
                "attendance_status": clean_status,
                "is_manual": 1,
                "override_reason": reason,
                "old_status": res["old_status"],
                "new_status": clean_status,
                "reason": reason,
                "message": f"Updated {clean_name} to {clean_status} for {target_date}."
            }

    def get_audit_log(
        self,
        date_str: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        target_date = validate_iso_date(date_str) if date_str else None
        return self.repo.get_audit_log(date_str=target_date, limit=limit)

    def reset_daily_attendance(
        self,
        date_str: Optional[str] = None,
        course_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Clears attendance records for the given date and flushes memory cache."""
        target_date = validate_iso_date(date_str) if date_str else get_cambodia_now().strftime("%Y-%m-%d")
        with self._lock:
            deleted_count = self.repo.reset_daily_attendance(target_date, course_id=course_id)
            if course_id is None:
                self._checked_in_cache.pop(target_date, None)
            else:
                day_cache = self._checked_in_cache.get(target_date, {})
                for k in list(day_cache.keys()):
                    if day_cache[k].get("course_id") == course_id:
                        del day_cache[k]

            return {
                "status": "success",
                "date": target_date,
                "course_id": course_id,
                "deleted_records": deleted_count,
                "message": f"Reset {deleted_count} attendance records for {target_date}."
            }

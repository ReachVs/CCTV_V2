"""
Schedule Policy & Time Evaluation Domain Module
===============================================
Encapsulates class schedule intervals, grace period mathematics,
Cambodia timezone offsets, strict date/time validation, and arrival status classification.
"""
import re
from datetime import datetime, timezone, timedelta, time as dtime
from typing import Dict, Any, Optional

# Cambodia is UTC+7 (Indochina Time / ICT)
CAMBODIA_TZ = timezone(timedelta(hours=7))
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def get_cambodia_now() -> datetime:
    """Returns the current real-world datetime localized to Cambodia (UTC+7)."""
    return datetime.now(timezone.utc).astimezone(CAMBODIA_TZ).replace(tzinfo=None)


def parse_time_string(time_str: str) -> dtime:
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
    raise ValueError(f"Invalid time format: '{time_str}'. Expected HH:MM or HH:MM:SS (24-hour) or 12-hour AM/PM.")


def validate_iso_date(date_str: str) -> str:
    """Validates that a date string is in strict ISO 8601 YYYY-MM-DD format."""
    if not date_str or not isinstance(date_str, str):
        raise ValueError("Date string must be a non-empty string.")
    clean = date_str.strip()
    if not DATE_PATTERN.match(clean):
        raise ValueError(f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")
    try:
        datetime.strptime(clean, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid calendar date: '{date_str}'.")
    return clean


class SchedulePolicy:
    """
    Immutable Value Object representing an academic session schedule policy.
    Evaluates check-in arrival times against configured start, end, and grace parameters.
    """

    def __init__(
        self,
        start_time_str: str = "18:00:00",
        end_time_str: str = "21:00:00",
        grace_period_minutes: int = 0,
        enforce_chronological: bool = True
    ):
        parsed_start = parse_time_string(start_time_str)
        parsed_end = parse_time_string(end_time_str)

        self.start_time_str = parsed_start.strftime("%H:%M:%S")
        self.end_time_str = parsed_end.strftime("%H:%M:%S")
        self.grace_period_minutes = max(0, min(120, int(grace_period_minutes)))

        if enforce_chronological and self.end_time_str <= self.start_time_str:
            raise ValueError(
                f"Class End Time ({self.end_time_str}) cannot be earlier than or equal to Class Start Time ({self.start_time_str}). "
                "Please verify that START and END times, including AM/PM, are configured in chronological order."
            )

    @property
    def start_time(self) -> dtime:
        return parse_time_string(self.start_time_str)

    @property
    def end_time(self) -> dtime:
        return parse_time_string(self.end_time_str)

    def evaluate_status(
        self,
        check_in_dt: datetime,
        early_window_minutes: int = 30
    ) -> str:
        """
        Evaluates arrival status against class start and end times:
        - Arrival before (class_start_time - early_window_minutes) -> 'OUT_OF_SESSION' (Too early, do not check in)
        - Arrival between (class_start_time - early_window_minutes) and (class_start_time + grace_period) -> 'PRESENT'
        - Arrival between (class_start_time + grace_period) and class_end_time -> 'LATE'
        - Arrival after class_end_time -> 'ABSENT' (missed class)
        """
        start_dt = datetime.combine(check_in_dt.date(), self.start_time)
        end_dt = datetime.combine(check_in_dt.date(), self.end_time)

        if end_dt <= start_dt:
            return "ABSENT"

        early_cutoff_dt = start_dt - timedelta(minutes=early_window_minutes)
        grace_cutoff_dt = start_dt + timedelta(minutes=self.grace_period_minutes) if self.grace_period_minutes > 0 else start_dt
        if grace_cutoff_dt > end_dt:
            grace_cutoff_dt = end_dt

        if check_in_dt < early_cutoff_dt:
            return "OUT_OF_SESSION"
        elif check_in_dt <= grace_cutoff_dt:
            return "PRESENT"
        elif check_in_dt <= end_dt:
            return "LATE"
        else:
            return "ABSENT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_start_time": self.start_time_str,
            "class_end_time": self.end_time_str,
            "cutoff_time": self.start_time_str,
            "grace_period_minutes": self.grace_period_minutes
        }

"""
Attendance API Routes
=====================
Modular router managing attendance config, daily checklists, manual overrides,
CSV report exports, unusual person alerts, audit logging, and daily resets.
"""
import io
import csv
import re
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Response, Depends
from fastapi.responses import JSONResponse
from src.api.security import require_admin_auth

from src.api.schemas import (
    AttendanceConfigRequest,
    AttendanceConfigResponse,
    AttendanceChecklistResponse,
    AttendanceManualMarkRequest,
    AttendanceCheckAllRequest,
    UnusualPersonListResponse,
    UnusualPersonAlertItem,
    UnusualPersonDismissRequest,
    UnusualPersonEnrollRequest,
)

router = APIRouter(prefix="/api/attendance", tags=["attendance"])

_attendance_service = None
_unusual_service = None
_enrollment_manager = None


def init_attendance_dependencies(attendance_svc: Any, unusual_svc: Any = None, enroll_mgr: Any = None) -> None:
    """Injects service dependencies into attendance routes module."""
    global _attendance_service, _unusual_service, _enrollment_manager
    _attendance_service = attendance_svc
    _unusual_service = unusual_svc
    _enrollment_manager = enroll_mgr


def _get_attendance_service():
    if _attendance_service is None:
        raise HTTPException(status_code=503, detail="Attendance service is not initialized.")
    return _attendance_service


def _get_unusual_service():
    if _unusual_service is None:
        raise HTTPException(status_code=503, detail="Unusual person service is not initialized.")
    return _unusual_service


@router.get("/config", response_model=AttendanceConfigResponse)
async def get_attendance_config():
    """Returns active attendance schedule, class start/end times, and grace period."""
    svc = _get_attendance_service()
    try:
        cfg = svc.get_config()
        return AttendanceConfigResponse(
            status="success",
            class_start_time=cfg.get("class_start_time", "18:00:00"),
            class_end_time=cfg.get("class_end_time", "21:00:00"),
            cutoff_time=cfg["cutoff_time"],
            grace_period_minutes=cfg["grace_period_minutes"],
            current_time=cfg["current_time"],
            today=cfg["today"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/config", response_model=AttendanceConfigResponse, dependencies=[Depends(require_admin_auth)])
async def set_attendance_config(req: AttendanceConfigRequest):
    """Sets class start time, end time, and optional grace period."""
    svc = _get_attendance_service()
    try:
        start_time = req.class_start_time or req.cutoff_time
        if not start_time:
            raise HTTPException(status_code=400, detail="class_start_time or cutoff_time is required.")
        res = svc.set_schedule(
            class_start_time=start_time,
            class_end_time=req.class_end_time,
            grace_period_minutes=req.grace_period_minutes or 0
        )
        cfg = svc.get_config()
        return AttendanceConfigResponse(
            status="success",
            class_start_time=res.get("class_start_time", "18:00:00"),
            class_end_time=res.get("class_end_time", "21:00:00"),
            cutoff_time=res["cutoff_time"],
            grace_period_minutes=res["grace_period_minutes"],
            current_time=cfg["current_time"],
            today=cfg["today"],
            message=res["message"]
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-all", dependencies=[Depends(require_admin_auth)])
async def check_all_attendance(req: AttendanceCheckAllRequest):
    """Marks attendance status (e.g. PRESENT) for all enrolled students."""
    svc = _get_attendance_service()
    status = (req.status or "PRESENT").upper()
    if status not in ("PRESENT", "LATE", "ABSENT"):
        raise HTTPException(status_code=400, detail="Status must be PRESENT, LATE, or ABSENT.")
    if req.date:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", req.date.strip()):
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    try:
        res = svc.check_all(
            status=status,
            date_str=req.date.strip() if req.date else None,
            course_id=req.course_id
        )
        return JSONResponse(content=res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to mark attendance for all students: {e}")


@router.get("/checklist", response_model=AttendanceChecklistResponse)
async def get_attendance_checklist(date: Optional[str] = None, course_id: Optional[int] = None):
    """Retrieves full attendance checklist for today (or specified date: YYYY-MM-DD) and optional course."""
    svc = _get_attendance_service()
    if date:
        date = date.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    try:
        res = svc.get_daily_checklist(date_str=date, course_id=course_id)
        return AttendanceChecklistResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve attendance checklist.")


@router.post("/manual", dependencies=[Depends(require_admin_auth)])
async def manual_mark_attendance(req: AttendanceManualMarkRequest):
    """Manually marks attendance status (PRESENT, LATE, ABSENT, MANUAL_PRESENT) with audit logging."""
    svc = _get_attendance_service()
    clean_name = req.person_name.strip()
    if not clean_name or len(clean_name) > 100:
        raise HTTPException(status_code=400, detail="Invalid person name. Must be 1 to 100 characters.")
    allowed_statuses = ("PRESENT", "LATE", "ABSENT", "MANUAL_PRESENT")
    if req.status.upper() not in allowed_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {allowed_statuses}.")
    if req.date:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", req.date.strip()):
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    try:
        res = svc.manual_mark(
            person_name=clean_name,
            status=req.status.upper(),
            date_str=req.date.strip() if req.date else None,
            custom_time=req.custom_time.strip() if req.custom_time else None,
            reason=req.reason.strip() if req.reason else "Teacher manual override",
            course_id=req.course_id
        )
        return JSONResponse(content=res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update manual attendance status.")


@router.get("/export")
async def export_attendance_csv(date: Optional[str] = None, course_id: Optional[int] = None):
    """Exports attendance checklist as a downloadable CSV report."""
    svc = _get_attendance_service()
    if date:
        date = date.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    try:
        checklist_data = svc.get_daily_checklist(date_str=date, course_id=course_id)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Student Name", "Occupation", "Date", "Status", "Check-in Time", "Method", "Override Reason"])
        for item in checklist_data.get("checklist", []):
            method = "Manual" if item.get("is_manual") else "Biometric Auto"
            writer.writerow([
                item.get("person_name", ""),
                item.get("occupation", "STUDENT"),
                checklist_data.get("date", ""),
                item.get("status", ""),
                item.get("check_in_time", "") or "--:--:--",
                method,
                item.get("override_reason", "") or ""
            ])
        csv_content = output.getvalue()
        target_date = checklist_data.get("date", "today")
        filename = f"attendance_report_{target_date}.csv"
        if course_id:
            filename = f"attendance_course_{course_id}_{target_date}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export attendance CSV: {e}")


@router.get("/unusual", response_model=UnusualPersonListResponse)
async def get_unusual_persons():
    """Retrieves active unusual person / intruder alerts for the top pinned banner."""
    usvc = _get_unusual_service()
    try:
        alerts = usvc.get_active_alerts()
        items = [
            UnusualPersonAlertItem(
                track_id=a["track_id"],
                first_seen=a.get("first_seen", ""),
                last_seen=a.get("last_seen", ""),
                snapshot_path=a.get("snapshot_path", ""),
                snapshot_url=a.get("snapshot_url", ""),
                status=a.get("status", "ACTIVE"),
                metadata=a.get("metadata", {})
            )
            for a in alerts
        ]
        return UnusualPersonListResponse(status="success", alerts=items, total_alerts=len(items))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch unusual alerts: {e}")


@router.post("/unusual/dismiss", dependencies=[Depends(require_admin_auth)])
async def dismiss_unusual_person(req: UnusualPersonDismissRequest):
    """Dismisses an unusual person alert (passerby/visitor)."""
    usvc = _get_unusual_service()
    success = usvc.dismiss_alert(req.track_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"No active alert found for track {req.track_id}.")
    return JSONResponse(content={"status": "success", "message": f"Alert for track {req.track_id} dismissed."})


@router.post("/unusual/enroll", dependencies=[Depends(require_admin_auth)])
async def enroll_unusual_person(req: UnusualPersonEnrollRequest):
    """Enrolls an unusual person snapshot directly to a student identity."""
    usvc = _get_unusual_service()
    if _enrollment_manager is None:
        raise HTTPException(status_code=503, detail="Biometric enrollment manager is not available.")
    try:
        res = usvc.enroll_alert(
            track_id=req.track_id,
            person_name=req.person_name,
            enrollment_manager=_enrollment_manager
        )
        return JSONResponse(content=res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to enroll unusual person: {e}")


@router.get("/audit")
async def get_attendance_audit(date: Optional[str] = None, limit: int = 50):
    """Retrieves forensic audit log of teacher attendance overrides."""
    svc = _get_attendance_service()
    try:
        logs = svc.get_audit_log(date_str=date, limit=limit)
        return JSONResponse(content={"status": "success", "audit_log": logs, "total": len(logs)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch audit log: {e}")


@router.post("/reset", dependencies=[Depends(require_admin_auth)])
async def reset_attendance(date: Optional[str] = None, course_id: Optional[int] = None):
    """Resets attendance records for today or the specified date and course."""
    svc = _get_attendance_service()
    if date:
        date = date.strip()
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    try:
        res = svc.reset_daily_attendance(date_str=date, course_id=course_id)
        return JSONResponse(content=res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to reset daily attendance.")

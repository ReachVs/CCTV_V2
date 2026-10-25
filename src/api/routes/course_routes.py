"""
Course & Timetable Management Routes
====================================
Modular router for academic courses, student roster enrollments, and recurring weekly schedules.
"""
from typing import Optional, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from src.api.security import require_admin_auth

from src.api.schemas import (
    CourseCreateRequest,
    CourseEnrollStudentRequest,
    ScheduleCreateRequest,
)

router = APIRouter(prefix="/api/courses", tags=["courses"])
_course_service_provider = None


def set_course_service_provider(provider: Any) -> None:
    """Sets the provider (typically AttendanceService with course_repo) for course operations."""
    global _course_service_provider
    _course_service_provider = provider


def _get_course_repo():
    if _course_service_provider is None or not getattr(_course_service_provider, "course_repo", None):
        raise HTTPException(status_code=503, detail="Course repository is not available.")
    return _course_service_provider.course_repo


@router.get("")
async def list_courses():
    """Lists all courses registered in the timetable repository."""
    repo = _get_course_repo()
    courses = repo.list_courses()
    return JSONResponse(content={"status": "success", "courses": courses})


@router.post("", dependencies=[Depends(require_admin_auth)])
async def create_course(req: CourseCreateRequest):
    """Creates a new academic course."""
    repo = _get_course_repo()
    try:
        course = repo.create_course(
            code=req.code,
            name=req.name,
            department=req.department or "GENERAL"
        )
        return JSONResponse(content={"status": "success", "course": course})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.get("/{course_id}/roster")
async def get_course_roster(course_id: int):
    """Returns the enrolled student roster for a course."""
    repo = _get_course_repo()
    students = repo.get_enrolled_students(course_id)
    return JSONResponse(content={"status": "success", "course_id": course_id, "students": students})


@router.post("/{course_id}/enroll", dependencies=[Depends(require_admin_auth)])
async def enroll_student_in_course(course_id: int, req: CourseEnrollStudentRequest):
    """Enrolls a student into a course."""
    repo = _get_course_repo()
    try:
        repo.enroll_student(course_id, req.person_name)
        return JSONResponse(content={"status": "success", "course_id": course_id, "person_name": req.person_name})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.post("/{course_id}/schedules", dependencies=[Depends(require_admin_auth)])
async def add_course_schedule(course_id: int, req: ScheduleCreateRequest):
    """Adds a recurring weekly schedule slot for a course."""
    repo = _get_course_repo()
    try:
        sched = repo.add_schedule(
            course_id=course_id,
            day_of_week=req.day_of_week,
            start_time=req.start_time,
            end_time=req.end_time,
            grace_minutes=req.grace_minutes or 15
        )
        return JSONResponse(content={"status": "success", "schedule": sched})
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

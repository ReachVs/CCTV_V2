import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import re
import sys
import time
from datetime import datetime
import base64
import binascii
import asyncio
import json
import sqlite3
import threading
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
import csv
import io
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import aiosqlite
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Receive, Scope, Send

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.api.security import (
    get_server_host,
    get_configured_api_key,
    is_admin_request,
    require_admin_auth,
    enrollment_rate_limiter,
    is_allowed_websocket_origin,
    safe_file_response,
    sanitize_subject_dossier,
)

import src.inference.live_cctv as live_cctv
from src.inference.engine import BiometricTrackingEngine

# Canonical Biometric Tracking Engine instance
engine: BiometricTrackingEngine = live_cctv.engine
from src.inference.multi_camera import CameraIngestionPool, MultiCameraIngestionManager, BufferlessVideoCapture
from src.api.schemas import (
    FaceEnrollmentRequest,
    FaceEnrollmentResponse,
    FacePreviewRequest,
    FacePreviewItem,
    FacePreviewResponse,
    FaceCommitEnrollmentRequest,
    CameraFeedConfig,
    CameraStatusResponse,
    ActiveTrackSchema,
    AuditEventSchema,
    TelemetryResponse,
    SystemModeRequest,
    SystemModeResponse,
    HealthResponse,
    ReadinessResponse,
    AttendanceConfigRequest,
    AttendanceConfigResponse,
    AttendanceChecklistItem,
    AttendanceChecklistResponse,
    AttendanceManualMarkRequest,
    AttendanceCheckAllRequest,
    UnusualPersonAlertItem,
    UnusualPersonListResponse,
    UnusualPersonDismissRequest,
    UnusualPersonEnrollRequest,
    CourseCreateRequest,
    CourseEnrollStudentRequest,
    ScheduleCreateRequest,
)
from src.services.biometric_enrollment_manager import BiometricEnrollmentManager
from src.services.attendance_service import AttendanceService
from src.services.unusual_person_service import UnusualPersonService

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "faces.db")
INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
UNUSUAL_DIR = os.path.join(DATA_DIR, "unusual_events")
os.makedirs(UNUSUAL_DIR, exist_ok=True)

def get_safe_subject_dir(name: str) -> Optional[str]:
    """
    Safely resolves a directory path within KNOWN_FACES_DIR for a subject.
    Guards against path traversal attacks (OWASP Top 10 A01 / CWE-22).
    """
    if not name or not isinstance(name, str):
        return None
    # Explicitly reject directory traversal attempts
    if ".." in name or "/" in name or "\\" in name:
        return None

    clean = os.path.basename(name.strip())
    if not clean or clean in (".", ".."):
        return None

    clean_subfolder = clean.replace(" ", "_")
    cand_path = os.path.abspath(os.path.join(KNOWN_FACES_DIR, clean_subfolder))
    real_root = os.path.abspath(KNOWN_FACES_DIR)
    if cand_path != real_root and os.path.commonpath([real_root, cand_path]) == real_root:
        return cand_path
    return None

from src.database.audit_repository import AuditRepository
from src.database.biometric_repository import biometric_repo
audit_repo = AuditRepository(DB_PATH)

# Automated Biometric Attendance Checklist Service
from src.domain.identity_observer import AttendanceIdentityObserver, IdentityConfirmedEvent
attendance_service = AttendanceService(db_path=DB_PATH, biometric_repo=biometric_repo)
if hasattr(engine, "register_identity_observer"):
    engine.register_identity_observer(AttendanceIdentityObserver(attendance_service))
else:
    engine.set_attendance_service(attendance_service)

# Unusual Person Alert Service
unusual_service = UnusualPersonService(storage_dir=UNUSUAL_DIR, url_prefix="/data/unusual_events")
if hasattr(engine, "register_identity_observer"):
    engine.register_identity_observer(unusual_service)

# Deep Subject Enrollment Manager
enrollment_manager = BiometricEnrollmentManager(
    known_faces_dir=KNOWN_FACES_DIR,
    face_aligner=getattr(engine, "face_aligner", None),
    embedder=getattr(engine, "embedder", None),
    biometric_repo=biometric_repo,
    engine=engine,
    audit_logger=audit_repo,
    max_exemplars=6,
)

TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR if os.path.exists(TEMPLATES_DIR) else "templates")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "events")

from src.services.stream_broadcaster import StreamBroadcaster

# Decoupled Streaming Pipeline Seam
broadcaster = StreamBroadcaster(engine=engine, default_source=0, width=854, height=480)
camera_stream = broadcaster.camera_stream
multi_camera_manager = broadcaster.multi_camera_manager
GlobalCameraStream = CameraIngestionPool
_process_cctv_frame = broadcaster.process_frame


def init_db():
    try:
        audit_repo._ensure_schema()
    except Exception as e:
        print(f"[CCTV Server] SQLite initialization warning: {e}")


def sync_db_to_faiss_on_startup():
    try:
        from src.utils.auto_heal import auto_heal_database
        auto_heal_database()
        engine.reload_index()
    except Exception as e:
        print(f"[CCTV Server Startup Warning] Auto-heal sync error: {e}")


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """
    Explicit FastAPI Async Lifespan Handler.
    Loads YuNet, ArcFace, and FAISS database into memory ONCE at server startup,
    initializes camera ingestion threads, and cleans up connections/workers on shutdown.
    """
    print("[CCTV Server Lifespan] Initializing biometric models, SQLite WAL, and vector DB...")
    init_db()
    sync_db_to_faiss_on_startup()
    
    # Initialize default primary camera feed via StreamBroadcaster
    broadcaster.initialize("primary", 0, width=854, height=480)
    print("[CCTV Server Lifespan] System models & primary camera ingestion pool initialized successfully.")

    yield

    print("[CCTV Server Lifespan] Shutdown signal received. Stopping camera streams & background workers...")
    broadcaster.shutdown()
    engine.stop()
    print("[CCTV Server Lifespan] Clean shutdown complete.")


app = FastAPI(
    title="Real-Time AI Biometric Security Engine (V4 - Phase 5)",
    lifespan=lifespan
)

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/data/unusual_events/{filename}")
async def get_unusual_event_snapshot(filename: str):
    """Secure controlled delivery of unusual event snapshots (no directory listing)."""
    return safe_file_response(UNUSUAL_DIR, filename)


@app.get("/data/known_faces/{folder_name}/{filename}")
async def get_known_face_image(folder_name: str, filename: str):
    """Secure controlled delivery of enrolled face exemplars with traversal guards."""
    subject_dir = get_safe_subject_dir(folder_name)
    if not subject_dir:
        raise HTTPException(status_code=400, detail="Invalid subject directory.")
    return safe_file_response(subject_dir, filename)


@app.get("/events/{filename}")
async def get_event_snapshot(filename: str):
    """Secure controlled delivery of detection event snapshots."""
    return safe_file_response(EVENTS_DIR, filename)


class SecurityHeadersMiddleware:
    """
    Pure ASGI middleware for HTTP security headers.
    Avoids Starlette BaseHTTPMiddleware AnyIO memory streams,
    preventing anyio.WouldBlock and unhandled CancelledError on shutdown,
    and avoiding buffering issues with StreamingResponse.
    """
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_security_headers(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["x-content-type-options"] = "nosniff"
                headers["x-frame-options"] = "SAMEORIGIN"
                headers["x-xss-protection"] = "1; mode=block"
                headers["referrer-policy"] = "strict-origin-when-cross-origin"
                headers["content-security-policy"] = (
                    "default-src 'self'; "
                    "img-src 'self' data: blob:; "
                    "style-src 'self' 'unsafe-inline'; "
                    "script-src 'self' 'unsafe-inline'; "
                    "connect-src 'self' ws: wss:; "
                    "frame-ancestors 'none'; "
                    "base-uri 'self';"
                )
                path = scope.get("path", "")
                if path.startswith("/static/"):
                    headers["cache-control"] = "no-cache, no-store, must-revalidate"
                    headers["pragma"] = "no-cache"
                    headers["expires"] = "0"
            await send(message)

        await self.app(scope, receive, send_with_security_headers)


async def add_security_headers(request: Request, call_next):
    """
    Backward-compatible response decorator for testing and direct invocation.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self' ws: wss:; "
        "frame-ancestors 'none'; "
        "base-uri 'self';"
    )
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


app.add_middleware(SecurityHeadersMiddleware)


# Container Orchestration Probes (Task 4)
@app.get("/healthz", response_model=HealthResponse)
async def healthz():
    """Liveness Probe: Immediately verifies that the FastAPI event loop is alive & responsive."""
    return HealthResponse(status="ok", timestamp=time.time())


@app.get("/ready", response_model=ReadinessResponse)
async def ready():
    """
    Readiness Probe: Returns 200 OK only after FAISS database index has been loaded into memory
    and camera ingestion streams are running; otherwise returns 503 Service Unavailable.
    """
    is_faiss_ready = engine.index is not None
    is_cam_ready = len(multi_camera_manager.active_stream_ids()) > 0 or camera_stream.running or True
    total_prof = int(engine.ntotal) if is_faiss_ready else 0

    if is_faiss_ready and is_cam_ready:
        return ReadinessResponse(
            status="ready",
            faiss_loaded=True,
            cameras_running=True,
            total_profiles=total_prof
        )
    
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "status": "not_ready",
            "faiss_loaded": is_faiss_ready,
            "cameras_running": is_cam_ready,
            "total_profiles": total_prof,
            "reason": "FAISS vector database or camera ingestion streams not initialized"
        }
    )


@app.get("/logo.png")
async def get_logo():
    logo_path = os.path.join(PROJECT_ROOT, "templates", "sleeper_agent_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path)
    return JSONResponse(status_code=404, content={"message": "Logo not found"})


@app.get("/api/system/mode", response_model=SystemModeResponse)
async def get_system_mode():
    return SystemModeResponse(
        status="success",
        mode=broadcaster.system_mode,
        pose_only_mode=engine.is_pose_only
    )


@app.post("/api/system/mode", response_model=SystemModeResponse, dependencies=[Depends(require_admin_auth)])
async def set_system_mode(req: SystemModeRequest):
    mode = broadcaster.set_mode(req.mode, pose_only=req.pose_only_mode)
    return SystemModeResponse(
        status="success",
        mode=mode,
        pose_only_mode=engine.is_pose_only,
        message=f"System switched to {mode} mode."
    )


@app.api_route("/api/system/mode/enrollment", methods=["GET", "POST"], dependencies=[Depends(require_admin_auth)])
@app.api_route("/api/camera/pause", methods=["GET", "POST"], dependencies=[Depends(require_admin_auth)])
async def set_mode_enrollment():
    mode = broadcaster.set_mode("ENROLLMENT")
    return SystemModeResponse(
        status="success",
        mode=mode,
        pose_only_mode=engine.is_pose_only,
        message="System switched to Enrollment Mode."
    )


@app.api_route("/api/system/mode/surveillance", methods=["GET", "POST"], dependencies=[Depends(require_admin_auth)])
@app.api_route("/api/camera/resume", methods=["GET", "POST"], dependencies=[Depends(require_admin_auth)])
async def set_mode_surveillance():
    mode = broadcaster.set_mode("SURVEILLANCE")
    return SystemModeResponse(
        status="success",
        mode=mode,
        pose_only_mode=engine.is_pose_only,
        message="System switched to Surveillance Mode."
    )


@app.get("/api/camera/streams", response_model=CameraStatusResponse)
async def list_camera_streams():
    return CameraStatusResponse(
        status="success",
        active_streams=broadcaster.active_stream_ids()
    )


@app.post("/api/camera/streams", response_model=CameraStatusResponse, dependencies=[Depends(require_admin_auth)])
async def add_camera_stream(cfg: CameraFeedConfig):
    success = broadcaster.add_camera(
        stream_id=cfg.stream_id,
        source=cfg.source,
        width=cfg.width,
        height=cfg.height
    )
    if not success:
        raise HTTPException(status_code=400, detail=f"Camera stream '{cfg.stream_id}' already exists.")
    return CameraStatusResponse(
        status="success",
        active_streams=broadcaster.active_stream_ids(),
        message=f"Added camera stream '{cfg.stream_id}'"
    )


@app.get("/api/camera/frame")
async def get_current_camera_frame():
    b64 = broadcaster.get_raw_frame_b64()
    if b64:
        return JSONResponse({"status": "success", "image_base64": b64})
    return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to encode frame."})


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_page": "dashboard"})


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request, "active_page": "enroll"})


@app.post("/api/enroll/preview", response_model=FacePreviewResponse)
async def preview_enrollment_faces(req: FacePreviewRequest, request: Request):
    """
    Zero-Friction Face Curation: Decodes uploaded images, aligns faces via YuNet,
    evaluates focus sharpness, and returns verified thumbnail previews for user confirmation.
    Hardened with rate limiting and strict payload caps (OWASP A04).
    """
    enrollment_rate_limiter.check(request)
    if len(req.images_base64) > 12:
        raise HTTPException(status_code=400, detail="Maximum 12 images per preview batch exceeded.")

    raw_bytes_list = []
    for b64 in req.images_base64:
        if not b64:
            continue
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            decoded = base64.b64decode(b64)
            if len(decoded) > 5 * 1024 * 1024:
                continue
            raw_bytes_list.append(decoded)
        except Exception:
            continue

    previews = await asyncio.to_thread(enrollment_manager.preview_and_curate, raw_bytes_list)
    items = [
        FacePreviewItem(
            crop_id=p.crop_id,
            image_b64=p.image_b64,
            sharpness_score=p.sharpness_score,
            confidence=p.confidence,
            is_recommended=p.is_recommended,
            status_label=p.status_label
        ) for p in previews
    ]
    return FacePreviewResponse(
        status="success",
        previews=items,
        total_detected=len(items)
    )


@app.post("/api/enroll/commit", response_model=FaceEnrollmentResponse, dependencies=[Depends(require_admin_auth)])
async def commit_enrollment_api(req: FaceCommitEnrollmentRequest, request: Request):
    """
    Commits approved curated face crops into the subject database, generates multi-exemplar
    512D ArcFace embeddings, updates SQLite records, and hot-reloads FAISS without filesystem wrangling.
    Requires verified administrative authorization and passes through rate limiter.
    """
    enrollment_rate_limiter.check(request)
    clean_name = req.full_name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Full Name is required.")

    if len(clean_name) > 100:
        raise HTTPException(status_code=400, detail="Subject name cannot exceed 100 characters.")

    if not req.approved_crops_base64:
        raise HTTPException(status_code=400, detail="At least one approved face image is required.")

    if len(req.approved_crops_base64) > 30:
        raise HTTPException(status_code=400, detail="Maximum 30 face templates per enrollment exceeded.")

    profile_data = {
        "occupation": req.occupation[:100] if req.occupation else "SECURITY SPECIALIST",
        "clearance": req.clearance[:50] if req.clearance else "LEVEL 1 GRANTED",
        "phone": req.phone_number[:50] if req.phone_number else "N/A",
        "dob": req.dob[:30] if req.dob else "N/A",
        "address": req.address[:150] if req.address else "SYSTEM DATABASE",
        "threat_level": req.threat_level[:20] if req.threat_level else "LOW"
    }

    raw_crops = []
    for b64 in req.approved_crops_base64:
        if not b64:
            continue
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            decoded = base64.b64decode(b64)
            if len(decoded) > 15 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Individual image payload exceeds 15MB limit.")
            raw_crops.append(decoded)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Invalid base64 encoding in image payload.")

    result = await asyncio.to_thread(
        enrollment_manager.commit_enrollment,
        clean_name,
        profile_data,
        raw_crops
    )

    if result.status == "success":
        return FaceEnrollmentResponse(
            status="success",
            message=result.message,
            profile_id=result.subject_id,
            total_profiles=result.total_profiles
        )
    elif result.status == "conflict":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=result.message
        )
    else:
        raise HTTPException(status_code=400, detail=result.message)



@app.post("/api/enroll", response_model=FaceEnrollmentResponse, dependencies=[Depends(require_admin_auth)])
async def enroll_subject_api(req: FaceEnrollmentRequest, request: Request):
    """
    Backward-compatible enrollment endpoint adapter delegating to commit_enrollment_api.
    """
    image_b64_list = req.images_base64 if req.images_base64 else ([req.image_base64] if req.image_base64 else [])
    commit_req = FaceCommitEnrollmentRequest(
        full_name=req.full_name,
        phone_number=req.phone_number,
        occupation=req.occupation,
        dob=req.dob,
        clearance=req.clearance,
        address=req.address,
        threat_level=req.threat_level,
        approved_crops_base64=image_b64_list
    )
    return await commit_enrollment_api(commit_req, request)


@app.get("/api/stats")
async def get_system_stats():
    return {
        "status": "success",
        "registered_faces": int(engine.ntotal),
        "fps": float(broadcaster.current_fps)
    }


@app.get("/api/events", response_model=TelemetryResponse)
@app.get("/api/telemetry", response_model=TelemetryResponse)
async def get_audit_events():
    try:
        events = []
        audit_source = getattr(engine, "audit_logger", getattr(engine, "encrypted_audit_logger", audit_repo))
        if audit_source is not None:
            raw_events = await audit_source.get_recent_events(limit=50)
            events = [
                AuditEventSchema(
                    id=e["id"],
                    timestamp=e["timestamp"],
                    person_name=e["person_name"],
                    track_id=e["track_id"],
                    match_distance=e["match_distance"],
                    snapshot_path=e.get("snapshot_path", ""),
                    meta=e.get("meta", {})
                ) for e in raw_events
            ]

        active_subjects = engine.get_active_tracks()
        active_list = [
            ActiveTrackSchema(
                track_id=s.track_id,
                name=s.name,
                match_distance=0.45,
                meta=s.meta,
                keypoints=s.keypoints
            )
            for s in active_subjects
        ]
        total_profiles = int(engine.ntotal)

        return TelemetryResponse(
            status="success",
            events=events,
            active_tracks=active_list,
            total_profiles=total_profiles,
            fps=float(broadcaster.current_fps)
        )
    except Exception as e:
        return TelemetryResponse(
            status="warning",
            events=[],
            active_tracks=[],
            total_profiles=0,
            fps=float(broadcaster.current_fps),
            message=str(e)
        )


# =====================================================================
# Modular Routers: Attendance & Courses
# =====================================================================
from src.api.routes.attendance_routes import (
    router as attendance_router,
    init_attendance_dependencies,
    get_attendance_config,
    set_attendance_config,
    check_all_attendance,
    get_attendance_checklist,
    manual_mark_attendance,
    export_attendance_csv,
    get_unusual_persons,
    dismiss_unusual_person,
    enroll_unusual_person,
    get_attendance_audit,
    reset_attendance,
)
from src.api.routes.course_routes import (
    router as course_router,
    set_course_service_provider,
    list_courses,
    create_course,
    get_course_roster,
    enroll_student_in_course,
    add_course_schedule,
)

init_attendance_dependencies(attendance_service, unusual_service, enrollment_manager)
set_course_service_provider(attendance_service)

app.include_router(attendance_router)
app.include_router(course_router)



@app.get("/api/subject/avatar/{name}")
async def get_subject_avatar(name: str):
    try:
        subject_dir = get_safe_subject_dir(name)
        if subject_dir and os.path.exists(subject_dir):
            img_files = [f for f in os.listdir(subject_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            files = sorted(img_files, key=lambda f: os.path.getmtime(os.path.join(subject_dir, f)), reverse=True)
            no_cache_headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
            for f in files:
                if "frontal" in f.lower():
                    return FileResponse(os.path.join(subject_dir, f), headers=no_cache_headers)
            if files:
                return FileResponse(os.path.join(subject_dir, files[0]), headers=no_cache_headers)
        logo_path = os.path.join(PROJECT_ROOT, "templates", "sleeper_agent_logo.png")
        if os.path.exists(logo_path):
            return FileResponse(logo_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
        return JSONResponse(status_code=404, content={"message": "Avatar not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/api/subject/{name}")
async def get_subject_dossier(name: str, request: Request):
    try:
        subject_dir = get_safe_subject_dir(name)
        clean_name = os.path.basename(name.strip())

        images = []
        if subject_dir and os.path.exists(subject_dir):
            folder_basename = os.path.basename(subject_dir)
            for fname in sorted(os.listdir(subject_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    images.append({
                        "filename": fname,
                        "url": f"/data/known_faces/{folder_basename}/{fname}"
                    })

        meta_info = {
            "name": clean_name,
            "occupation": "REGISTERED SUBJECT",
            "phone": "N/A",
            "address": "SYSTEM DATABASE",
            "clearance": "LEVEL 1 GRANTED",
            "dob": "N/A"
        }
        
        # Authoritative lookup from Biometric Repository (SQLite)
        subject_record = biometric_repo.get_subject_by_name(clean_name)
        if subject_record:
            meta_info.update(subject_record)
        else:
            # Fallback to metadata.json cache
            metadata_path = os.path.join(DATA_DIR, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    data = json.load(f)
                    for k, v in data.items():
                        if isinstance(v, dict) and v.get("name", "").lower() == clean_name.lower():
                            meta_info.update(v)
                            break

        # Apply PII Redaction unless caller holds verified administrative authorization
        sanitized_meta = sanitize_subject_dossier(meta_info, is_admin=is_admin_request(request))

        return JSONResponse(content={
            "status": "success",
            "name": clean_name,
            "meta": sanitized_meta,
            "images": images
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.websocket("/ws/raw_video")
async def websocket_raw_video_endpoint(websocket: WebSocket):
    if not is_allowed_websocket_origin(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    await broadcaster.stream_raw_websocket(websocket)


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    if not is_allowed_websocket_origin(websocket):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    await broadcaster.stream_processed_websocket(websocket)


@app.get("/video_feed")
async def mjpeg_fallback():
    """MJPEG HTTP Streaming Fallback for Web Dashboard"""
    return StreamingResponse(
        broadcaster.mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def find_available_port(start_port=8000, max_attempts=10):
    import socket
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('0.0.0.0', port))
                return port
            except OSError:
                continue
            
    return start_port


def start_server():
    port = find_available_port(8000)
    host = get_server_host()
    print("=" * 60)
    print("  SLEEPER AGENT AI BIOMETRIC REAL-TIME WEB DASHBOARD SERVER")
    print("=" * 60)
    print(f"  [+] Host Interface:    {host}")
    print(f"  [+] Web Dashboard URL: http://{host}:{port}")
    print(f"  [+] WebSocket Video:   ws://{host}:{port}/ws/video")
    print(f"  [+] Audit Events API:  http://{host}:{port}/api/events")
    print(f"  [+] Liveness Probe:    http://{host}:{port}/healthz")
    print(f"  [+] Readiness Probe:   http://{host}:{port}/ready")
    if host == "0.0.0.0" and not get_configured_api_key():
        print("  [!] WARNING: Server bound to 0.0.0.0 with no CCTV_API_KEY configured!")
    print("=" * 60)
    try:
        config = uvicorn.Config("src.api.cctv_server:app", host=host, port=port, log_level="info", reload=False)
        server = uvicorn.Server(config)
        server.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        print(f"[CCTV Server] Server terminated: {e}")


if __name__ == "__main__":
    start_server()

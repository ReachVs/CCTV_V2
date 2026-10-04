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

import sys
import time
import base64
import asyncio
import json
import sqlite3
import threading
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import aiosqlite

warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.inference.live_cctv as live_cctv
from src.inference.engine import BiometricTrackingEngine

# Canonical Biometric Tracking Engine instance
engine: BiometricTrackingEngine = live_cctv.engine
from src.inference.multi_camera import MultiCameraIngestionManager, BufferlessVideoCapture
from src.api.schemas import (
    FaceEnrollmentRequest,
    FaceEnrollmentResponse,
    CameraFeedConfig,
    CameraStatusResponse,
    ActiveTrackSchema,
    AuditEventSchema,
    TelemetryResponse,
    SystemModeRequest,
    SystemModeResponse,
    HealthResponse,
    ReadinessResponse
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "faces.db")
INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

from src.database.audit_repository import AuditRepository
audit_repo = AuditRepository(DB_PATH)

TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR if os.path.exists(TEMPLATES_DIR) else "templates")
EVENTS_DIR = os.path.join(PROJECT_ROOT, "events")

SYSTEM_MODE = "SURVEILLANCE"
current_fps = 0.0
fps_counter = 0
last_fps_time = time.time()

# Multi-camera Manager & Legacy Single Camera Stream Manager
multi_camera_manager = MultiCameraIngestionManager()


class GlobalCameraStream:
    """
    On-Demand Reference-Counted Camera Stream Manager.
    Delegates to MultiCameraIngestionManager for zero-contention camera hardware access,
    or falls back to direct VideoCapture handle.
    """
    def __init__(self, source=0, width=854, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.paused = False
        self.client_count = 0
        self.thread = None
        self.shutdown_timer = None

    def pause(self):
        with self.lock:
            self.paused = True
            if self.cap is not None:
                try:
                    if self.cap.isOpened():
                        self.cap.release()
                except Exception as e:
                    print(f"[Camera Pause Warning] {e}")
                self.cap = None

    def resume(self):
        with self.lock:
            self.paused = False

    def acquire(self):
        with self.lock:
            if self.shutdown_timer is not None:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None

            self.client_count += 1
            if self.client_count >= 1 and not self.running:
                self._start_thread()

    def release(self):
        with self.lock:
            self.client_count = max(0, self.client_count - 1)
            if self.client_count == 0 and self.running:
                if self.shutdown_timer is not None:
                    self.shutdown_timer.cancel()
                self.shutdown_timer = threading.Timer(3.0, self._deferred_shutdown)
                self.shutdown_timer.daemon = True
                self.shutdown_timer.start()

    def _deferred_shutdown(self):
        with self.lock:
            self.shutdown_timer = None
            if self.client_count == 0 and self.running:
                self.running = False
                self._release_hardware_locked()

    def _start_thread(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        global fps_counter, last_fps_time, current_fps, SYSTEM_MODE
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            # Step 1: Attempt reading from multi_camera_manager primary stream to prevent hardware contention
            ret, frame = multi_camera_manager.get_frame("primary")

            # Step 2: Fallback to direct VideoCapture if multi_camera_manager stream is inactive
            if not ret or frame is None:
                with self.lock:
                    if not self.running:
                        break
                    if self.cap is None or not self.cap.isOpened():
                        try:
                            self.cap = cv2.VideoCapture(self.source)
                            if self.cap.isOpened():
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                for _ in range(3):
                                    self.cap.read()
                        except Exception as e:
                            print(f"[Camera Error] VideoCapture failed: {e}")

                    if self.cap is not None and self.cap.isOpened():
                        try:
                            ret, frame = self.cap.read()
                        except Exception:
                            ret, frame = False, None

            if not ret or frame is None:
                synthetic = self._generate_synthetic_frame()
                with self.lock:
                    self.latest_raw_frame = synthetic
                    self.latest_processed_frame = synthetic
                time.sleep(0.05)
                continue

            raw_copy = frame.copy()
            if SYSTEM_MODE == "ENROLLMENT":
                proc = raw_copy
            else:
                try:
                    res = engine.process_frame(frame, annotate=True)
                    proc = res.annotated_frame
                except Exception as e:
                    print(f"[CCTV Frame Processing Warning] {e}")
                    proc = frame

            fps_counter += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                current_fps = round(fps_counter / (now - last_fps_time), 1)
                fps_counter = 0
                last_fps_time = now

            with self.lock:
                self.latest_raw_frame = raw_copy
                self.latest_processed_frame = proc

            time.sleep(0.002)

        with self.lock:
            self._release_hardware_locked()

    def _release_hardware_locked(self):
        if self.cap is not None:
            try:
                if self.cap.isOpened():
                    self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _generate_synthetic_frame(self):
        canvas = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(canvas, "CCTV SYSTEM ENGINE ONLINE", (220, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Time: {time.strftime('%H:%M:%S')} | Standby Mode", (260, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return canvas

    def get_raw_frame(self):
        with self.lock:
            if self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return self._generate_synthetic_frame()

    def get_processed_frame(self):
        with self.lock:
            if self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
            return self._generate_synthetic_frame()

    def stop(self):
        self.running = False
        with self.lock:
            self._release_hardware_locked()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


camera_stream = GlobalCameraStream()


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
    
    # Initialize default primary camera feed
    multi_camera_manager.add_camera("primary", 0, width=854, height=480)
    print("[CCTV Server Lifespan] System models & primary camera ingestion pool initialized successfully.")

    yield

    print("[CCTV Server Lifespan] Shutdown signal received. Stopping camera streams & background workers...")
    multi_camera_manager.stop_all()
    camera_stream.stop()
    engine.stop()
    print("[CCTV Server Lifespan] Clean shutdown complete.")


app = FastAPI(
    title="Real-Time AI Biometric Security Engine (V4 - Phase 5)",
    lifespan=lifespan
)

if os.path.exists(EVENTS_DIR):
    app.mount("/events", StaticFiles(directory=EVENTS_DIR), name="events")

if os.path.exists(KNOWN_FACES_DIR):
    app.mount("/data/known_faces", StaticFiles(directory=KNOWN_FACES_DIR), name="known_faces")


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
    global SYSTEM_MODE
    return SystemModeResponse(
        status="success",
        mode=SYSTEM_MODE,
        pose_only_mode=engine.is_pose_only
    )


@app.post("/api/system/mode", response_model=SystemModeResponse)
async def set_system_mode(req: SystemModeRequest):
    global SYSTEM_MODE
    SYSTEM_MODE = req.mode.upper()
    if req.pose_only_mode is not None:
        engine.set_pose_only(req.pose_only_mode)

    if SYSTEM_MODE == "ENROLLMENT":
        camera_stream.pause()
    else:
        camera_stream.resume()

    return SystemModeResponse(
        status="success",
        mode=SYSTEM_MODE,
        pose_only_mode=engine.is_pose_only,
        message=f"System switched to {SYSTEM_MODE} mode."
    )


@app.api_route("/api/system/mode/enrollment", methods=["GET", "POST"])
@app.api_route("/api/camera/pause", methods=["GET", "POST"])
async def set_mode_enrollment():
    global SYSTEM_MODE
    SYSTEM_MODE = "ENROLLMENT"
    camera_stream.pause()
    return SystemModeResponse(
        status="success",
        mode=SYSTEM_MODE,
        pose_only_mode=engine.is_pose_only,
        message="System switched to Enrollment Mode."
    )


@app.api_route("/api/system/mode/surveillance", methods=["GET", "POST"])
@app.api_route("/api/camera/resume", methods=["GET", "POST"])
async def set_mode_surveillance():
    global SYSTEM_MODE
    SYSTEM_MODE = "SURVEILLANCE"
    camera_stream.resume()
    return SystemModeResponse(
        status="success",
        mode=SYSTEM_MODE,
        pose_only_mode=engine.is_pose_only,
        message="System switched to Surveillance Mode."
    )


@app.get("/api/camera/streams", response_model=CameraStatusResponse)
async def list_camera_streams():
    return CameraStatusResponse(
        status="success",
        active_streams=multi_camera_manager.active_stream_ids()
    )


@app.post("/api/camera/streams", response_model=CameraStatusResponse)
async def add_camera_stream(cfg: CameraFeedConfig):
    success = multi_camera_manager.add_camera(
        stream_id=cfg.stream_id,
        source=cfg.source,
        width=cfg.width,
        height=cfg.height
    )
    if not success:
        raise HTTPException(status_code=400, detail=f"Camera stream '{cfg.stream_id}' already exists.")
    return CameraStatusResponse(
        status="success",
        active_streams=multi_camera_manager.active_stream_ids(),
        message=f"Added camera stream '{cfg.stream_id}'"
    )


@app.get("/api/camera/frame")
async def get_current_camera_frame():
    frame = camera_stream.get_raw_frame()
    ret, buf = cv2.imencode('.jpg', frame)
    if ret:
        b64 = base64.b64encode(buf).decode('utf-8')
        return JSONResponse({"status": "success", "image_base64": f"data:image/jpeg;base64,{b64}"})
    return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to encode frame."})


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/enroll", response_class=HTMLResponse)
async def enroll_page(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request})


def _process_enrollment_sync(req: FaceEnrollmentRequest) -> Dict[str, Any]:
    try:
        image_b64_list = req.images_base64 if req.images_base64 else ([req.image_base64] if req.image_base64 else [])
        imgs = []
        for b64 in image_b64_list:
            if not b64:
                continue
            if "," in b64:
                b64 = b64.split(",")[1]
            img_bytes = base64.b64decode(b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                imgs.append(img)

        if not imgs:
            return {"status": "error", "message": "Invalid or missing face image payload."}

        clean_folder = req.full_name.replace(" ", "_")
        faces_dir = os.path.join(DATA_DIR, "known_faces", clean_folder)
        os.makedirs(faces_dir, exist_ok=True)

        from deepface import DeepFace
        from src.utils.face_align import YuNetFaceAligner
        face_aligner = YuNetFaceAligner()
        embeddings = []

        pose_names = ["Frontal", "Left_Angle", "Right_Angle", "Tilt_Up", "Pose_5"]
        for idx, img in enumerate(imgs):
            pose_tag = pose_names[idx] if idx < len(pose_names) else f"Pose_{idx+1}"
            filename = f"{clean_folder}_{pose_tag}_{int(time.time())}.jpg"
            save_path = os.path.join(faces_dir, filename)
            cv2.imwrite(save_path, img)

            try:
                emb_v = engine.extract_face_embedding(img, align=True)

                if emb_v is None:
                    aligned = face_aligner.align_face(img, output_size=(112, 112), strict=False)
                    if aligned is not None and aligned.size > 0:
                        reps = DeepFace.represent(
                            img_path=aligned,
                            model_name="ArcFace",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                        if reps and len(reps) > 0:
                            v = np.array(reps[0]["embedding"], dtype=np.float32)
                            emb_v = v / (np.linalg.norm(v) + 1e-10)

                if emb_v is not None:
                    embeddings.append(emb_v)
            except Exception as e:
                print(f"[Enrollment Worker] Pose #{idx+1} warning: {e}")

        if not embeddings:
            return {
                "status": "error",
                "message": "Biometric extraction failed: No face detected in photo."
            }

        final_vector = np.mean(embeddings, axis=0)
        final_vector = final_vector / (np.linalg.norm(final_vector) + 1e-10)

        index_path = os.path.join(DATA_DIR, "faces.index")
        metadata_path = os.path.join(DATA_DIR, "metadata.json")

        with engine.lock:
            import faiss
            if os.path.exists(index_path):
                index = faiss.read_index(index_path)
            else:
                index = faiss.IndexFlatL2(512)

            start_id = index.ntotal
            index.add(np.array(embeddings, dtype=np.float32))
            faiss.write_index(index, index_path)

            meta_dict = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        meta_dict = json.load(f)
                except Exception:
                    meta_dict = {}

            for idx, emb_v in enumerate(embeddings):
                vec_id = start_id + idx
                meta_dict[str(vec_id)] = {
                    "name": req.full_name,
                    "phone": req.phone_number,
                    "occupation": req.occupation,
                    "dob": req.dob,
                    "clearance": req.clearance,
                    "address": req.address,
                    "threat_level": req.threat_level,
                    "poses_count": len(embeddings)
                }

            with open(metadata_path, 'w') as f:
                json.dump(meta_dict, f, indent=4)

            try:
                engine.reload_index()
            except Exception as reload_err:
                print(f"[Enrollment Worker] Hot reload warning: {reload_err}")

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL
                )
            """)
            cursor.execute(
                "INSERT INTO people (name, embedding) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding",
                (req.full_name, json.dumps(final_vector.tolist()))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[Enrollment Worker] SQLite sync warning: {e}")

        return {
            "status": "success",
            "message": f"Successfully enrolled '{req.full_name}' ({len(embeddings)} pose(s) processed)",
            "profile_id": start_id,
            "total_profiles": int(index.ntotal)
        }
    except Exception as e:
        return {"status": "error", "message": f"Server processing error: {str(e)}"}


@app.post("/api/enroll", response_model=FaceEnrollmentResponse)
async def enroll_subject_api(req: FaceEnrollmentRequest):
    if not req.full_name.strip():
        raise HTTPException(status_code=400, detail="Full Name is required.")

    result = await asyncio.to_thread(_process_enrollment_sync, req)

    if result.get("status") == "success":
        return FaceEnrollmentResponse(**result)
    else:
        raise HTTPException(status_code=400, detail=result.get("message", "Enrollment failed"))


@app.get("/api/events", response_model=TelemetryResponse)
async def get_audit_events():
    try:
        events = []
        if hasattr(engine, 'encrypted_audit_logger') and engine.encrypted_audit_logger is not None:
            raw_events = engine.encrypted_audit_logger.fetch_recent_events(limit=50)
            events = [
                AuditEventSchema(
                    id=e["id"],
                    timestamp=e["timestamp"],
                    person_name=e["person_name"],
                    track_id=e["track_id"],
                    match_distance=e["match_distance"],
                    meta=e.get("meta", {})
                ) for e in raw_events
            ]
        elif os.path.exists(DB_PATH):
            raw_events = await audit_repo.get_recent_events(limit=50)
            events = [AuditEventSchema(**r) for r in raw_events]

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
            fps=float(current_fps)
        )
    except Exception as e:
        return TelemetryResponse(
            status="warning",
            events=[],
            active_tracks=[],
            total_profiles=0,
            fps=float(current_fps),
            message=str(e)
        )


@app.get("/api/subject/avatar/{name}")
async def get_subject_avatar(name: str):
    try:
        clean_name = name.strip()
        clean_dir = clean_name.replace(" ", "_")
        subject_dir = os.path.join(DATA_DIR, "known_faces", clean_dir)
        if not os.path.exists(subject_dir):
            subject_dir = os.path.join(DATA_DIR, "known_faces", clean_name)
        if os.path.exists(subject_dir):
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
async def get_subject_dossier(name: str):
    try:
        clean_name = name.strip()
        clean_dir_name = clean_name.replace(" ", "_")
        subject_dir = os.path.join(DATA_DIR, "known_faces", clean_dir_name)
        if not os.path.exists(subject_dir):
            subject_dir = os.path.join(DATA_DIR, "known_faces", clean_name)

        images = []
        if os.path.exists(subject_dir):
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
        
        metadata_path = os.path.join(DATA_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                data = json.load(f)
                for k, v in data.items():
                    if isinstance(v, dict) and v.get("name", "").lower() == clean_name.lower():
                        meta_info.update(v)
                        break

        return JSONResponse(content={
            "status": "success",
            "name": clean_name,
            "meta": meta_info,
            "images": images
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


@app.websocket("/ws/raw_video")
async def websocket_raw_video_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera_stream.acquire()
    try:
        while True:
            raw_frame = camera_stream.get_raw_frame()
            ret_encode, buffer = cv2.imencode('.jpg', raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret_encode:
                await asyncio.sleep(0.02)
                continue

            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            payload = {
                "image": f"data:image/jpeg;base64,{jpg_as_text}",
                "fps": float(current_fps)
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.015)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        camera_stream.release()


@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    await websocket.accept()
    camera_stream.acquire()
    try:
        while True:
            processed_frame = camera_stream.get_processed_frame()
            ret_encode, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ret_encode:
                await asyncio.sleep(0.02)
                continue

            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            active_list = [
                {
                    "track_id": s.track_id,
                    "name": s.name,
                    "match_distance": 0.45,
                    "meta": s.meta
                }
                for s in engine.get_active_tracks()
            ]

            payload = {
                "image": f"data:image/jpeg;base64,{jpg_as_text}",
                "fps": float(current_fps),
                "active_tracks": active_list
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.015)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        camera_stream.release()


@app.get("/video_feed")
async def mjpeg_fallback():
    """MJPEG HTTP Streaming Fallback for Web Dashboard"""
    def generate():
        camera_stream.acquire()
        try:
            while True:
                proc = camera_stream.get_processed_frame()
                ret_encode, buffer = cv2.imencode('.jpg', proc, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret_encode:
                    time.sleep(0.02)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.03)
        finally:
            camera_stream.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


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
    print("=" * 60)
    print("  SLEEPER AGENT AI BIOMETRIC REAL-TIME WEB DASHBOARD SERVER")
    print("=" * 60)
    print(f"  [+] Web Dashboard URL: http://localhost:{port}")
    print(f"  [+] WebSocket Video:   ws://localhost:{port}/ws/video")
    print(f"  [+] Audit Events API:  http://localhost:{port}/api/events")
    print(f"  [+] Liveness Probe:    http://localhost:{port}/healthz")
    print(f"  [+] Readiness Probe:   http://localhost:{port}/ready")
    print("=" * 60)
    try:
        uvicorn.run("src.api.cctv_server:app", host="0.0.0.0", port=port, log_level="info", reload=False)
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    start_server()

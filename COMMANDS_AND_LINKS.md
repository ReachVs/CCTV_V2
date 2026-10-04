# 🔗 CCTV Security & Biometrics Pipeline — Commands & File Index

A complete directory of all project files, source modules, configuration files, web interfaces, and executable commands.

---

## 🗂️ 1. Project Directory & File Links

### 📄 Documentation & Core Files
* [README.md](file:///Users/ceaser/Documents/School/CCTV%20Project/README.md) — Main enterprise documentation & architectural overview
* [cctv_system_architecture.md](file:///Users/ceaser/Documents/School/CCTV%20Project/cctv_system_architecture.md) — Deep technical architecture, component design & flowcharts
* [COMMANDS.md](file:///Users/ceaser/Documents/School/CCTV%20Project/COMMANDS.md) — Quick command reference
* [requirements.txt](file:///Users/ceaser/Documents/School/CCTV%20Project/requirements.txt) — Python dependencies specification
* [docker-compose.yml](file:///Users/ceaser/Documents/School/CCTV%20Project/docker-compose.yml) — Multi-container Docker orchestration
* [Dockerfile](file:///Users/ceaser/Documents/School/CCTV%20Project/Dockerfile) — Docker build instructions

### 🚀 Entry Point Wrappers
* [cctv_server.py](file:///Users/ceaser/Documents/School/CCTV%20Project/cctv_server.py) — Web server launcher
* [cctv_server_fastapi.py](file:///Users/ceaser/Documents/School/CCTV%20Project/cctv_server_fastapi.py) — FastAPI ASGI server entry-point
* [live_cctv.py](file:///Users/ceaser/Documents/School/CCTV%20Project/live_cctv.py) — Standalone OpenCV inference launcher
* [live_cctv_optimized.py](file:///Users/ceaser/Documents/School/CCTV%20Project/live_cctv_optimized.py) — Optimized live inference launcher
* [enroll.py](file:///Users/ceaser/Documents/School/CCTV%20Project/enroll.py) — Face enrollment runner
* [enroll_webcam.py](file:///Users/ceaser/Documents/School/CCTV%20Project/enroll_webcam.py) — Guided 3-second webcam face enrollment
* [db_setup.py](file:///Users/ceaser/Documents/School/CCTV%20Project/db_setup.py) — Database schema initialization runner
* [tune_tracker.py](file:///Users/ceaser/Documents/School/CCTV%20Project/tune_tracker.py) — ByteTrack hyperparameter tuning runner
* [test_pipeline.py](file:///Users/ceaser/Documents/School/CCTV%20Project/test_pipeline.py) — Unit test suite
* [mock_pipeline_validator.py](file:///Users/ceaser/Documents/School/CCTV%20Project/mock_pipeline_validator.py) — Mock pipeline validator
* [run_stress_tests.py](file:///Users/ceaser/Documents/School/CCTV%20Project/run_stress_tests.py) — System load & stress testing engine
* [diagnose.py](file:///Users/ceaser/Documents/School/CCTV%20Project/diagnose.py) — Environment & system diagnostics
* [export_models.py](file:///Users/ceaser/Documents/School/CCTV%20Project/export_models.py) — Model conversion runner (ONNX / CoreML)
* [export_coreml.py](file:///Users/ceaser/Documents/School/CCTV%20Project/export_coreml.py) — CoreML exporter runner
* [warm.py](file:///Users/ceaser/Documents/School/CCTV%20Project/warm.py) — Pre-warms neural models for fast startup

### 🧩 Source Code Modules (`src/`)

#### 🌐 API Server Layer (`src/api/`)
* [src/api/cctv_server.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/api/cctv_server.py) — Main FastAPI + WebSockets + aiosqlite server
* [src/api/cctv_server_fastapi.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/api/cctv_server_fastapi.py) — Core ASGI FastAPI application setup
* [src/api/mcp_server.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/api/mcp_server.py) — Model Context Protocol (MCP) server integration

#### 🗄️ Database & Enrollment Layer (`src/database/`)
* [src/database/db_setup.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/database/db_setup.py) — SQLite WAL mode database schema setup
* [src/database/enroll.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/database/enroll.py) — AdaFace embedding & FAISS vector index builder

#### 📹 Inference & Biometrics Layer (`src/inference/`)
* [src/inference/live_cctv.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/live_cctv.py) — YOLOv8 + ByteTrack + AdaFace real-time vision loop
* [src/inference/biometric_core.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/biometric_core.py) — Biometric search, vector matching & identity engine
* [src/inference/live_cctv_optimized.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/live_cctv_optimized.py) — High-throughput pipeline variant

#### 🛠️ Utility Layer (`src/utils/`)
* [src/utils/jitter_filter.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/jitter_filter.py) — `JitterFilterEngine` (deadzone & EMA spatial smoother)
* [src/utils/tune_tracker.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/tune_tracker.py) — Grid-search ByteTrack hyperparameter optimizer
* [src/utils/export_coreml.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/export_coreml.py) — CoreML conversion tools
* [src/utils/face_align.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/face_align.py) — YuNet landmark facial alignment (Umeyama transform)
* [src/utils/fsrcnn_upscaler.py](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/fsrcnn_upscaler.py) — FSRCNN super-resolution pre-processor

### 🎨 Web Templates & Assets (`templates/`)
* [templates/dashboard.html](file:///Users/ceaser/Documents/School/CCTV%20Project/templates/dashboard.html) — HTML5 Security Monitoring Dashboard UI
* [templates/enroll.html](file:///Users/ceaser/Documents/School/CCTV%20Project/templates/enroll.html) — Web-based Face Enrollment Interface

### ⚙️ Configuration Files
* [bytetrack.yaml](file:///Users/ceaser/Documents/School/CCTV%20Project/bytetrack.yaml) — ByteTrack tracker configuration
* [botsort.yaml](file:///Users/ceaser/Documents/School/CCTV%20Project/botsort.yaml) — BotSORT tracker configuration

---

## 💻 2. Complete Command Catalog

> **Prerequisite:** Make sure to activate the Python virtual environment before executing commands:
> ```bash
> source venv/bin/activate
> ```

### 1. Environment Setup & Hardware Diagnostics
```bash
# 1. Create Python virtual environment
python3 -m venv venv

# 2. Activate virtual environment
source venv/bin/activate

# 3. Install/upgrade core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify Apple Silicon Metal (MPS) GPU Acceleration
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available()); print('MPS Built:', torch.backends.mps.is_built())"

# 5. Run full system environment diagnostics
python3 diagnose.py
```

### 2. Database & Schema Initialization
```bash
# Initialize SQLite audit database (data/faces.db) with WAL mode
python3 db_setup.py

# Direct module invocation
python3 -m src.database.db_setup
```

### 3. Subject Face Enrollment
```bash
# Enroll faces from images stored in data/known_faces/
python3 enroll.py

# Direct module invocation
python3 -m src.database.enroll

# Interactive webcam enrollment wizard (captures sample frames)
python3 enroll_webcam.py
```

### 4. Launching Servers & APIs
```bash
# Start FastAPI + WebSockets Server (Recommended)
python3 cctv_server.py

# Direct FastAPI entrypoint
python3 cctv_server_fastapi.py

# Direct module invocation
python3 -m src.api.cctv_server

# Run Model Context Protocol (MCP) Server
python3 -m src.api.mcp_server
```

### 5. Live CCTV Standalone Application
```bash
# Launch OpenCV GUI Live Inference Pipeline
python3 live_cctv.py

# Launch Optimized High-Performance Variant
python3 live_cctv_optimized.py

# Direct module invocation
python3 -m src.inference.live_cctv
```

### 6. Neural Model Export & Pre-warming
```bash
# Export models (PyTorch to ONNX / CoreML)
python3 export_models.py

# Export YOLO to Apple Silicon CoreML (.mlpackage)
python3 export_coreml.py

# Direct CoreML exporter module invocation
python3 -m src.utils.export_coreml

# Pre-warm deep learning models into memory (eliminates first-frame lag)
python3 warm.py
```

### 7. Hyperparameter & Tracker Tuning
```bash
# Run ByteTrack grid search hyperparameter optimizer
python3 tune_tracker.py

# Direct module invocation
python3 -m src.utils.tune_tracker
```

### 8. Testing, Validation & Stress Testing
```bash
# Run complete test suite
python3 test_pipeline.py

# Run mock pipeline validation test
python3 mock_pipeline_validator.py

# Run automated stress test suite (FPS & memory stability under load)
python3 run_stress_tests.py
```

### 9. Docker Deployment
```bash
# Build and run containers in foreground
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build

# View real-time container logs
docker-compose logs -f

# Stop containers
docker-compose down
```

---

## 🌐 3. Web Service Endpoints

Once `cctv_server.py` is running:

| Service / Endpoint | URL | Description |
|---|---|---|
| **Web Security Dashboard** | `http://localhost:8000/` | Real-time monitoring UI |
| **Face Enrollment Page** | `http://localhost:8000/enroll` | Subject registration page |
| **Swagger API Docs** | `http://localhost:8000/docs` | Interactive OpenAPI documentation |
| **ReDoc API Docs** | `http://localhost:8000/redoc` | OpenAPI reference docs |
| **WebSocket Video Feed** | `ws://localhost:8000/ws/video` | Real-time frame & telemetry stream |

---

## 📁 4. Data Storage & Model Assets

| Path | Description |
|---|---|
| `data/faces.db` | SQLite audit database storing detection event logs |
| `data/faces.index` | FAISS HNSW vector search index storing 512D facial embeddings |
| `data/metadata.json` | JSON mapping of vector IDs to registered subject names |
| `data/known_faces/` | Directory containing subject profile images for enrollment |
| `yolo26n.pt` | YOLO object detection PyTorch weights |
| `yolov8n.pt` | Standard YOLOv8 model weights |
| `yolo26n.mlpackage/` | Exported CoreML Apple Silicon model package |


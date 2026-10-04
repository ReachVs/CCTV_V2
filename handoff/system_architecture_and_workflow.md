# CCTV AI & Biometric Recognition System: System Architecture & Workflow Guide

This document provides a comprehensive technical overview of the **CCTV Security AI Engine**, detailing the complete system architecture, component layers, thread synchronization model, data schemas, and end-to-end execution workflow.

---

## 🏗️ 1. System Architecture Diagram

```mermaid
flowchart TD
    subgraph INGESTION["1. Frame Ingestion Layer"]
        A[720p HD Camera / RTSP Feed] --> B[BufferlessVideoCapture Daemon Thread]
    end

    subgraph TRACKING["2. Object Detection & Spatial Tracking Layer"]
        B --> C[YOLOv8 + ByteTrack Engine]
        C --> D[Confidence-Based Kalman Filter CBKF]
        D -->|60-Frame Retention Memory| E{Smart Tracker-Gating Cache}
    end

    subgraph BIOMETRICS["3. Biometric Alignment & Vector Identification Layer"]
        E -->|Unconfirmed Track| F1[Minimum 80x80px Pixel-Size Gate]
        F1 --> F2[YuNet 5-Landmark Umeyama Affine Aligner]
        F2 -->|Valid Face Crop| G[BatchFaceEmbedder Queue]
        F2 -->|Non-Face Crop| H1[Silent Drop / No Embedding]
        G --> H2[ArcFace 512D Feature Extractor]
        H2 --> I[FAISS L2 / HNSW Vector Search]
        I --> J[Top-2 Margin Gating & Temporal Consensus]
        J -->|Confirmed Identity| K[Active Identity Claims Registry]
        
        E -->|Confirmed Identity| K
    end

    subgraph STABILIZATION["4. Spatial Stabilization & Deadzone Engine"]
        K --> L[Dimension Hysteresis Inertia Filter]
        L --> M[Zero-Jitter Velocity Deadzone Filter v < 3.5px]
    end

    subgraph STREAMING["5. API, Telemetry & Web Dashboard Layer"]
        M --> N[FastAPI ASGI Web Server http://localhost:8000]
        N --> O[WebSocket Stream ws://localhost:8000/ws/video]
        N --> P[aiosqlite Audit Log Database data/faces.db]
    end
```

---

## 🧩 2. Deep Component Architecture

### A. Ingestion Layer (`BufferlessVideoCapture`)
- **File**: [`src/inference/live_cctv.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/live_cctv.py)
- **Purpose**: Eliminates macOS Cocoa GUI window freeze and camera frame queuing latency.
- **Mechanism**: A dedicated background daemon thread continuously drains OpenCV hardware frame buffers (`max-buffers=1 drop=true`). Calling `read()` fetches the newest hardware camera frame with **0ms frame queue latency**.

### B. Object Detection & Spatial Tracking Layer (`YOLOv8` + `ByteTrack` + `CBKF`)
- **Components**: `YOLOv8-Nano` detector running on a 640x360 inference canvas paired with `ByteTrack` motion tracking.
- **CBKF (Confidence-Based Kalman Filter)**: Dynamically adjusts measurement noise covariance based on detection confidence:
  $$R(conf) = R_{base} \cdot e^{-\beta \cdot (conf - c_0)}$$
- **Spatial Continuity Inheritance**: When a track ID drops temporarily due to occlusion or turning, lost tracks are kept in a 60-frame retention buffer (2.0s). If a new box appears in the same spatial location ($\text{IoU} \ge 0.40$), the confirmed identity is inherited instantly without re-scanning.

### C. Biometric Alignment & Verification Layer (`YuNet` + `ArcFace` + `FAISS`)
- **Pixel-Size Gate**: Crops smaller than $80 \times 80\text{px}$ are ignored before reaching alignment or embedding generation.
- **YuNet 5-Landmark Affine Alignment (`YuNetFaceAligner`)**:
  - Uses `cv2.FaceDetectorYN` to extract 5 canonical facial landmarks (eyes, nose, mouth corners) in $< 4\text{ms}$.
  - Applies 2D Umeyama similarity transform to normalize off-angle faces into $112 \times 112$ canonical frontal coordinates.
  - **Strict Landmark Verification**: If 0 landmarks are detected (e.g., on hands, arms, clothing folds, or background noise), `align_face(strict=True)` returns `None`, immediately dropping the crop before ArcFace or FAISS search.
- **Batch Embedding Extractor (`BatchFaceEmbedder`)**:
  - A background worker thread that processes face crops asynchronously using ArcFace 512D.
  - Normalizes output vectors to unit $L_2$ norm ($|z|_2 = 1.0$).
- **Vector Search Index & Decision Engine (`BiometricVerificationEngine`)**:
  - FAISS index storing 512D registered face vectors.
  - **Threshold Criteria**:
    - **Absolute $L_2$ Match Threshold**: $d \le 0.68$ (matches profile; $> 0.68$ rejected as `Unknown Person`).
    - **Top-2 Relative Margin**: $\Delta d = d_2 - d_1 \ge 0.15$ (rejects ambiguous collisions).
    - **Temporal Consensus Voting**: Requires 2 consistent identification votes (`consensus_votes = 2`) before locking target status.

### D. Spatial Stabilization Engine (`JitterFilterEngine`)
- **Dimension Hysteresis**: Applies 80/20 box dimension memory between consecutive frames to prevent UI bounding box flickering.
- **Zero-Jitter Velocity Deadzone**: If displacement velocity $v < 3.5\text{px/frame}$, bounding box coordinates are **frozen 100% in place**, eliminating camera sensor micro-wiggles.

### E. Web API & Security Telemetry Layer (`cctv_server.py`)
- **FastAPI / Uvicorn Server**: Runs ASGI server on `http://localhost:8000`.
- **WebSocket Video Feed (`/ws/video`)**: Encodes HD frames to base64 JPEG and broadcasts frame + track telemetry payload at 30+ FPS.
- **SQLite Audit Logger (`aiosqlite`)**: Records security verification events into `data/faces.db` in WAL (Write-Ahead Logging) mode.

---

## 🔄 3. Complete End-to-End Frame Processing Workflow

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Hardware Camera
    participant Cap as BufferlessVideoCapture
    participant YOLO as YOLOv8 + ByteTrack
    participant Engine as Live CCTV Engine
    participant Embedder as BatchFaceEmbedder
    participant FAISS as FAISS Index
    participant Web as Web Dashboard Client

    Cam->>Cap: Capture 720p Video Frame
    Cap->>Engine: Fetch Latest Frame (0ms Latency)
    Engine->>YOLO: Track Bounding Boxes (640x360 Canvas)
    YOLO-->>Engine: Return Active Track IDs & Boxes
    
    alt Track is Already Confirmed (e.g. 'Ceaser')
        Engine->>Engine: Smart Bypass FAISS (Keep Confirmed Identity)
    else Track is Unconfirmed (Scanning Target)
        Engine->>Engine: Minimum Pixel-Size Check (w >= 80, h >= 80)
        alt Box < 80x80px (Hand/Noise)
            Engine->>Engine: Ignore Crop Extraction
        else Box >= 80x80px
            Engine->>Embedder: Enqueue Crop for Alignment & Embedding
            Embedder->>Embedder: YuNet 5-Landmark Alignment (strict=True)
            alt No Facial Landmarks Found
                Embedder-->>Engine: Return None (Drop Crop)
            else Facial Landmarks Confirmed
                Embedder->>FAISS: Search 512D ArcFace Vector
                FAISS-->>Embedder: Return Nearest Neighbors & L2 Distances
                Embedder->>Engine: Trigger Callback handle_face_embedding_result
                Engine->>Engine: Consensus Voting & L2 Gate (d <= 0.68)
            end
        end
    end

    Engine->>Engine: Apply JitterFilter Bounding Box Deadzone
    Engine->>Web: Broadcast WebSocket Frame & Active Tracks Telemetry
```

---

## 📊 4. Database Schemas & State Registry

### A. Active Track Memory Registry (`active_tracks`)
```python
active_tracks = {
    1: "Ceaser",            # Confirmed Registered Identity (Green Box)
    2: "Scanning Track 2",  # Unconfirmed Active Scanning Target (Orange Box)
    3: "Unknown Person"     # Locked Unregistered Target (Red Box)
}
```

### B. SQLite Audit Event Table Schema (`data/faces.db`)
```sql
CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    track_id INTEGER NOT NULL,
    subject_name TEXT NOT NULL,
    l2_distance REAL NOT NULL,
    access_level TEXT DEFAULT 'GRANTED',
    snapshot_path TEXT
);
```

### C. Vector Index & Identity Metadata (`data/metadata.json`)
```json
[
  {
    "id": 0,
    "name": "Ceaser",
    "occupation": "Security Specialist",
    "access_level": "LEVEL 1 GRANTED"
  }
]
```

---

## 🔒 5. Multithreading & Hardware Safety Locks

To prevent OpenBLAS / OpenMP threading conflicts and C++ memory segfaults (Exit Code 139) on macOS:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

All shared memory operations between the camera thread, Uvicorn ASGI server, and `BatchFaceEmbedder` worker are synchronized using `threading.Lock()` (`state_lock`).

---

## 🧪 6. Automated Diagnostic Verification Suites

| Suite Name | Execution Command | Purpose | Expected Result |
| :--- | :--- | :--- | :--- |
| **Unit Test Suite** | `python3 test_pipeline.py` | Validates FAISS search, embedding extraction & database reloads. | **10/10 PASSED** |
| **Logic Gate Validator** | `python3 mock_pipeline_validator.py` | Tests L2 distance gating, margin thresholding & spatial uniqueness. | **5/5 PASSED** |
| **Crossover & Robustness** | `python3 test_crossover_handwave.py` | Tests spatial inheritance, occlusion recovery & camera cover flushes. | **6/6 PASSED** |
| **Stress Test Suite** | `python3 run_stress_tests.py` | Evaluates HOTA tracking accuracy, LocA localization & memory leaks. | **3/3 PASSED** |

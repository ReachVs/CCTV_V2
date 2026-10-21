# Enterprise CCTV Security & Biometrics AI Engine
## Comprehensive Technical Documentation, Architecture Specification & System Manual

---

## 📌 Table of Contents
1. [Executive Summary & System Purpose](#1-executive-summary--system-purpose)
2. [Technology Stack & Development Tools Catalog](#2-technology-stack--development-tools-catalog)
3. [Core Computer Vision & Mathematical Algorithms](#3-core-computer-vision--mathematical-algorithms)
   - [3.1 Motion Detection & Person Tracking](#31-motion-detection--person-tracking)
   - [3.2 Confidence-Based Kalman Filtering (CBKF)](#32-confidence-based-kalman-filtering-cbkf)
   - [3.3 Velocity-Adaptive Spatial Stabilization & Deadzone Locking](#33-velocity-adaptive-spatial-stabilization--deadzone-locking)
   - [3.4 Anatomical Pre-Gating & Limb Noise Suppression](#34-anatomical-pre-gating--limb-noise-suppression)
   - [3.5 Facial Super-Resolution & Canonical Umeyama Affine Alignment](#35-facial-super-resolution--canonical-umeyama-affine-alignment)
   - [3.6 Deep Biometric Embeddings (ArcFace & AdaFace)](#36-deep-biometric-embeddings-arcface--adaface)
   - [3.7 FAISS Vector Search & HNSW Indexing](#37-faiss-vector-search--hnsw-indexing)
   - [3.8 Multi-Frame Consensus Voting & Margin Gating](#38-multi-frame-consensus-voting--margin-gating)
   - [3.9 Crossover Contention & Anti-ID-Swapping Arbitration](#39-crossover-contention--anti-id-swapping-arbitration)
   - [3.10 Liveness & Anti-Spoofing Detection](#310-liveness--anti-spoofing-detection)
   - [3.11 Cryptographic Security & Privacy Enforcement](#311-cryptographic-security--privacy-enforcement)
4. [System Workflows](#4-system-workflows)
   - [4.1 Real-Time Video Ingestion & Tracking Pipeline](#41-real-time-video-ingestion--tracking-pipeline)
   - [4.2 Smart Tracker-Gating Lifecycle](#42-smart-tracker-gating-lifecycle)
   - [4.3 Standard 7-Pose Biometric Enrollment Protocol](#43-standard-7-pose-biometric-enrollment-protocol)
   - [4.4 Multi-Camera Ingestion & Hardware Lifecycle Management](#44-multi-camera-ingestion--hardware-lifecycle-management)
5. [System Architecture in Current State (Deep Seam Design)](#5-system-architecture-in-current-state-deep-seam-design)
   - [5.1 High-Level Architectural Flowchart](#51-high-level-architectural-flowchart)
   - [5.2 Deep Module & Seam Specifications](#52-deep-module--seam-specifications)
   - [5.3 State Machine: SubjectSession & TrackStatus](#53-state-machine-subjectsession--trackstatus)
   - [5.4 Presentation, API & Agent Fleet (MCP) Integration](#54-presentation-api--agent-fleet-mcp-integration)
6. [Directory Structure & Module Reference](#6-directory-structure--module-reference)
7. [Operating Commands & Quick Reference](#7-operating-commands--quick-reference)

---

## 1. Executive Summary & System Purpose

The **Enterprise CCTV Security & Biometrics AI Engine** is an end-to-end, high-throughput surveillance and biometric identity verification platform designed specifically for real-time edge processing on **Apple Silicon (CoreML / Metal MPS)** and heterogeneous hardware architectures (CUDA, Jetson TensorRT, CPU).

The system unifies deep learning object detection, continuous skeletal pose tracking, sub-millisecond facial landmark alignment, deep metric learning vector embeddings, and approximate nearest neighbor (ANN) graph search into an asynchronous, zero-latency streaming pipeline. 

### Key Architectural Strengths
- **Sub-Millisecond Tracking Latency**: Implements **Smart Tracker-Gating**, dropping per-frame compute from $25\text{ms}$ down to $<1.5\text{ms}$ once a subject's identity is locked.
- **Zero Spatial Jitter**: Employs velocity-adaptive Exponential Moving Average (EMA) smoothing and a $3.5\text{px}$ stationary deadzone lock, eliminating bounding-box flickering.
- **Anti-ID Swapping & Crossover Contention**: Detects track overlap ($IoU > 0.30$ or $Containment > 0.40$), freezes facial embedding during occlusions, and enforces a 3-frame spatial hysteresis and 45-frame identity blacklist.
- **Hermetic Deep-Seam Architecture**: Built using decoupled abstract seams (`PersonDetector`, `FaceEmbedder`, `AuditLogger`, `TrackRegistry`, `BiometricMatcher`, `IdentityConsensus`, `HUDAnnotator`), allowing instant zero-dependency unit testing and zero-overhead imports.
- **Encrypted Auditability**: Employs AES-256 Fernet Write-Ahead Logging (WAL) in SQLite for immutable, zero-delay event tracking.

---

## 2. Technology Stack & Development Tools Catalog

The table below catalogs every technology, framework, runtime tool, and library utilized within the project, detailing the precise role each serves in the architecture.

| Category | Technology / Tool | Version / Spec | Functional Role & Responsibility |
| :--- | :--- | :--- | :--- |
| **Core Runtime** | **Python** | 3.10+ / 3.11 | Primary language runtime hosting mathematical logic, asynchronous loops, and native bindings. |
| **Hardware Acceleration** | **Apple Silicon Metal (MPS)** | `torch.backends.mps` | Provides native GPU acceleration for PyTorch neural inference on macOS (M1/M2/M3/M4 Apple Silicon). |
| **Model Compiler** | **Apple CoreML Tools** | `coremltools >= 7.0` | Quantizes and packages YOLO and embedding models into native `.mlpackage` bundles for Apple Neural Engine (ANE). |
| **Computer Vision Engine** | **OpenCV (cv2)** | `>= 4.8.0` | Handles frame manipulation, affine transformation matrices, color-space conversion, JPEG encoding, and video I/O. |
| **Deep Object Detection** | **Ultralytics YOLO** | `YOLOv8n` / `YOLO11n` | Real-time person detection and 17-keypoint skeletal pose estimation (`yolo11n-pose.pt`, `yolov8n.pt`). |
| **Facial Landmark Alignment**| **YuNet ONNX** | `cv2.FaceDetectorYN` | Ultra-fast ($<4.0\text{ms}$) 5-point facial landmark detector executed via OpenCV DNN for eye, nose, and mouth corner coordinates. |
| **Super-Resolution** | **FSRCNN** | Fast Super-Resolution CNN | Upscales degraded face crops ($<60\text{px}$) to preserve biometric fidelity prior to vector embedding. |
| **Metric Face Recognition** | **DeepFace / ArcFace / AdaFace**| 512-Dimensional | Deep metric learning feature extractors producing quality-adaptive, $L_2$-normalized 512D unit vectors. |
| **Vector Similarity Search**| **FAISS (Facebook AI Similarity Search)**| `IndexHNSWFlat` / `IndexFlatL2` | Performs ultra-fast vector similarity graph search ($L_2$ distance evaluation) over enrolled facial identities in $<0.2\text{ms}$. |
| **Web & Streaming Framework**| **FastAPI + Uvicorn** | `fastapi >= 0.100`, `uvicorn >= 0.23` | High-performance ASGI web server hosting REST APIs and low-latency WebSockets video streaming. |
| **Asynchronous Streaming** | **WebSockets** | `websockets >= 11.0` | Real-time, bi-directional protocol streaming 720p base64/binary JPEG frames and telemetry data to web dashboards at 30+ FPS. |
| **Data Persistence** | **SQLite + WAL Mode** | Native SQLite3 | Local transactional persistence for biometric identity profiles and audit logs with Write-Ahead Logging enabled. |
| **Async Database Driver** | **aiosqlite** | `>= 0.19.0` | Asynchronous, non-blocking SQLite database driver preventing thread starvation during high-frequency telemetry writes. |
| **Cryptographic Security** | **Cryptography (Fernet)** | `>= 41.0.0` | AES-256-CBC with HMAC-SHA256 authenticated symmetric encryption securing audit log records at rest. |
| **Data Validation** | **Pydantic V2** | `>= 2.0` | Strict data validation, request parsing, and schema generation for API endpoints and engine configurations. |
| **AI Agent Interface** | **Model Context Protocol (FastMCP)**| FastMCP Spec | Exposes real-time active tracks, security alerts, and camera telemetry as callable tools for autonomous AI agents. |
| **Testing & Verification** | **unittest & pytest** | Standard Lib / pytest | Comprehensive test suite testing deep seams, mock pipelines, stress contention, and limb noise rejection. |
| **Containerization** | **Docker & Docker Compose** | Multi-Stage Build | Standardized Linux deployment wrapping runtime dependencies, model weights, and streaming services. |

---

## 3. Core Computer Vision & Mathematical Algorithms

### 3.1 Motion Detection & Person Tracking
The tracking foundation utilizes **ByteTrack** and **BoT-SORT** algorithms configured via [`bytetrack.yaml`](file:///Users/ceaser/Documents/School/CCTV%20Project/bytetrack.yaml) and [`botsort.yaml`](file:///Users/ceaser/Documents/School/CCTV%20Project/botsort.yaml).
- **Two-Stage Association**: High-confidence bounding boxes are associated first using spatial Intersection-over-Union (IoU) with Kalman predictions. Low-confidence detections are subsequently matched against unmatched tracklets, recovering occluded or blurred targets.
- **60-Frame Retention Buffer**: Lost tracks remain cached for 60 frames (2.0 seconds at 30 FPS). This prevents track ID fragmentation during temporary visual occlusions.

### 3.2 Confidence-Based Kalman Filtering (CBKF)
Standard Kalman filters assume constant measurement noise covariance $R$. In real-world surveillance, detection bounding boxes jitter when detection confidence drops. The system implements **Confidence-Based Kalman Filtering (CBKF)**, scaling measurement noise exponentially based on detector confidence:

$$R(\text{conf}) = R_{\text{base}} \cdot \exp\left(-\beta \cdot (\text{conf} - c_0)\right)$$

- When confidence is high ($\text{conf} \ge 0.85$), $R$ is small, and the filter trusts the incoming detection.
- When confidence drops (e.g. during a turn or partial occlusion), $R$ surges, compelling the Kalman filter to trust its internal motion velocity model, eliminating spatial jumping.

### 3.3 Velocity-Adaptive Spatial Stabilization & Deadzone Locking
Stationary individuals often suffer from bounding-box oscillation due to pixel quantization noise. The `JitterFilterEngine` ([`src/utils/jitter_filter.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/utils/jitter_filter.py)) applies a multi-stage spatial stabilization algorithm:

1. **Center Displacement Velocity**:
   $$v_t = \sqrt{(cx_t - cx_{t-1})^2 + (cy_t - cy_{t-1})^2}$$

2. **Zero-Jitter Stationary Deadzone ($3.5\text{ px}$)**:
   $$\hat{B}_t = \hat{B}_{t-1} \quad \text{if } v_t < 3.5\text{ px/frame}$$
   If displacement velocity is beneath $3.5\text{ px}$, the bounding box coordinates are completely locked, eliminating all micro-wobble.

3. **Velocity-Adaptive Smoothing Factor ($\alpha$)**:
   $$\alpha = \min\left(0.85, \max\left(0.08, \frac{v_t - 1.5}{16.0}\right)\right)$$
   For slow movements, $\alpha \to 0.08$ (heavy smoothing); for fast running, $\alpha \to 0.85$ (rapid tracking without latency).

4. **Exponential Moving Average (EMA) Update**:
   $$\hat{p}_t = (1 - \alpha)\hat{p}_{t-1} + \alpha p_t, \quad \forall p \in \{x_1, y_1, x_2, y_2\}$$

5. **Dimension Hysteresis (80/20 Inertia)**:
   Width and height maintain an 80/20 temporal memory to prevent box aspect-ratio flickering:
   $$W_t = 0.80 \cdot W_{t-1} + 0.20 \cdot w_t, \quad H_t = 0.80 \cdot H_{t-1} + 0.20 \cdot h_t$$

### 3.4 Anatomical Pre-Gating & Limb Noise Suppression
To prevent isolated arms, hands, legs, or background clutter from triggering false face-scanning boxes, the `TrackingPreFilter` ([`src/inference/tracking_pre_filter.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/tracking_pre_filter.py)) enforces **Head-Shoulder Pre-Gating**:
- Uses YOLO pose estimation to evaluate the presence of COCO keypoints $0 \dots 6$ (nose, left/right eyes, left/right ears, left/right shoulders).
- A candidate bounding box is rejected from biometric scanning unless:
  $$\max_{k \in \{0\dots 6\}} \text{confidence}(k) \ge 0.35 \quad \text{OR} \quad \text{YuNet detects facial landmarks within box}$$
- Completely suppresses phantom tracks originating from waving hands or limb movements.

### 3.5 Facial Super-Resolution & Canonical Umeyama Affine Alignment
1. **FSRCNN Super-Resolution**: Face crops smaller than $60 \times 60\text{ px}$ are upscaled using a 4-layer Fast Super-Resolution Convolutional Neural Network before feature extraction.
2. **YuNet 5-Point Landmark Detection**: Locates coordinates for:
   $$\{\text{Left Eye}, \text{Right Eye}, \text{Nose Tip}, \text{Mouth Left}, \text{Mouth Right}\}$$
3. **Umeyama 2D Similarity Affine Transformation**:
   Computes the optimal transformation matrix $M \in \mathbb{R}^{2 \times 3}$ that minimizes the least-squares error between observed landmarks and canonical $112 \times 112$ ArcFace template coordinates:
   $$M^* = \arg\min_M \sum_{i=1}^5 \|M \cdot [x_i, y_i, 1]^T - [u_i, v_i]^T\|^2$$
   This scales, rotates, and normalizes the face crop into a standardized frontal pose.

### 3.6 Deep Biometric Embeddings (ArcFace & AdaFace)
Facial crops are passed to deep metric learning models (AdaFace/ArcFace).
- **Quality-Adaptive Margin**: AdaFace incorporates image quality directly into the margin function. For unconstrained CCTV frames with blur or low illumination, it uses the feature norm $\|z\|_2$ as a real-time quality proxy to attenuate noise.
- **Unit Hypersphere Projection**:
  $$v_{\text{norm}} = \frac{v}{\|v\|_2} \in \mathbb{R}^{512}, \quad \|v_{\text{norm}}\|_2 = 1.0$$
  Mapping representations to a 512-dimensional unit hypersphere allows distance metrics to be computed efficiently via Euclidean distance:
  $$d_{L_2}(u, v) = \sqrt{2 - 2(u \cdot v)}$$

### 3.7 FAISS Vector Search & HNSW Indexing
Vectors are queried against a **FAISS** (Facebook AI Similarity Search) index:
- **Hierarchical Navigable Small World (`IndexHNSWFlat`)**: A multi-layer graph index providing logarithmic time complexity $\mathcal{O}(\log N)$ for approximate nearest neighbor search with sub-millisecond query latency ($<0.2\text{ms}$).
- **Exact L2 Fallback (`IndexFlatL2`)**: Used in small gallery databases for zero-approximation exact nearest neighbor retrieval.

### 3.8 Multi-Frame Consensus Voting & Margin Gating
To eliminate biometric false positives, `IdentityConsensus` ([`src/inference/consensus.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/consensus.py)) enforces four strict arbitration policies:
1. **Pose-Aware Dynamic Distance Thresholds**:
   - Frontal Pose: $d_{L_2} \le 0.68$
   - Side-Profile Pose: $d_{L_2} \le 0.30$ (stricter threshold prevents identity cross-contamination on profile yaw).
2. **Top-1 vs Top-2 Margin Gating**:
   Let $d_1$ be the distance to the best match and $d_2$ be the distance to the second-best match of a *different* identity:
   $$\Delta_{\text{margin}} = d_2 - d_1 \ge 0.15$$
   If $\Delta_{\text{margin}} < 0.15$, the match is flagged as ambiguous and rejected.
3. **Multi-Frame Consensus Voting**:
   Requires $N=2$ matching identity votes within a sliding window of 5 frames before identity is locked.
4. **Side-Profile Consensus Guardrail**:
   A side-profile detection cannot lock an identity on its own; it requires at least one frontal or quarter-view corroborating vote.
5. **Exclusive Identity Claim Leases**:
   Once an identity is assigned to active `track_id` $A$, no concurrent track $B$ can claim that identity, preventing concurrent ID collisions.

### 3.9 Crossover Contention & Anti-ID-Swapping Arbitration
When two subjects walk past one another, bounding boxes intersect, often triggering identity swapping in standard trackers. The `TrackingPreFilter` and `IdentityConsensus` implement a rigorous anti-ID-swapping protocol:
1. **Overlap Trigger**:
   Monitors pairwise bounding box spatial metrics:
   $$\text{IoU}(B_1, B_2) > 0.30 \quad \text{OR} \quad \text{Containment}(B_1, B_2) > 0.40$$
   $$\text{Containment}(A, B) = \frac{\text{Area}(A \cap B)}{\min(\text{Area}(A), \text{Area}(B))}$$
2. **Contention Freezing**:
   When triggered, both tracks enter the `CONTENDED` state. Face extraction and biometric matching are **frozen** during overlap to prevent reading the other person's face.
3. **Spatial Separation Hysteresis**:
   After separation, extraction remains frozen until both tracks maintain $\text{IoU} < 0.15$ for at least 3 consecutive frames.
4. **Spatial Blacklist**:
   Past spatial history from collided tracks is blacklisted from inheritance for 45 frames.

### 3.10 Liveness & Anti-Spoofing Detection
To counter presentation attacks (e.g. holding up a smartphone photo or printed sheet of paper):
1. **Laplacian Variance Analysis**: Measures high-frequency edge sharpness to detect printed or blurred artifacts:
   $$\sigma^2 = \text{Var}\left(\nabla^2 f(x, y)\right)$$
2. **2D FFT Frequency Moire Spectrum Analysis**: Computes 2D Fast Fourier Transform over the face crop. Digital LCD/OLED screens exhibit periodic high-frequency grid peaks (moire patterns) in the frequency domain that are absent in live human skin.

### 3.11 Cryptographic Security & Privacy Enforcement
1. **AES-256 Fernet WAL Encryption**: Security audit records written to [`data/faces.db`](file:///Users/ceaser/Documents/School/CCTV%20Project/data/faces.db) are encrypted using AES-256 in CBC mode with an HMAC-SHA256 signature via `EncryptedWALAuditLogger`.
2. **GDPR Memory Sanitizer**: If unconfirmed or temporary face crops remain unverified after attempt exhaustion, they are cryptographically zeroed out in memory and purged.
3. **Pose-Only Mode**: A privacy-preserving toggle that completely bypasses face detection, crops, and vector embeddings, rendering solely 17-keypoint anonymous skeletal stick figures.

---

## 4. System Workflows

### 4.1 Real-Time Video Ingestion & Tracking Pipeline

```
  [Camera / RTSP Feed]
          │
          ▼
┌────────────────────────────────────────┐
│  CameraIngestionPool                   │
│  (Bufferless read thread, 0ms lag)     │
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│  PersonDetector Seam                   │
│  (YOLOv8n / YOLO11n / Pose Estimator)  │
└────────────────────────────────────────┘
          │ Raw Bounding Boxes & Pose Keypoints
          ▼
┌────────────────────────────────────────┐
│  TrackingPreFilter                     │
│  - Head/Shoulder anatomical gating     │
│  - Bounding box NMS & sub-box filter   │
│  - Zero-Jitter deadzone (3.5px) & EMA  │
│  - Crossover contention arbitration    │
└────────────────────────────────────────┘
          │ Filtered Track Candidates
          ▼
┌────────────────────────────────────────────────────────┐
│  TrackRegistry & SubjectSession State Machine          │
│  (Thread-Safe RLock, single source of truth)           │
└────────────────────────────────────────────────────────┘
     │                                    │
     │ [If Status == CONFIRMED]           │ [If Status == SCANNING]
     ▼                                    ▼
┌───────────────────────────┐   ┌────────────────────────────────────────┐
│  Tracker-Gating Bypass    │   │  EmbeddingBatchWorker Queue            │
│  (Compute: < 1.5ms)       │   │  - FSRCNN super-resolution             │
└───────────────────────────┘   │  - YuNet 5-point Umeyama alignment     │
     │                          │  - ArcFace/AdaFace 512D embedding      │
     │                          └────────────────────────────────────────┘
     │                                    │
     │                                    ▼
     │                          ┌────────────────────────────────────────┐
     │                          │  BiometricMatcher & VectorIndexAdapter │
     │                          │  (FAISS HNSW L2 distance search)       │
     │                          └────────────────────────────────────────┘
     │                                    │
     │                                    ▼
     │                          ┌────────────────────────────────────────┐
     │                          │  IdentityConsensus Engine              │
     │                          │  - Dynamic pose thresholds (<0.68/0.30)│
     │                          │  - Margin gating (delta >= 0.15)       │
     │                          │  - 2-vote consensus over 5 frames      │
     │                          │  - Side-profile corroboration check    │
     │                          │  - Exclusive identity leases           │
     │                          └────────────────────────────────────────┘
     │                                    │
     ├────────────────────────────────────┘
     ▼
┌────────────────────────────────────────┐
│  CyberpunkHUDAnnotator                 │
│  - Corner brackets & status palettes   │
│  - Skeletal stick figures & telemetry  │
└────────────────────────────────────────┘
     │
     ├──► [FastAPI WebSockets Stream ws://localhost:8000/ws/video] ──► HTML5 Dashboard
     │
     └──► [EncryptedWALAuditLogger (Non-blocking queue)] ────────────► data/faces.db
```

### 4.2 Smart Tracker-Gating Lifecycle
To achieve 30+ FPS real-time performance on high-resolution streams, biometric extraction is not executed continuously on every frame:

```
[New Track Appears]
       │
       ▼
[Status: SCANNING] ────► Run Facial Extraction & Vector Search every N frames (Subsampling)
       │
       ▼
[Consensus Achieved: 2 Votes + Margin Gate Passed]
       │
       ▼
[Status: CONFIRMED] ───► Face extraction & FAISS matching 100% BYPASSED!
       │                 Track maintained exclusively via ByteTrack + Spatial Smoothing (<1.5ms)
       │
       ▼ (If Overlap / Contention Occurs)
[Status: CONTENDED] ───► Invalidate confirmation, freeze extraction during overlap
       │
       ▼ (After Spatial Separation Hysteresis: IoU < 0.15 for 3 frames)
[Re-enter SCANNING] ───► Re-verify identity consensus before re-locking
```

### 4.3 Standard 7-Pose Biometric Enrollment Protocol
High-accuracy biometric identification requires robust reference templates across diverse viewing angles. The system implements a **Standard 7-Pose Template Registration Protocol** ([`enroll_webcam.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/enroll_webcam.py)):

1. **Frontal ($0^\circ$)**: Direct gaze into the camera lens. Acts as the primary anchor for frontal consensus verification.
2. **Working_Angle**: Natural working posture oriented towards the computer screen.
3. **Left_Angle ($\sim 25^\circ$)**: Gentle yaw rotation to the subject's left (both eyes visible).
4. **Side_Left_Angle ($\sim 50^\circ$)**: Extended profile yaw to the subject's left (far right eye stays in FOV).
5. **Right_Angle ($\sim 25^\circ$)**: Gentle yaw rotation to the subject's right (both eyes visible).
6. **Side_Right_Angle ($\sim 50^\circ$)**: Extended profile yaw to the subject's right (far left eye stays in FOV).
7. **Tilt_Up ($\sim 15^\circ$)**: Slight upward pitch elevation.

Each angle enforces YuNet 5-point landmark quality validation before saving. Upon completion, the system automatically reconstructs the FAISS vector index and updates the SQLite metadata store.

### 4.4 Multi-Camera Ingestion & Hardware Lifecycle Management
Managed by `CameraIngestionPool` ([`src/inference/multi_camera.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/multi_camera.py)):
- **Reference-Counted Acquisition (`acquire()` / `release()`)**: Tracks how many consumers (WebSocket streams, GUI windows, test runners) are using a given camera device index or RTSP URL.
- **3.0-Second Deferred Teardown**: When a browser client disconnects or refreshes, the camera feed is not immediately destroyed. A 3.0-second grace timer holds hardware buffers open, eliminating camera re-opening thrash.
- **Hardware Isolation for Enrollment**: When enrollment begins, the ingestion pool pauses background surveillance streams to prevent USB bus or hardware contention.
- **Synthetic Fallback**: If a camera fails or disconnects, the pool seamlessly emits test card frames to keep downstream pipelines alive without throwing unhandled exceptions.

---

## 5. System Architecture in Current State (Deep Seam Design)

### 5.1 High-Level Architectural Flowchart

```mermaid
flowchart TD
    subgraph INGESTION["1. Hardware & Ingestion Seam"]
        CAM["Webcam / RTSP / Video Feed"] --> CIP["CameraIngestionPool (Ref-Counted, Bufferless)"]
    end

    subgraph DETECTION["2. Detection & Pre-Filter Seam"]
        CIP --> PD["PersonDetector Seam (Ultralytics / CoreML / Haar)"]
        PD --> TPF["TrackingPreFilter (CBKF, 3.5px Deadzone, EMA, Head/Shoulder Gating)"]
    end

    subgraph STATE["3. State Management & Lifecycle Registry"]
        TPF --> TR["TrackRegistry (Authoritative Single Source of Truth, RLock)"]
        TR --> SS["SubjectSession Entities (SCANNING, CONFIRMED, CONTENDED, UNVERIFIED)"]
    end

    subgraph BIOMETRICS["4. Asynchronous Biometric Extraction Seam"]
        SS -->|Unconfirmed Track| EBW["EmbeddingBatchWorker Queue"]
        EBW --> FE["FaceEmbedder Seam (FSRCNN + YuNet 5-Point + ArcFace 512D)"]
        FE --> BM["BiometricMatcher Seam"]
        BM --> VIA["VectorIndexAdapter (FAISS HNSW Index)"]
        VIA --> IC["IdentityConsensus Engine (Voting, Leases, Margin Gating)"]
        IC -->|Confirm ID| TR
    end

    subgraph PRESENTATION["5. Presentation & Streaming Seam"]
        TR --> HA["HUDAnnotator Seam (CyberpunkHUDAnnotator / NoOp)"]
        HA --> API["FastAPI ASGI Server & WebSockets (/ws/video)"]
        HA --> MCP["Model Context Protocol (MCP Fleet Server)"]
        TR --> AL["AuditLogger Seam (EncryptedWALAuditLogger, AES-256)"]
    end
```

### 5.2 Deep Module & Seam Specifications

The codebase is organized around **Deep Modules**—abstractions that hide substantial complexity behind clean, minimalist seams:

#### 1. `PersonDetector` Seam ([`src/inference/person_detector.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/person_detector.py))
- **Interface**: `detect_and_track(frame) -> List[DetectedBox]`
- **Implementations**:
  - `UltralyticsPersonDetector`: Encapsulates YOLOv8/YOLO11 neural models and ByteTrack; includes automatic OpenCV Haar cascade fallback if weights are missing.
  - `ScriptedPersonDetector`: Deterministic, model-free test double for hermetic unit testing without neural model loading overhead.

#### 2. `TrackingPreFilter` Deep Module ([`src/inference/tracking_pre_filter.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/tracking_pre_filter.py))
- **Interface**: `filter_candidates(boxes, raw_boxes, frame, pose_model) -> PreFilterResult`
- **Encapsulates**:
  - Velocity calculation & $3.5\text{px}$ stationary deadzone lock.
  - Dynamic EMA smoothing ($\alpha = 0.08 \dots 0.85$).
  - COCO keypoint $0\dots 6$ anatomical presence verification.
  - Spatial IoU & Containment sub-box suppression.
  - Pairwise crossover contention detection and separation tracking.

#### 3. `TrackRegistry` & `SubjectSession` ([`src/domain/track_registry.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/domain/track_registry.py), [`src/domain/subject_session.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/domain/subject_session.py))
- **Role**: The authoritative single source of truth for all observed subjects.
- **Encapsulates**:
  - Internal `threading.RLock()` ensuring thread safety across video streaming, API queries, and MCP servers.
  - Eliminates previously scattered dictionary tables (`active_tracks`, `track_cache`, `track_last_seen`, `track_first_seen`, `track_keypoints`).
  - Provides thread-safe backward-compatible dictionary proxy facades (`ActiveTracksProxy`, `TrackCacheProxy`).

#### 4. `FaceEmbedder` Seam ([`src/inference/face_embedder.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/face_embedder.py))
- **Interface**: `extract_embedding(face_crop) -> Optional[np.ndarray]`
- **Implementations**:
  - `ArcFaceEmbedder`: Chains FSRCNN super-resolution, YuNet 5-point Umeyama alignment, and 512D unit-norm vector extraction.
  - `ScriptedFaceEmbedder`: Deterministic mock embedder for lightning-fast hermetic test execution.

#### 5. `BiometricMatcher` Seam ([`src/inference/biometric_matcher.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/biometric_matcher.py))
- **Interface**: `match_embedding(embedding, pose_type) -> Optional[MatchCandidate]`
- **Encapsulates**:
  - Queries `VectorIndexAdapter` (`FAISSIndexAdapter` or `InMemoryIndexAdapter`).
  - Applies pose-dependent distance thresholds ($d < 0.68$ frontal, $d < 0.30$ side-profile).
  - Evaluates inter-subject top-1 vs top-2 competition margins ($\Delta \ge 0.15$).

#### 6. `IdentityConsensus` Seam ([`src/inference/consensus.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/consensus.py))
- **Interface**: `record_vote(track_id, candidate, yaw_angle) -> ConsensusDecision`
- **Encapsulates**:
  - Sliding temporal voting window ($N=2$ within 5 frames).
  - Side-profile corroboration safety check.
  - Identity lease locking to prevent cross-track duplicate claims.
  - Automatic transition to `UNKNOWN` after attempt exhaustion ($12\text{ attempts}$).

#### 7. `AuditLogger` Seam ([`src/database/audit_repository.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/database/audit_repository.py))
- **Interface**: `log_event(...)`, `get_recent_events(...)`, `get_stats(...)`
- **Implementations**:
  - `EncryptedWALAuditLogger`: Non-blocking queue worker that encrypts audit payloads using AES-256 Fernet and flushes them to SQLite in WAL mode with $<0.1\text{ms}$ latency impact on video processing.
  - `ScriptedAuditLogger`: In-memory test spy for validation without touching disk.

#### 8. `HUDAnnotator` Seam ([`src/rendering/annotator.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/rendering/annotator.py))
- **Interface**: `annotate_frame(frame, tracks, telemetry) -> np.ndarray`
- **Implementations**:
  - `CyberpunkHUDAnnotator`: Renders tactical corner brackets, glowing status badges, skeletal pose sticks, and top telemetry bars.
  - `NoOpAnnotator`: Zero-copy, zero-overhead pass-through for headless servers and CLI benchmarks.

#### 9. `CompositionRoot` Lazy Factory ([`src/inference/live_cctv.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/inference/live_cctv.py))
- Implements **PEP 562** module `__getattr__` and `ModuleType` proxying.
- Eliminates the 4-second eager neural network import penalty when importing utility scripts or launching CLI tools. The heavy `BiometricTrackingEngine` is only instantiated when `.engine` or `get_engine()` is accessed for the first time.

### 5.3 State Machine: SubjectSession & TrackStatus

Each observed subject is modeled as an active `SubjectSession` transitioning through explicit lifecycle states:

```
                  ┌────────────────────────┐
                  │        SCANNING        │ ◄─────────────────────────┐
                  └───────────┬────────────┘                           │
                              │                                        │
           ┌──────────────────┴──────────────────┐                     │
           │ Consensus Met                       │ Exhausted           │
           ▼                                     ▼                     │
┌────────────────────────┐            ┌────────────────────────┐       │
│       CONFIRMED        │            │        UNKNOWN         │       │
└──────────┬─────────────┘            └────────────────────────┘       │
           │                                                           │
           │ Contention (IoU > 0.30 or Containment > 0.40)             │
           ▼                                                           │
┌────────────────────────┐                                             │
│       CONTENDED        │ ────────────────────────────────────────────┘
└────────────────────────┘   Separated (IoU < 0.15 for 3 frames)
```

| `TrackStatus` | HUD Visualization | Engine Behavior |
| :--- | :--- | :--- |
| `SCANNING` | Cyan / Yellow Pulsing Box | Actively cropping faces; queueing embeddings; tallying consensus votes. |
| `CONFIRMED` | Neon Green Corners + Name Badge | Identity locked; **facial extraction 100% bypassed**; motion tracking only. |
| `CONTENDED` | Amber Alert Box + Warning Banner | Overlap detected; face extraction **frozen** to prevent identity cross-contamination. |
| `UNKNOWN` | Crimson Red Box + "UNKNOWN" | Max voting attempts exhausted without match; logged to security audit. |
| `UNVERIFIED` | Subtle Grey Box | Unconfirmed candidate awaiting upper-body anatomical presence validation. |
| `POSE_TARGET`| Magenta Skeletal Stick Figure | Privacy-compliant Pose-Only Mode; biometric extraction fully bypassed. |

### 5.4 Presentation, API & Agent Fleet (MCP) Integration

#### FastAPI Web Server ([`src/api/cctv_server.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/api/cctv_server.py))
- **`/ws/video`**: High-performance WebSocket stream broadcasting base64/binary JPEG frames paired with real-time JSON telemetry payloads.
- **`/api/enroll`**: REST endpoint initiating the 7-pose webcam enrollment wizard.
- **`/api/audit_logs`**: Async endpoint querying encrypted SQLite audit records using `aiosqlite`.
- **`/api/cameras`**: Configures multi-camera feeds and queries hardware status.

#### Model Context Protocol (MCP) Fleet Intelligence ([`src/api/mcp_server.py`](file:///Users/ceaser/Documents/School/CCTV%20Project/src/api/mcp_server.py))
Exposes the surveillance engine to AI coding assistants and autonomous agents via standard MCP tools:
- `get_active_tracks()`: Returns active subject IDs, coordinates, and verified identities.
- `get_audit_events(limit)`: Retrieves recent security verification events and unrecognized intruder alerts.
- `get_system_telemetry()`: Inspects hardware utilization, FPS, active camera status, and vector gallery counts.

---

## 6. Directory Structure & Module Reference

```text
CCTV Project/
├── data/                                 # Data storage & biometric indexes
│   ├── faces.db                          # Encrypted SQLite audit database (WAL mode)
│   ├── faces.index                       # FAISS 512D HNSW vector index
│   ├── metadata.json                     # Vector ID to Subject Name mapping
│   └── known_faces/                      # Enrolled subject source imagery
├── src/
│   ├── api/                              # Network & presentation layer
│   │   ├── cctv_server.py                # Primary FastAPI + WebSockets ASGI application
│   │   ├── mcp_server.py                 # FastMCP Model Context Protocol fleet interface
│   │   └── schemas.py                    # Pydantic V2 request & telemetry schemas
│   ├── config/                           # System configuration
│   │   └── model_config.py               # Hardware targets (MPS/CPU/CUDA) & model specifications
│   ├── database/                         # Persistence & vector storage layer
│   │   ├── audit_repository.py           # EncryptedWALAuditLogger (AES-256 Fernet + SQLite WAL)
│   │   ├── biometric_repository.py       # Subject metadata and profile storage
│   │   ├── vector_store.py               # FAISSIndexAdapter & InMemoryIndexAdapter
│   │   ├── db_setup.py                   # Database schema initialization
│   │   └── enroll.py                     # Batch vector enrollment pipeline
│   ├── domain/                           # Core business domain models
│   │   ├── subject_session.py            # SubjectSession entity & TrackStatus state machine
│   │   └── track_registry.py             # Thread-safe TrackRegistry seam & proxy facades
│   ├── inference/                        # Core AI inference & computer vision engine
│   │   ├── engine.py                     # BiometricTrackingEngine deep orchestrator
│   │   ├── live_cctv.py                  # Lazy CompositionRoot factory (PEP 562 zero-overhead)
│   │   ├── person_detector.py            # PersonDetector seam (Ultralytics + Haar fallback)
│   │   ├── tracking_pre_filter.py        # TrackingPreFilter (Deadzone, EMA, Head-Shoulder gating)
│   │   ├── face_embedder.py              # FaceEmbedder seam (FSRCNN + YuNet + ArcFace)
│   │   ├── embedding_worker.py           # Asynchronous EmbeddingBatchWorker queue
│   │   ├── biometric_matcher.py          # BiometricMatcher seam (pose-aware thresholding)
│   │   ├── consensus.py                  # IdentityConsensus seam (voting, leases, margin gating)
│   │   ├── multi_camera.py               # CameraIngestionPool (bufferless, deferred teardown)
│   │   └── temporal_subsampler.py        # Per-track frame subsampling optimizer
│   ├── rendering/                        # Visual presentation layer
│   │   └── annotator.py                  # CyberpunkHUDAnnotator, NoOpAnnotator seams
│   ├── services/                         # Application domain services
│   │   └── biometric_enrollment_manager.py # Guided 7-pose template registration coordinator
│   └── utils/                            # Low-level computer vision utilities
│       ├── face_align.py                 # YuNet 5-landmark detector & Umeyama affine transform
│       ├── fsrcnn_upscaler.py            # FSRCNN super-resolution pre-processor
│       ├── jitter_filter.py              # JitterFilterEngine (Velocity deadzone & EMA)
│       ├── privacy_compliance.py         # GDPRMemorySanitizer & PoseOnlyEstimator
│       ├── tune_tracker.py               # ByteTrack hyperparameter grid search optimizer
│       └── export_coreml.py              # CoreML model conversion utility
├── templates/                            # Web dashboard user interface
│   ├── dashboard.html                    # Real-time HTML5 tactical monitoring dashboard
│   └── enroll.html                       # Interactive 7-pose face enrollment wizard
├── tests/                                # Comprehensive test suite
│   ├── test_person_detector_seam.py      # PersonDetector seam & fallback verification
│   ├── test_face_embedder_seam.py        # FaceEmbedder seam & vector extraction tests
│   ├── test_identity_consensus.py        # Multi-frame voting & margin gating tests
│   ├── test_crossover_contention_resolution.py # Anti-ID-swapping contention tests
│   ├── test_limb_noise_suppression.py    # Head-Shoulder anatomical pre-gating tests
│   ├── test_audit_logger_seam.py         # Encrypted WAL audit logging tests
│   └── test_camera_ingestion_pool.py     # Bufferless ingestion & ref-count tests
├── Dockerfile                            # Production container build specification
├── docker-compose.yml                    # Multi-service container orchestration
├── requirements.txt                      # Pinned Python package dependencies
└── README.md                             # High-level project README
```

---

## 7. Operating Commands & Quick Reference

### 1. Environment Setup
```bash
# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify Apple Silicon Metal (MPS) GPU acceleration
python3 -c "import torch; print('MPS Available:', torch.backends.mps.is_available())"
```

### 2. Database Initialization
```bash
# Initialize SQLite database with WAL mode and audit tables
python3 -m src.database.db_setup
```

### 3. Face Enrollment
```bash
# Option A: Guided interactive 7-pose webcam wizard (Recommended)
python3 enroll_webcam.py

# Option B: Batch enrollment from images in data/known_faces/
python3 -m src.database.enroll
```

### 4. Running the System
```bash
# Launch FastAPI + WebSockets Real-Time Web Dashboard (Recommended)
python3 cctv_server.py

# Alternative: Direct module invocation
python3 -m src.api.cctv_server

# Run with a pre-recorded test video feed
CAMERA_SOURCE=stress_occlusion.mp4 python3 cctv_server.py

# Run with a specific webcam index (e.g., external camera 1)
CAMERA_SOURCE=1 python3 cctv_server.py

# Run with an RTSP network IP camera
CAMERA_SOURCE="rtsp://admin:password@192.168.1.100:554/stream1" python3 cctv_server.py

# Run on a custom port
PORT=8080 python3 cctv_server.py
```

**Accessing the System:**
* 🖥️ **Live Web Dashboard:** [http://localhost:8000](http://localhost:8000)
* 👤 **Web Face Enrollment:** [http://localhost:8000/enroll](http://localhost:8000/enroll)
* 📑 **Interactive API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)
* 📡 **WebSocket Video Stream:** `ws://localhost:8000/ws/video`


### 5. Running the Test Suite
```bash
# Run all unit tests
python3 -m unittest discover -s tests -p "test_*.py"

# Run specific deep seam test
python3 -m unittest test_person_detector_seam.py
```

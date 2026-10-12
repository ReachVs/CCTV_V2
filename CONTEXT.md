# CCTV Biometric Security Domain Context

This glossary defines the ubiquitous language for the real-time CCTV biometric security engine.

## Core Domain Terms

### BiometricTrackingEngine
The deep module responsible for end-to-end motion detection, spatial stabilization, face alignment, biometric embedding extraction, vector search, consensus identity locking, and active track state management behind a single seam.

### TrackedSubject
An entity observed across continuous video frames by the motion tracking engine, identified by a stable integer `track_id`, carrying spatial bounding coordinates, confidence metrics, and identity verification status.

### ProcessResult
The immutable output contract returned by `BiometricTrackingEngine.process_frame(frame, annotate=True)`. Contains the annotated frame, the active list of `TrackedSubject` records, and any newly triggered audit events.

### Tracker-Gating
The optimization policy where biometric face alignment and vector extraction are completely bypassed for a `TrackedSubject` once consensus verification is confirmed, reducing per-frame compute to motion tracking only.

### IdentityConsensus
The dedicated biometric consensus and collision arbitration seam (`IdentityConsensus`, `MatchCandidate`, `ConsensusDecision` in `src.inference.consensus`). Encapsulates multi-frame voting ($N=2$ matching votes within a 5-frame window), top-1 vs top-2 inter-subject margin gating ($\ge 0.15$), side-profile yaw consensus guardrails (requiring frontal or quarter-view corroboration before identity lock-in), active identity claim leases to prevent ID-swapping across concurrent tracks, and automatic transition to confirmed `Unknown Person` status after attempt exhaustion. Fully decouples identity verification from video ingestion and frame rendering.

### VectorIndexAdapter
The interface abstraction for vector similarity search. In production, this uses FAISS $L_2$ / HNSW vector indexes; in hermetic unit testing, an in-memory numpy test adapter is injected to prevent `sys.modules` monkeypatching.

### PoseOnlyMode
A privacy-compliant operating state where facial recognition and biometric vector embeddings are fully bypassed. Bounding boxes and 17-keypoint skeletal structures are tracked anonymously, and unconfirmed face crops are purged via `GDPRMemorySanitizer`.

### CameraIngestionPool
The unified hardware and multi-stream ingestion module consolidating bufferless frame acquisition across local webcams, RTSP streams, and multi-camera arrays (`CameraIngestionPool`, aliasing `MultiCameraIngestionManager` and `GlobalCameraStream`). Incorporates thread-safe reference-counting acquisition (`acquire()` / `release()`), a 3.0-second deferred teardown timer to eliminate camera re-opening thrash across browser reconnects, pause/resume hardware arbitration for biometric enrollment isolation, and automatic synthetic fallback frame generation during hardware disconnection or warmup periods.

### CrossoverContention
The anti-ID-swapping protocol that monitors pairwise track overlap (IoU > 0.30 or containment > 0.40). When triggered, consensus certainty is invalidated (`is_confirmed=False`, `is_contended=True`), face extraction is frozen during occlusion to prevent cross-subject contamination, a 3-frame spatial separation hysteresis (`IoU < 0.15`) is enforced before unfreezing extraction, and recent track histories are blacklisted from spatial inheritance for 45 frames.

### HeadShoulderPreGating
The geometric and biometric pre-admission protocol requiring upper-body or head presence (either pose keypoints 0..6 [nose, eyes, ears, shoulders] with $\ge 0.35$ confidence, or detected facial landmarks via YuNet) before admitting unconfirmed candidate boxes into `active_tracks`. Isolated arms, hands, legs, or torso-less limb noise are rejected prior to biometric queueing and scanning UI instantiation.

### SubjectSession
The cohesive domain entity encapsulating the complete lifecycle, temporal bounding box history, biometric consensus voting tally, contention hysteresis, and pose metadata for an active track ID, consolidating previously scattered tracking state tables.

### TrackStatus
The explicit lifecycle state machine enum (`SCANNING`, `CONFIRMED`, `CONTENDED`, `UNVERIFIED`, `UNKNOWN`, `POSE_TARGET`) governing biometric gating, HUD visualization color palettes, and identity locking transitions without string-parsing heuristics.

### TrackRegistry
The deep state management and lifecycle registry seam (`TrackRegistry` in `src.domain.track_registry`). Encapsulates all active `SubjectSession` instances, thread-safe access (via internal `threading.RLock`), spatial coordinate updates, stale track garbage collection, and status-based query facades. Eliminates fragmented parallel dictionary tables (`active_tracks`, `track_cache`, `track_last_seen`, `track_first_seen`, `track_keypoints`) in `BiometricTrackingEngine` and serves as the authoritative single source of truth for all observed tracks.

### PersonDetector
The explicit motion and person detection seam (`PersonDetector`, `UltralyticsPersonDetector`, `ScriptedPersonDetector`, `DetectedBox`). Decouples `BiometricTrackingEngine` from Ultralytics YOLO models and encapsulates traditional Haar cascade fallback. Enables hermetic, model-free unit testing without monkeypatching private engine internals.

### FaceEmbedder
The unified biometric vector extraction seam (`FaceEmbedder`, `ArcFaceEmbedder`, `ScriptedFaceEmbedder`). Unifies facial super-resolution, YuNet 5-point landmark alignment, neural inference, and $L_2$ vector normalization across live tracking, enrollment, and database rebuild pipelines.
### AuditLogger
The unified security audit logging seam (`AuditLogger`, `EncryptedWALAuditLogger`, `ScriptedAuditLogger`, aliasing `AuditRepository` and `AESEncryptedWALAuditLogger`). Consolidates AES-256 Fernet payload encryption at rest, SQLite WAL mode, non-blocking queue ingestion for live video streaming (<0.1ms latency), and dual sync/async query APIs across FastAPI endpoints and MCP servers.

### CompositionRoot
The canonical, thread-safe lazy factory and dependency orchestrator (`get_engine()`, `set_engine()`, `reset_engine()`) in `src.inference.live_cctv`. Implements PEP 562 module `__getattr__` and `ModuleType` wrapper to eliminate 4-second eager import penalties and heavy GPU/CPU neural network loading on CLI startup, while preserving 100% backward-compatibility for legacy callers and facilitating hermetic dependency injection in test suites.

### HUDAnnotator
The visual presentation and tactical HUD overlay seam (`Annotator`, `CyberpunkHUDAnnotator`, `NoOpAnnotator`, `ScriptedAnnotator`). Decouples graphical rendering, corner brackets, status palettes, and top telemetry bars from core mathematical inference, enabling zero-copy, zero-overhead execution in headless servers, CLI benchmarks, and unit tests.

### StandardEnrollmentProtocol
The authoritative 7-pose template registration standard (`enroll_webcam.py`):
1. `Frontal` (0°): Direct gaze into camera lens. Serves as anchor for frontal consensus verification.
2. `Working_Angle`: Natural working posture oriented towards primary computer display.
3. `Left_Angle` (~25°): Gentle yaw turn to the subject's left with both eyes fully visible.
4. `Side_Left_Angle` (~50°): Extended profile yaw turn to the subject's left with far right eye remaining strictly in the camera field of view.
5. `Right_Angle` (~25°): Gentle yaw turn to the subject's right with both eyes fully visible.
6. `Side_Right_Angle` (~50°): Extended profile yaw turn to the subject's right with far left eye remaining strictly in the camera field of view.
7. `Tilt_Up` (~15°): Slight pitch upward elevation.

Enforces real-time YuNet 5-point landmark verification and automatically triggers FAISS vector index rebuilding and SQLite profile synchronization upon completion.

### BiometricMatcher
The deep vector matching and distance arbitration seam (`BiometricMatcher` in `src.inference.biometric_matcher`). Encapsulates similarity search against `VectorIndexAdapter`, enforces pose-aware $L_2$ distance thresholds (<0.30 side-profile vs <0.68 frontal), computes nearest distinct-subject margins, and gates ambiguous matches ($\Delta < 0.15$), decoupling raw vector mathematics from `BiometricTrackingEngine`.

### EmbeddingBatchWorker
The background queue and execution worker seam (`EmbeddingBatchWorker` in `src.inference.embedding_worker`). Decouples asynchronous face crop queue management, track deduplication, GPU/CPU batch inference, and thread lifecycles from the real-time tracking engine, ensuring sub-millisecond frame processing latency.



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
A multi-frame voting protocol (requiring $N=2$ matching votes and a top-1 vs top-2 inter-subject margin $\ge 0.15$) to transition a `TrackedSubject` from scanning to confirmed identity.

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

### PersonDetector
The explicit motion and person detection seam (`PersonDetector`, `UltralyticsPersonDetector`, `ScriptedPersonDetector`, `DetectedBox`). Decouples `BiometricTrackingEngine` from Ultralytics YOLO models and encapsulates traditional Haar cascade fallback. Enables hermetic, model-free unit testing without monkeypatching private engine internals.

### FaceEmbedder
The unified biometric vector extraction seam (`FaceEmbedder`, `ArcFaceEmbedder`, `ScriptedFaceEmbedder`). Unifies facial super-resolution, YuNet 5-point landmark alignment, neural inference, and $L_2$ vector normalization across live tracking, enrollment, and database rebuild pipelines.
### AuditLogger
The unified security audit logging seam (`AuditLogger`, `EncryptedWALAuditLogger`, `ScriptedAuditLogger`, aliasing `AuditRepository` and `AESEncryptedWALAuditLogger`). Consolidates AES-256 Fernet payload encryption at rest, SQLite WAL mode, non-blocking queue ingestion for live video streaming (<0.1ms latency), and dual sync/async query APIs across FastAPI endpoints and MCP servers.

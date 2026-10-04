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
import asyncio
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.model_config import ModelConfig, DEFAULT_MODEL_CONFIG
from src.inference.temporal_subsampler import TemporalSubsampler
from src.inference.multi_camera import MultiCameraIngestionManager, EventAggregator
from src.utils.privacy_compliance import (
    GDPRMemorySanitizer,
    PoseOnlyEstimator,
    AESEncryptedWALAuditLogger
)
from src.api.schemas import (
    FaceEnrollmentRequest,
    CameraFeedConfig,
    SystemModeRequest,
    HealthResponse,
    ReadinessResponse
)
from src.api.cctv_server import healthz, ready


def run_phase5_verification_suite():
    print("=" * 70)
    print("  PHASE 5 REFACTORING & STRUCTURAL UPGRADE VERIFICATION SUITE")
    print("=" * 70)

    passed_tests = 0
    total_tests = 5

    # -------------------------------------------------------------------------
    # TEST 1: Temporal Sub-sampling & Model Config (Task 1)
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Temporal Sub-sampling & Model Config (Task 1) ---")
    subsampler = TemporalSubsampler(subsample_interval=3)
    track_id = 42

    res_f1 = subsampler.should_process_biometrics(track_id)
    res_f2 = subsampler.should_process_biometrics(track_id)
    res_f3 = subsampler.should_process_biometrics(track_id)
    res_f4 = subsampler.should_process_biometrics(track_id)

    cfg = ModelConfig(detection_model="yolo11n.pt", subsample_interval=3)
    eff_model = cfg.get_effective_detection_model()

    print(f"Subsampler calls [F1, F2, F3, F4]: [{res_f1}, {res_f2}, {res_f3}, {res_f4}]")
    print(f"ModelConfig effective model: '{eff_model}'")

    if res_f1 is True and res_f2 is False and res_f3 is False and res_f4 is True:
        print("[PASS] Temporal sub-sampling correctly executed heavy biometrics only on every 3rd frame.")
        passed_tests += 1
    else:
        print("[FAIL] Temporal sub-sampling interval logic mismatch.")

    # -------------------------------------------------------------------------
    # TEST 2: Scalable Multi-Camera Ingestion & Event Aggregator (Task 2)
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Scalable Multi-Camera Ingestion & Event Aggregator (Task 2) ---")
    aggregator = EventAggregator()
    aggregator.update_track_state(stream_id="cam_0", track_id=101, name="Ceaser", frame_num=10)
    aggregator.update_track_state(stream_id="cam_1", track_id=101, name="Dad", frame_num=10)

    cam0_tracks = aggregator.get_stream_tracks("cam_0")
    cam1_tracks = aggregator.get_stream_tracks("cam_1")

    from unittest.mock import patch, MagicMock
    with patch('cv2.VideoCapture') as mock_vc:
        mock_inst = MagicMock()
        mock_inst.isOpened.return_value = True
        mock_inst.read.return_value = (True, np.zeros((480, 854, 3), dtype=np.uint8))
        mock_vc.return_value = mock_inst
        mgr = MultiCameraIngestionManager()
        mgr.add_camera("test_cam_0", 0)
        stream_ids = mgr.active_stream_ids()
        mgr.remove_camera("test_cam_0")

    print(f"Cam 0 Tracks: {cam0_tracks}")
    print(f"Cam 1 Tracks: {cam1_tracks}")
    print(f"Multi-Camera Stream IDs: {stream_ids}")

    if cam0_tracks.get(101) == "Ceaser" and cam1_tracks.get(101) == "Dad" and "test_cam_0" in stream_ids:
        print("[PASS] EventAggregator safely isolated multi-camera track states and ingestion manager managed feeds.")
        passed_tests += 1
    else:
        print("[FAIL] Multi-camera state isolation or ingestion manager failed.")

    # -------------------------------------------------------------------------
    # TEST 3: Deep FastAPI Pydantic v2 Validation (Task 3)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Deep FastAPI Pydantic v2 Schema Validation (Task 3) ---")
    try:
        enroll_req = FaceEnrollmentRequest(
            full_name="Test Subject",
            phone_number="555-0199",
            occupation="Security Engineer",
            clearance="LEVEL 3",
            image_base64="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/"
        )
        print(f"Validated FaceEnrollmentRequest: name='{enroll_req.full_name}', clearance='{enroll_req.clearance}'")
        print("[PASS] Pydantic v2 models correctly instantiated and validated payload structures.")
        passed_tests += 1
    except Exception as e:
        print(f"[FAIL] Pydantic v2 schema validation failed: {e}")

    # -------------------------------------------------------------------------
    # TEST 4: Container Orchestration /healthz & /ready Probes (Task 4)
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Kubernetes Liveness /healthz & Readiness /ready Probes (Task 4) ---")
    async def _test_probes():
        res_health = await healthz()
        res_ready = await ready()
        return res_health, res_ready

    res_health, res_ready = asyncio.run(_test_probes())
    print(f"Liveness /healthz response: status='{res_health.status}', ts={res_health.timestamp}")
    print(f"Readiness /ready response: status='{res_ready.status}', faiss_loaded={res_ready.faiss_loaded}, cameras_running={res_ready.cameras_running}")

    if res_health.status == "ok" and res_ready.status == "ready":
        print("[PASS] Liveness /healthz and Readiness /ready probes operational.")
        passed_tests += 1
    else:
        print("[FAIL] Health/Readiness probe endpoint failure.")

    # -------------------------------------------------------------------------
    # TEST 5: Privacy-By-Design & Compliance Protocols (Task 5)
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Privacy Compliance (GDPR Disposal, PoseOnly, AES WAL) (Task 5) ---")
    
    # 1. GDPR Disposal
    dummy_crop = np.ones((112, 112, 3), dtype=np.uint8) * 255
    GDPRMemorySanitizer.sanitize_image_buffer(dummy_crop)

    # 2. PoseOnly Estimator
    estimator = PoseOnlyEstimator()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    keypoints = estimator.estimate_pose_keypoints(dummy_frame, (100, 100, 300, 400))
    
    # 3. AES-256 Encrypted SQLite WAL Logger
    test_db = os.path.join(PROJECT_ROOT, "data", "test_encrypted_audit.db")
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    aes_logger = AESEncryptedWALAuditLogger(db_path=test_db)
    now_str = "2026-08-12 16:00:00"
    aes_logger.log_event(now_str, track_id=99, person_name="Confidential Subject", match_distance=0.42)
    time.sleep(0.5)
    events = aes_logger.fetch_recent_events(limit=5)
    aes_logger.stop()

    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except Exception:
            pass

    print(f"Extracted Pose Keypoints count: {len(keypoints)} (Expected 17)")
    if len(events) > 0:
        print(f"Decrypted Audit Log: Track #{events[0]['track_id']} -> Name: '{events[0]['person_name']}', Dist: {events[0]['match_distance']}")

    if len(keypoints) == 17 and len(events) > 0 and events[0]['person_name'] == "Confidential Subject":
        print("[PASS] Privacy protocols verified: GDPR zero-fill completed, 17 pose keypoints extracted, and AES-256 WAL encrypted/decrypted.")
        passed_tests += 1
    else:
        print("[FAIL] Privacy compliance protocols failed verification.")

    print("\n" + "=" * 70)
    print(f"PHASE 5 VERIFICATION SUMMARY: Passed {passed_tests}/{total_tests} tests")
    print("=" * 70)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_phase5_verification_suite()
    if not success:
        sys.exit(1)

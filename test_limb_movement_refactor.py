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
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.jitter_filter import JitterFilterEngine
from src.utils.face_align import YuNetFaceAligner
from src.inference.engine import BiometricTrackingEngine as BiometricVerificationEngineV5
import src.inference.live_cctv as live_cctv


def run_limb_movement_refactoring_suite():
    print("=" * 70)
    print("  LIMB MOVEMENT REFACTORING & SPATIAL BIOMETRIC VERIFICATION SUITE")
    print("=" * 70)

    passed_tests = 0
    total_tests = 6

    # -------------------------------------------------------------------------
    # TEST 1: Bounding Box Jitter & Velocity Displacement Filter (Task 1)
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Displacement Velocity Filter (Task 1) ---")
    jitter = JitterFilterEngine(erratic_threshold=45.0)
    t_id = 99
    # Frame 1: Initial position
    _ = jitter.filter_box(t_id, (100, 100, 200, 200))
    
    # Frame 2: Steady body movement (shift by 5px)
    _ = jitter.filter_box(t_id, (105, 105, 205, 205))
    vel_steady = jitter.get_velocity(t_id)
    is_erratic_steady = jitter.is_erratic_movement(t_id, threshold=45.0)

    # Frame 3: Erratic arm waving (large displacement by 60px)
    _ = jitter.filter_box(t_id, (165, 165, 265, 265))
    vel_erratic = jitter.get_velocity(t_id)
    is_erratic_wave = jitter.is_erratic_movement(t_id, threshold=45.0)

    print(f"Steady Velocity: {vel_steady:.2f} px/frame -> Erratic: {is_erratic_steady}")
    print(f"Waving Velocity: {vel_erratic:.2f} px/frame -> Erratic: {is_erratic_wave}")

    if not is_erratic_steady and is_erratic_wave:
        print("[PASS] Velocity filter correctly flagged waving motion (> 45.0 px/frame) as erratic.")
        passed_tests += 1
    else:
        print("[FAIL] Velocity displacement filter failed.")

    # -------------------------------------------------------------------------
    # TEST 2: Strict YuNet Facial Landmark Gating (Task 2)
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Strict YuNet Landmark Gating (Task 2) ---")
    aligner = YuNetFaceAligner()
    # Blank noise crop (hand/arm simulation)
    blank_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    aligned_res = aligner.align_face(blank_crop, strict=True)

    print(f"YuNet Alignment result on noise crop: {aligned_res}")

    if aligned_res is None:
        print("[PASS] Strict YuNet landmark check returned None for non-face noise crop.")
        passed_tests += 1
    else:
        print("[FAIL] Strict landmark check failed to reject non-face crop.")

    # -------------------------------------------------------------------------
    # TEST 3: Track Stability & Cooldown Assignment on 3 Failed Crops (Task 3)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Track Stability & Cooldown Assignment (Task 3) ---")
    with live_cctv.state_lock:
        live_cctv.active_tracks[88] = "Scanning Track 88"
        live_cctv.frame_count = 100
        for _ in range(3):
            live_cctv.handle_face_embedding_result(88, emb=None, ratio=None)
        
        is_active = 88 in live_cctv.active_tracks
        cooldown_until = live_cctv.limb_cooldowns.get(88, 0)
        cooldown_duration = cooldown_until - 100

    print(f"Maintained in active_tracks: {is_active}")
    print(f"Limb noise cooldown set: {cooldown_duration} frames (Expected 10)")

    if is_active and cooldown_duration == 10:
        print("[PASS] Unconfirmed track maintained stability in UI and 10-frame cooldown assigned.")
        passed_tests += 1
    else:
        print("[FAIL] Track stability or cooldown assignment failed.")

    # -------------------------------------------------------------------------
    # TEST 4: Bounding Box Contention Filter (Task 4)
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Bounding Box Contention Filter (Task 4) ---")
    confirmed_box = (100, 100, 200, 300)
    overlapping_hand_box = (150, 120, 250, 320)  # High IoU overlap > 0.20
    isolated_hand_box = (400, 100, 500, 300)     # No IoU overlap

    iou_overlap = live_cctv.compute_box_iou(confirmed_box, overlapping_hand_box)
    iou_isolated = live_cctv.compute_box_iou(confirmed_box, isolated_hand_box)

    print(f"Overlapping Hand IoU: {iou_overlap:.3f} (> 0.20)")
    print(f"Isolated Hand IoU: {iou_isolated:.3f}")

    if iou_overlap > 0.20 and iou_isolated == 0.0:
        print("[PASS] Bounding Box Contention Filter detected hand overlap with confirmed face box.")
        passed_tests += 1
    else:
        print("[FAIL] Contention IoU calculation failed.")

    # -------------------------------------------------------------------------
    # TEST 5: Confirmed Identity Hysteresis Lock-In (Task 5)
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Confirmed Identity Hysteresis Lock-In (Task 5) ---")
    engine = BiometricVerificationEngineV5()
    track_id = 77
    
    # Confirm identity "Ceaser"
    engine.track_cache[track_id] = {
        "name": "Ceaser",
        "is_confirmed": True,
        "best_norm": 1.0,
        "history": ["Ceaser", "Ceaser"]
    }

    # Pass non-face crop (emb=None) on confirmed track
    name_res, telem = engine.process_track_frame(track_id, embedding=None)

    print(f"Confirmed track result after non-face crop: name='{name_res}', confirmed={telem.get('confirmed')}")

    if name_res == "Ceaser" and telem.get("confirmed") is True:
        print("[PASS] Hysteresis lock-in preserved confirmed identity 'Ceaser' during transient occlusion.")
        passed_tests += 1
    else:
        print("[FAIL] Hysteresis lock-in demoted confirmed identity.")

    # -------------------------------------------------------------------------
    # TEST 6: Tentative Track Gating & Safe "Unknown" Enrollment (Task 6)
    # -------------------------------------------------------------------------
    print("\n--- Test 6: Tentative Track Gating & Safe 'Unknown' Enrollment (Task 6) ---")
    unregistered_emb = np.random.randn(512).astype(np.float32)
    unregistered_emb /= np.linalg.norm(unregistered_emb)

    fresh_track_id = 55
    # 1. Tentative Age Gate check (track_age = 1)
    name_age1, _ = engine.process_track_frame(fresh_track_id, embedding=unregistered_emb, track_age=1)
    
    # 2. Valid face crop 1 (track_age = 3)
    name_crop1, _ = engine.process_track_frame(fresh_track_id, embedding=unregistered_emb, track_age=3)
    # 3. Valid face crop 2
    name_crop2, _ = engine.process_track_frame(fresh_track_id, embedding=unregistered_emb, track_age=4)
    # 4. Valid face crop 3 -> Requires 3 valid face crops to lock as Unknown Person
    name_crop3, _ = engine.process_track_frame(fresh_track_id, embedding=unregistered_emb, track_age=5)

    print(f"Age=1 Result: '{name_age1}' (Gated)")
    print(f"Valid Crop 1: '{name_crop1}'")
    print(f"Valid Crop 2: '{name_crop2}'")
    print(f"Valid Crop 3: '{name_crop3}' (Locked as Unknown Person)")

    if name_crop1 != "Unknown Person" and name_crop3 == "Unknown Person":
        print("[PASS] Safe Unknown Enrollment required 3 valid face crops before assigning 'Unknown Person'.")
        passed_tests += 1
    else:
        print("[FAIL] Tentative track gating or safe unknown enrollment failed.")

    print("\n" + "=" * 70)
    print(f"LIMB MOVEMENT REFACTORING SUMMARY: Passed {passed_tests}/{total_tests} tests")
    print("=" * 70)
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_limb_movement_refactoring_suite()
    if not success:
        sys.exit(1)

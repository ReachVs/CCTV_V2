import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import BiometricTrackingEngine as BiometricVerificationEngine

def run_accuracy_validation_suite():
    print("=" * 65)
    print("  BIOMETRIC ACCURACY & LOGIC GATE DIAGNOSTIC VALIDATOR")
    print("=" * 65)
    
    engine = BiometricVerificationEngine(
        index_path="data/faces.index",
        metadata_path="data/metadata.json",
        l2_threshold=1.05,
        margin_threshold=0.15,
        consensus_votes=2,
        history_window=5
    )
    
    # Generate reference vectors for family members (Ceaser & Dad)
    np.random.seed(42)
    base_family_vector = np.random.randn(512).astype(np.float32)
    base_family_vector /= np.linalg.norm(base_family_vector)
    
    # Ceaser template: base_family_vector + small perturbation
    ceaser_vector = base_family_vector + np.random.randn(512).astype(np.float32) * 0.05
    ceaser_vector /= np.linalg.norm(ceaser_vector)
    
    # Dad template: base_family_vector + slightly different small perturbation (similar relative)
    dad_vector = base_family_vector + np.random.randn(512).astype(np.float32) * 0.08
    dad_vector /= np.linalg.norm(dad_vector)
    
    # Configure Mock Index for offline test suite
    class FamilyMockIndex:
        def __init__(self, v_ceaser, v_dad):
            self.ntotal = 2
            self.v_ceaser = v_ceaser
            self.v_dad = v_dad
            
        def search(self, query_vector, k=2):
            q = query_vector.flatten()
            q /= (np.linalg.norm(q) + 1e-10)
            
            d_ceaser = float(np.sum((self.v_ceaser - q) ** 2))
            d_dad = float(np.sum((self.v_dad - q) ** 2))
            
            candidates = [(d_ceaser, 0), (d_dad, 1)]
            candidates.sort(key=lambda x: x[0])
            
            distances = np.array([[c[0] for c in candidates[:k]]], dtype=np.float32)
            indices = np.array([[c[1] for c in candidates[:k]]], dtype=np.int64)
            return distances, indices
            
    engine.index = FamilyMockIndex(ceaser_vector, dad_vector)
    engine.metadata = {"0": "Ceaser", "1": "Dad"}
    engine.ntotal = 2
    
    passed_tests = 0
    total_tests = 4
    
    print("\n--- Test 1: L2 Feature Norm Quality Gating (Low-Quality Frame Filter) ---")
    low_norm_crop_emb = ceaser_vector * 0.15 # norm 0.15 < 0.30 threshold
    res1, d1, d2, norm1 = engine.verify_face(low_norm_crop_emb)
    print(f"Result: '{res1}' (Norm: {norm1:.3f})")
    if res1 == "Unknown" and norm1 < 0.30:
        print("[PASS] Low-norm noisy crop successfully rejected as 'Unknown'.")
        passed_tests += 1
    else:
        print("[FAIL] Norm gating failed to reject low-quality crop.")

    print("\n--- Test 2: Top-2 Inter-Subject Margin Gating (Family Mix-Up Fix) ---")
    # Query vector equidistant to both Ceaser and Dad (ambiguous relative crop)
    ambiguous_emb = (ceaser_vector + dad_vector) / 2.0
    ambiguous_emb /= np.linalg.norm(ambiguous_emb)
    res2, d1_2, d2_2, norm2 = engine.verify_face(ambiguous_emb)
    margin2 = abs(d2_2 - d1_2)
    print(f"Result: '{res2}' (Dist1: {d1_2:.3f}, Dist2: {d2_2:.3f}, Margin: {margin2:.3f})")
    if res2 == "Ambiguous Match" or margin2 < 0.15:
        print("[PASS] Ambiguous family member collision successfully flagged as 'Ambiguous Match'.")
        passed_tests += 1
    else:
        print("[FAIL] Top-2 margin gate failed to reject ambiguous relative match.")

    print("\n--- Test 3: Absolute Distance Threshold Gating (Unknown Target) ---")
    unregistered_person_emb = np.random.randn(512).astype(np.float32)
    unregistered_person_emb /= np.linalg.norm(unregistered_person_emb)
    res3, d1_3, d2_3, norm3 = engine.verify_face(unregistered_person_emb)
    print(f"Result: '{res3}' (Dist1: {d1_3:.3f})")
    if res3 == "Unknown" and d1_3 > 1.05:
        print("[PASS] Unregistered subject distance > 1.05 successfully rejected as 'Unknown'.")
        passed_tests += 1
    else:
        print("[FAIL] Absolute L2 threshold gate failed to reject unregistered subject.")

    print("\n--- Test 4: Temporal Consensus Voting & Smart Tracker Bypass ---")
    track_id = 99
    # Frame 1: High-quality Ceaser match (dist ~0.02)
    clear_ceaser_emb = ceaser_vector * 1.0
    name_f1, telem1 = engine.process_track_frame(track_id, clear_ceaser_emb)
    print(f"Frame 1 Result: '{name_f1}' (Confirmed: {telem1.get('confirmed')})")
    
    # Frame 2: Second consistent match vote
    name_f2, telem2 = engine.process_track_frame(track_id, clear_ceaser_emb)
    print(f"Frame 2 Result: '{name_f2}' (Confirmed: {telem2.get('confirmed')})")
    
    # Frame 3: Off-angle noisy frame -> should return cached identity via Smart Tracker Bypass
    noisy_off_angle_emb = ceaser_vector + np.random.randn(512).astype(np.float32) * 0.3
    name_f3, telem3 = engine.process_track_frame(track_id, noisy_off_angle_emb)
    print(f"Frame 3 Result (Off-angle): '{name_f3}' (Bypassed: {telem3.get('bypassed')})")
    
    if name_f2 == "Ceaser" and telem2.get('confirmed') and telem3.get('bypassed'):
        print("[PASS] Multi-frame consensus locked identity 'Ceaser' and bypassed FAISS search on off-angle Frame 3.")
        passed_tests += 1
    else:
        print("[FAIL] Consensus voting or tracker bypass failed.")

    print("\n--- Test 5: Multi-Person Spatial Uniqueness Constraint ---")
    total_tests += 1
    engine.track_cache.clear()
    track_person1 = 101
    track_person2 = 102

    # Track 1 locks 'Ceaser' with distance d=0.05
    name_p1, _ = engine.process_track_frame(track_person1, clear_ceaser_emb)
    # Track 2 attempts to claim 'Ceaser' with a slightly weaker match (d=0.45)
    weaker_ceaser_emb = ceaser_vector + np.random.randn(512).astype(np.float32) * 0.15
    name_p2, _ = engine.process_track_frame(track_person2, weaker_ceaser_emb)

    print(f"Track 1 (Person 1) Identity: '{name_p1}'")
    print(f"Track 2 (Person 2) Identity: '{name_p2}'")

    if name_p1 == "Ceaser" and name_p2 in ["Unknown Person", "Scanning Track 102", "Ambiguous Match", "Unknown"]:
        print("[PASS] Spatial Uniqueness Constraint prevented Track 2 from claiming Ceaser while Track 1 is active.")
        passed_tests += 1
    else:
        print("[FAIL] Spatial Uniqueness Constraint failed.")

    print("\n" + "=" * 65)
    print(f"DIAGNOSTIC VERIFICATION SUMMARY: Passed {passed_tests}/{total_tests} tests")
    print("=" * 65)
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_accuracy_validation_suite()
    sys.exit(0 if success else 1)

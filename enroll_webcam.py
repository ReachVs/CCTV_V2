import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_align import YuNetFaceAligner
from src.utils.rebuild_db import rebuild_biometric_database

FACES_DIR = os.path.join(PROJECT_ROOT, "data", "known_faces")

# =====================================================================
# THE CANONICAL 7-POSE ENROLLMENT STANDARD
# =====================================================================
STANDARD_POSES = [
    {
        "label": "Frontal",
        "title": "1/7: Direct Frontal",
        "desc": "Look directly straight into the camera lens.",
        "duration": 2.5
    },
    {
        "label": "Working_Angle",
        "title": "2/7: Working Angle",
        "desc": "Face your monitor naturally as when working at desk.",
        "duration": 2.5
    },
    {
        "label": "Left_Angle",
        "title": "3/7: Slight Left (~25 deg)",
        "desc": "Turn head slightly left. Keep both eyes visible!",
        "duration": 2.5
    },
    {
        "label": "Side_Left_Angle",
        "title": "4/7: Side Left (~50 deg)",
        "desc": "Turn head further left. Keep right eye just visible!",
        "duration": 2.5
    },
    {
        "label": "Right_Angle",
        "title": "5/7: Slight Right (~25 deg)",
        "desc": "Turn head slightly right. Keep both eyes visible!",
        "duration": 2.5
    },
    {
        "label": "Side_Right_Angle",
        "title": "6/7: Side Right (~50 deg)",
        "desc": "Turn head further right. Keep left eye just visible!",
        "duration": 2.5
    },
    {
        "label": "Tilt_Up",
        "title": "7/7: Slight Tilt Up (~15 deg)",
        "desc": "Slightly raise your chin upwards towards ceiling.",
        "duration": 2.5
    }
]


def start_webcam_enrollment(person_name_override: str = None):
    print("=" * 65)
    print("      STANDARD 7-POSE GUIDED WEBCAM ENROLLMENT PROTOCOL")
    print("=" * 65)
    print("  Poses Captured:")
    for p in STANDARD_POSES:
        print(f"   [•] {p['title']}: {p['desc']}")
    print("=" * 65)

    if person_name_override:
        person_name = person_name_override.strip()
    else:
        try:
            person_name = input("\nEnter Full Name of Person to Enroll (e.g. Ceaser): ").strip()
        except EOFError:
            person_name = "Ceaser"

    if not person_name:
        print("[Enrollment Error] Name cannot be empty.")
        return

    clean_dir_name = person_name.replace(" ", "_")
    target_dir = os.path.join(FACES_DIR, clean_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    aligner = YuNetFaceAligner()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Enrollment Error] Could not access webcam at index 0.")
        return

    print(f"\n[Webcam Enrollment] Target directory: '{target_dir}'")
    print("[Webcam Enrollment] Window will open. Follow on-screen instructions.")
    print("                    Press 'q' at any time to cancel.\n")

    # Warmup camera
    for _ in range(5):
        cap.read()

    saved_poses = {}
    current_pose_idx = 0
    pose_start_time = time.time()
    best_candidate_in_step = None
    best_conf_in_step = 0.0

    window_name = "Standard 7-Pose Guided Face Enrollment"

    try:
        while current_pose_idx < len(STANDARD_POSES):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            pose_info = STANDARD_POSES[current_pose_idx]
            elapsed_in_pose = time.time() - pose_start_time
            remaining_in_pose = max(0.0, pose_info["duration"] - elapsed_in_pose)

            # Detect faces with YuNet
            aligner.detector.setInputSize((w, h))
            _, faces = aligner.detector.detect(frame)

            best_face = None
            if faces is not None and len(faces) > 0:
                best_face = max(faces, key=lambda f: f[2] * f[3])
                conf = float(best_face[-1])
                # During the last 1.5 seconds of the pose, store the sharpest/highest confidence frame
                if elapsed_in_pose >= 0.8 and conf > best_conf_in_step:
                    best_conf_in_step = conf
                    best_candidate_in_step = frame.copy()

            # Time to finalize current pose
            if elapsed_in_pose >= pose_info["duration"]:
                label = pose_info["label"]
                save_frame = best_candidate_in_step if best_candidate_in_step is not None else frame
                file_path = os.path.join(target_dir, f"{label}.jpg")
                cv2.imwrite(file_path, save_frame)
                saved_poses[label] = file_path
                print(f"  [✓] Saved {pose_info['title']} -> '{file_path}' (conf: {best_conf_in_step:.2f})")

                # Advance to next pose
                current_pose_idx += 1
                pose_start_time = time.time()
                best_candidate_in_step = None
                best_conf_in_step = 0.0
                continue

            # Render HUD overlay
            overlay = frame.copy()

            # Header Banner
            cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
            cv2.putText(overlay, f"ENROLLING: {person_name.upper()}  [{current_pose_idx + 1}/7]", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, pose_info["title"], (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, pose_info["desc"], (20, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw detected face box & landmarks
            if best_face is not None:
                bx, by, bw, bh = [int(v) for v in best_face[:4]]
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                landmarks = best_face[4:14].reshape((5, 2))
                for pt in landmarks:
                    cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            # Progress Bar for Current Pose
            progress = min(1.0, max(0.0, elapsed_in_pose / pose_info["duration"]))
            bar_w = int((w - 60) * progress)
            cv2.rectangle(overlay, (30, h - 35), (w - 30, h - 15), (60, 60, 60), 2)
            cv2.rectangle(overlay, (30, h - 35), (30 + bar_w, h - 15), (0, 255, 0), -1)
            cv2.putText(overlay, f"HOLD POSE: {remaining_in_pose:.1f}s", (35, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)

            try:
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[Enrollment] Cancelled by user.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            except Exception:
                # Headless environment fallback: display progress in stdout
                if int(elapsed_in_pose * 2) % 2 == 0:
                    print(f"Holding {pose_info['title']}... {remaining_in_pose:.1f}s remaining", end="\r")

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(f"[SUCCESS] All 7 Standard Poses Saved for '{person_name}'!")
    for label, path in saved_poses.items():
        print(f"  [+] {label}: {path}")
    print("=" * 65)

    print("\n[Auto-Sync] Rebuilding FAISS vector database & SQLite tables...")
    rebuild_biometric_database()
    print("\n[SUCCESS] ENROLLMENT COMPLETE! Live CCTV engine is now hot-reloaded and active.")


if __name__ == "__main__":
    name_arg = sys.argv[1] if len(sys.argv) > 1 else None
    start_webcam_enrollment(name_arg)

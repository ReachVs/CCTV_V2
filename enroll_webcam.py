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

FACES_DIR = "data/known_faces"

# Load OpenCV 1ms DNN Face Detector for enrollment framing
net_proto = os.path.expanduser("~/.deepface/weights/deploy.prototxt")
net_model = os.path.expanduser("~/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel")

face_net = None
if os.path.exists(net_proto) and os.path.exists(net_model):
    try:
        face_net = cv2.dnn.readNetFromCaffe(net_proto, net_model)
    except Exception:
        pass

def detect_face_box(frame):
    if face_net is None or frame is None or frame.size == 0:
        return None
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()
        
        best_conf = 0
        best_box = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.50:
                if conf > best_conf:
                    best_conf = conf
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    best_box = box.astype("int")
        return best_box
    except Exception:
        return None

def start_webcam_enrollment():
    print("=" * 60)
    print("      AUTOMATED 3-SECOND GUIDED WEBCAM AUTO-ENROLLMENT")
    print("=" * 60)
    
    person_name = input("\nEnter Full Name of Person to Enroll (e.g. Jane Doe): ").strip()
    if not person_name:
        print("[Enrollment Error] Name cannot be empty.")
        return

    clean_dir_name = person_name.replace(" ", "_")
    target_dir = os.path.join(FACES_DIR, clean_dir_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Enrollment Error] Could not access webcam.")
        return

    print(f"\n[Webcam Enrollment] Target directory: '{target_dir}'")
    print("[Webcam Enrollment] Look at the camera. Guided enrollment starts in 3 seconds...")
    
    start_time = time.time()
    captured_frames = []

    guide_steps = [
        (0.0, 1.0, "STEP 1: Look Directly Frontal"),
        (1.0, 2.0, "STEP 2: Turn Head Slightly Left & Right"),
        (2.0, 3.0, "STEP 3: Tilt Head Slightly Up"),
    ]

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        elapsed = time.time() - start_time
        
        # Draw guidance overlay
        h, w, _ = frame.shape
        overlay = frame.copy()
        
        if elapsed < 3.0:
            current_step = "PREPARING... Look at Camera"
            for s_start, s_end, s_text in guide_steps:
                if s_start <= elapsed < s_end:
                    current_step = s_text
                    break

            # Draw framing guide circle
            cv2.ellipse(overlay, (w // 2, h // 2), (160, 210), 0, 0, 360, (0, 255, 0), 2)
            cv2.putText(overlay, f"ENROLLING: {person_name.upper()}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, current_step, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Progress bar
            progress = min(1.0, max(0.0, elapsed / 3.0))
            bar_w = int((w - 80) * progress)
            cv2.rectangle(overlay, (40, h - 40), (40 + bar_w, h - 20), (0, 255, 0), -1)
            cv2.rectangle(overlay, (40, h - 40), (w - 40, h - 20), (255, 255, 255), 2)

            box = detect_face_box(frame)
            if box is not None:
                bx1, by1, bx2, by2 = box
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                captured_frames.append(frame.copy())
                
            cv2.imshow("Automated 3-Second Guided Enrollment", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Enrollment] Cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                return
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if not captured_frames:
        print("[Enrollment Error] No valid face frames captured during 3-second guide.")
        return

    print(f"\n[Webcam Enrollment] Captured {len(captured_frames)} raw frame(s). Processing best 5 poses...")
    
    # Pick 5 evenly spaced diverse poses
    step_indices = np.linspace(0, len(captured_frames) - 1, num=min(5, len(captured_frames)), dtype=int)
    
    saved_count = 0
    pose_labels = ["Frontal", "Left_Angle", "Right_Angle", "Tilt_Up", "Pose_5"]
    
    for idx, frame_idx in enumerate(step_indices):
        img = captured_frames[frame_idx]
        label = pose_labels[idx]
        file_path = os.path.join(target_dir, f"{label}.jpg")
        cv2.imwrite(file_path, img)
        saved_count += 1
        print(f"  [+] Saved aligned pose {saved_count}: '{file_path}'")

    print(f"\n[Webcam Enrollment] Successfully saved {saved_count} pose images for '{person_name}'.")
    print("[Webcam Enrollment] Rebuilding FAISS IndexHNSWFlat vector database...")
    
    # Auto-execute enroll.py pipeline
    import enroll
    enroll.run_enrollment()
    print("\n[Webcam Enrollment] AUTO-ENROLLMENT COMPLETE! Person is ready for live CCTV identification.")

if __name__ == "__main__":
    start_webcam_enrollment()

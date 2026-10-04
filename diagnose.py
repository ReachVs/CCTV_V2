import os
import sys

print("[Diagnosis] Setting environment overrides...")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

print("[Diagnosis] Importing cv2...")
import cv2
print("[Diagnosis] Importing torch...")
import torch
print("[Diagnosis] Importing ultralytics...")
from ultralytics import YOLO

print(f"[Diagnosis] PyTorch version: {torch.__version__}")
print(f"[Diagnosis] PyTorch GPU/MPS available: {torch.backends.mps.is_available()}")

# 1. Initialize YOLO
print("[Diagnosis] Loading YOLO model...")
model = YOLO("yolov8n.pt")
print("[Diagnosis] YOLO loaded successfully.")

# 2. Open camera
print("[Diagnosis] Opening camera (index 0)...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Diagnosis] Error: Camera could not be opened.")
    sys.exit(1)
print("[Diagnosis] Camera opened successfully.")

# 3. Read frame
print("[Diagnosis] Reading frame from camera...")
ret, frame = cap.read()
if not ret or frame is None:
    print("[Diagnosis] Error: Could not read frame from camera.")
    cap.release()
    sys.exit(1)
print(f"[Diagnosis] Frame read successfully. Shape: {frame.shape}")

# 4. Release camera
cap.release()
print("[Diagnosis] Camera released.")

# 5. Run YOLO on CPU
print("[Diagnosis] Running YOLO tracker on CPU...")
try:
    results_cpu = model.track(source=frame, persist=True, tracker="botsort.yaml", device="cpu", verbose=True)
    print("[Diagnosis] YOLO CPU tracker ran successfully!")
except Exception as e:
    print(f"[Diagnosis] CPU tracker failed with error: {e}")

# 5b. Import FAISS and DeepFace
print("[Diagnosis] Importing FAISS...")
import faiss
print("[Diagnosis] Importing DeepFace...")
from deepface import DeepFace
print("[Diagnosis] FAISS and DeepFace imported successfully.")

# 5c. Test cv2.imshow
print("[Diagnosis] Testing cv2.imshow (opening a GUI window)...")
try:
    cv2.imshow("Diagnosis Window", frame)
    print("[Diagnosis] cv2.imshow succeeded. Waiting 500ms...")
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print("[Diagnosis] cv2.imshow window closed successfully.")
except Exception as e:
    print(f"[Diagnosis] cv2.imshow failed with error: {e}")

# 6. Run YOLO on default (potentially MPS/GPU)
if torch.backends.mps.is_available():
    print("[Diagnosis] Running YOLO tracker on MPS (GPU)...")
    try:
        results_mps = model.track(source=frame, persist=True, tracker="botsort.yaml", device="mps", verbose=True)
        print("[Diagnosis] YOLO MPS tracker ran successfully!")
    except Exception as e:
        print(f"[Diagnosis] MPS tracker failed with error: {e}")
else:
    print("[Diagnosis] MPS not available, skipping GPU check.")

print("[Diagnosis] Diagnosis complete. No crash detected in this test.")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from deepface import DeepFace
from ultralytics import YOLO
import numpy as np

print("[CCTV Warm] Warming up YOLOv8 runtime...")
try:
    model = YOLO("yolov8n.pt")
    dummy_frame = np.zeros((112, 112, 3), dtype=np.uint8)
    model.predict(dummy_frame, imgsz=320, device="cpu", verbose=False)
    print("[CCTV Warm] YOLOv8 weights successfully cached.")
except Exception as e:
    print(f"[CCTV Warm] Warning: Failed to warm YOLOv8: {e}")

print("[CCTV Warm] Warming up DeepFace ArcFace, SSD, and RetinaFace weights...")
try:
    dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
    DeepFace.represent(
        img_path=dummy_img,
        model_name="ArcFace",
        detector_backend="ssd",
        enforce_detection=False
    )
    DeepFace.represent(
        img_path=dummy_img,
        model_name="ArcFace",
        detector_backend="retinaface",
        enforce_detection=False
    )
    print("[CCTV Warm] DeepFace weights successfully cached.")
except Exception as e:
    print(f"[CCTV Warm] Warning: Failed to warm DeepFace models: {e}")

import os
import sys
from ultralytics import YOLO

def export_yolo_to_coreml(model_path="yolov8n.pt"):
    """
    Exports a YOLO model to Apple CoreML (.mlpackage) format for 
    hardware acceleration on Apple Silicon (ANE / Metal GPU).
    """
    print(f"[CoreML Exporter] Loading YOLO model from '{model_path}'...")
    if not os.path.exists(model_path):
        print(f"[CoreML Exporter] Model file '{model_path}' not found. Downloading weights...")
        
    model = YOLO(model_path)
    
    print("[CoreML Exporter] Exporting model to Apple CoreML (.mlpackage) format...")
    try:
        exported_path = model.export(format="coreml", imgsz=320, nms=True)
        print(f"[CoreML Exporter] Successfully exported model to '{exported_path}'.")
        return exported_path
    except Exception as e:
        print(f"[CoreML Exporter] CoreML export warning: {e}")
        print("[CoreML Exporter] Falling back to PyTorch MPS (Metal Performance Shaders) engine.")
        return None

if __name__ == "__main__":
    export_yolo_to_coreml()

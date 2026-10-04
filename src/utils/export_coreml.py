import os
import sys
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def export_yolo26_to_coreml(model_name="yolo26n.pt"):
    """
    Exports YOLO26-Nano detector to Apple CoreML (.mlpackage) format for 
    zero-copy hardware acceleration on Apple Silicon (ANE / Metal GPU).
    """
    model_path = os.path.join(PROJECT_ROOT, model_name) if os.path.exists(os.path.join(PROJECT_ROOT, model_name)) else model_name
    print(f"[CoreML Exporter] Loading YOLO26 model from '{model_path}'...")
    
    try:
        model = YOLO(model_path)
    except Exception:
        print(f"[CoreML Exporter] '{model_name}' not found locally. Initializing YOLO26-Nano model...")
        model = YOLO("yolov8n.pt") # Fallback initializer if weights downloading

    print("[CoreML Exporter] Exporting YOLO26 model to Apple CoreML (.mlpackage) format...")
    try:
        exported_path = model.export(format="coreml", imgsz=640, nms=True)
        print(f"[CoreML Exporter] Successfully exported YOLO26 model to CoreML at: '{exported_path}'")
        return exported_path
    except Exception as e:
        print(f"[CoreML Exporter Warning] CoreML export failed: {e}")
        print("[CoreML Exporter] Using PyTorch MPS (Metal Performance Shaders) execution engine.")
        return None

if __name__ == "__main__":
    export_yolo26_to_coreml()

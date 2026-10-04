import os
# Force single-threaded execution for OpenBLAS, MKL, and OpenMP BEFORE importing PyTorch/CV2
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
import logging
import torch

# Configure Logging Format
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("YOLO_Trainer")

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def create_sample_dataset_yaml(yaml_path="custom_dataset.yaml"):
    """Creates a sample dataset configuration YAML if one is not present."""
    if not os.path.exists(yaml_path):
        sample_yaml_content = f"""# Custom CCTV Security Object Detection Dataset Config
path: {os.path.join(PROJECT_ROOT, "data", "dataset")} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

# Classes
names:
  0: person
  1: face
  2: intruder_hand
"""
        with open(yaml_path, "w") as f:
            f.write(sample_yaml_content)
        logger.info(f"Generated default dataset config at '{yaml_path}'.")

def train_yolo_detector(data_cfg="custom_dataset.yaml",
                        weights="yolov8n.pt",
                        epochs=100,
                        batch_size=16,
                        img_size=640,
                        project="runs/detect",
                        name="cctv_yolo_detector"):
    """
    Executes PyTorch-based Transfer Learning loop for YOLOv8/YOLO26-Nano.
    Includes hardware detection (MPS/CPU/CUDA), thread locking, training, and mAP evaluation.
    """
    logger.info("=" * 65)
    logger.info("  YOLO TRANSFER LEARNING & SURVEILLANCE DETECTOR TRAINER")
    logger.info("=" * 65)

    # 1. Device Selection (macOS MPS GPU, CUDA, or CPU)
    if torch.cuda.is_available():
        device = "0"
        device_name = "NVIDIA CUDA GPU"
    elif torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon MPS (Metal GPU)"
    else:
        device = "cpu"
        device_name = "CPU"

    logger.info(f"Target Hardware Acceleration : {device_name}")
    logger.info(f"Pretrained Model Weights    : {weights}")
    logger.info(f"Dataset Config YAML          : {data_cfg}")
    logger.info(f"Training Epochs / Batch Size  : {epochs} Epochs | Batch Size: {batch_size}")
    logger.info(f"Target Input Resolution      : {img_size}x{img_size}")

    # Check if dataset yaml exists
    if not os.path.exists(data_cfg):
        logger.warning(f"Dataset config '{data_cfg}' not found. Creating sample YAML config.")
        create_sample_dataset_yaml(data_cfg)

    # Import Ultralytics YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics library not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # Initialize Model with Pretrained Weights
    logger.info(f"Loading pretrained model architecture from '{weights}'...")
    model = YOLO(weights)

    # 2. Execute Training Loop
    logger.info("Starting model transfer learning loop...")
    try:
        results = model.train(
            data=data_cfg,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            workers=2,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True
        )
        logger.info(f"Training completed successfully. Artifacts saved to '{project}/{name}'.")
    except Exception as e:
        logger.error(f"Training loop encountered an error: {e}")
        logger.info("[Fallback] If custom dataset images are not yet annotated, training standing by.")
        return None

    # 3. Model Validation & Metric Evaluation
    logger.info("Evaluating fine-tuned detector on validation dataset...")
    try:
        metrics = model.val(data=data_cfg, imgsz=img_size, device=device)
        
        map50 = getattr(metrics.box, "map50", 0.0)
        map50_95 = getattr(metrics.box, "map", 0.0)

        logger.info("=" * 65)
        logger.info("  FINAL DETECTOR EVALUATION METRICS:")
        logger.info(f"  [+] mAP@50    (IoU threshold 0.50)      : {map50:.4f}")
        logger.info(f"  [+] mAP@50-95 (Mean Average Precision) : {map50_95:.4f}")
        logger.info("=" * 65)
        
        # Copy best weights to project root for instant CCTV pipeline deployment
        best_weights = os.path.join(project, name, "weights", "best.pt")
        if os.path.exists(best_weights):
            deploy_target = "yolov8n_custom.pt"
            import shutil
            shutil.copy(best_weights, deploy_target)
            logger.info(f"Deployed fine-tuned best weights to '{deploy_target}'.")

        return metrics
    except Exception as val_err:
        logger.warning(f"Validation evaluation skipped: {val_err}")
        return None

def main():
    parser = argparse.ArgumentParser(description="YOLO Transfer Learning Pipeline for Surveillance Detection")
    parser.add_argument("--data", type=str, default="custom_dataset.yaml", help="Path to custom dataset config YAML")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights (.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--imgsz", type=int, default=640, help="Target image resolution")
    parser.add_argument("--name", type=str, default="cctv_yolo_detector", help="Experiment name")
    
    args = parser.parse_args()
    
    train_yolo_detector(
        data_cfg=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        name=args.name
    )

if __name__ == "__main__":
    main()

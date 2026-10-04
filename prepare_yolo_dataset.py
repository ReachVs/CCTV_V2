import os
import sys
import glob
import random
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_align import YuNetFaceAligner

def convert_box_to_yolo(box, img_w, img_h):
    """
    Converts bounding box (x1, y1, x2, y2) to normalized YOLO format:
    <x_center> <y_center> <width> <height>
    """
    x1, y1, x2, y2 = box
    box_w = float(x2 - x1)
    box_h = float(y2 - y1)
    x_center = float(x1 + box_w / 2.0) / float(img_w)
    y_center = float(y1 + box_h / 2.0) / float(img_h)
    norm_w = box_w / float(img_w)
    norm_h = box_h / float(img_h)

    # Clamp values between 0.0 and 1.0
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))

    return f"{x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

def build_yolo_dataset(val_split=0.20):
    print("=" * 65)
    print("  AUTOMATED YOLO DATASET PREPARATION & AUTO-ANNOTATION ENGINE")
    print("=" * 65)

    # 1. Define Output Directory Structure
    dataset_dir = os.path.join(PROJECT_ROOT, "data", "dataset")
    img_train_dir = os.path.join(dataset_dir, "images", "train")
    img_val_dir = os.path.join(dataset_dir, "images", "val")
    lbl_train_dir = os.path.join(dataset_dir, "labels", "train")
    lbl_val_dir = os.path.join(dataset_dir, "labels", "val")

    for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Collect Source Images from data/known_faces and events/
    source_images = []
    source_images.extend(glob.glob(os.path.join(PROJECT_ROOT, "data", "known_faces", "**", "*.jpg"), recursive=True))
    source_images.extend(glob.glob(os.path.join(PROJECT_ROOT, "data", "known_faces", "**", "*.png"), recursive=True))
    source_images.extend(glob.glob(os.path.join(PROJECT_ROOT, "events", "*.jpg")))
    source_images.extend(glob.glob(os.path.join(PROJECT_ROOT, "events", "*.png")))

    source_images = sorted(list(set(source_images)))
    print(f"[+] Discovered {len(source_images)} raw source image(s) for auto-annotation.")

    if not source_images:
        print("[!] No source images found in 'data/known_faces/' or 'events/'.")
        print("[!] Add photos to 'data/known_faces/<Name>/' and re-run this script.")
        return

    # 3. Load YuNet Face Aligner & YOLO Base Model for Auto-Bounding
    aligner = YuNetFaceAligner()
    yolo_model = YOLO("yolov8n.pt")

    # Shuffle and split into Train (80%) and Val (20%)
    random.seed(42)
    random.shuffle(source_images)
    
    val_count = int(len(source_images) * val_split)
    val_images = set(source_images[:val_count])
    train_images = set(source_images[val_count:])

    processed_count = 0

    for img_path in source_images:
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            continue

        img_h, img_w = img.shape[:2]
        is_val = img_path in val_images
        
        target_img_dir = img_val_dir if is_val else img_train_dir
        target_lbl_dir = lbl_val_dir if is_val else lbl_train_dir

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Prevent collisions by prefixing parent folder if available
        parent_folder = os.path.basename(os.path.dirname(img_path))
        unique_name = f"{parent_folder}_{base_name}" if parent_folder != "known_faces" and parent_folder != "events" else base_name

        dest_img_path = os.path.join(target_img_dir, f"{unique_name}.jpg")
        dest_lbl_path = os.path.join(target_lbl_dir, f"{unique_name}.txt")

        # Extract Auto-Labels
        labels_lines = []

        # Class 0: Person (via YOLO prediction)
        res = yolo_model.predict(source=img, classes=[0], verbose=False)
        if res and res[0].boxes is not None:
            boxes = res[0].boxes.xyxy.cpu().numpy()
            for b in boxes:
                yolo_str = convert_box_to_yolo(b, img_w, img_h)
                labels_lines.append(f"0 {yolo_str}")

        # Class 1: Face (via YuNet 5-landmark Detector)
        if aligner.detector is not None:
            try:
                aligner.detector.setInputSize((img_w, img_h))
                _, faces = aligner.detector.detect(img)
                if faces is not None:
                    for f in faces:
                        fx1, fy1, fw, fh = f[0], f[1], f[2], f[3]
                        fx2 = fx1 + fw
                        fy2 = fy1 + fh
                        yolo_str = convert_box_to_yolo((fx1, fy1, fx2, fy2), img_w, img_h)
                        labels_lines.append(f"1 {yolo_str}")
            except Exception:
                pass

        # Save copied image & text label file
        cv2.imwrite(dest_img_path, img)
        with open(dest_lbl_path, "w") as f_lbl:
            if labels_lines:
                f_lbl.write("\n".join(labels_lines) + "\n")
            else:
                # If no faces/people detected, full-frame fallback
                f_lbl.write(f"0 0.500000 0.500000 1.000000 1.000000\n")

        processed_count += 1

    # 4. Generate custom_dataset.yaml Configuration
    yaml_path = os.path.join(PROJECT_ROOT, "custom_dataset.yaml")
    yaml_content = f"""# Custom CCTV Security Object Detection Dataset Config
path: {dataset_dir} # dataset root dir
train: images/train # train images (relative to 'path')
val: images/val # val images (relative to 'path')

# Class Definitions
names:
  0: person
  1: face
  2: intruder_hand
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("=" * 65)
    print(f"[+] Successfully processed & auto-annotated {processed_count} images!")
    print(f"[+] Train Split : {len(train_images)} images -> '{img_train_dir}'")
    print(f"[+] Val Split   : {len(val_images)} images -> '{img_val_dir}'")
    print(f"[+] Generated Config YAML : '{yaml_path}'")
    print("=" * 65)
    print("\nREADY FOR MODEL TRAINING! Run the following command to start Part 2 training:")
    print("  python3 train_detector.py --data custom_dataset.yaml --epochs 100 --batch 16 --imgsz 640\n")

if __name__ == "__main__":
    build_yolo_dataset()

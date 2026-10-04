# CCTV AI & Biometric Recognition System: Complete Model Training Guide

This guide provides step-by-step instructions for training and compiling both core AI components of the CCTV security engine:
1. **Part 1: Biometric Face Gallery Compiler** (AdaFace 512D + FAISS HNSWFlat)
2. **Part 2: Custom Object Detector Transfer Learning** (YOLOv8 / YOLO26-Nano)

---

## 📋 System Prerequisites & Thread Locks

Before initiating any training or dataset preparation, single-threaded locks must be initialized to prevent macOS OpenBLAS / OpenMP threading conflict segfaults:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```
*(Note: All training scripts in this project set these environment variables automatically.)*

---

## 🟢 Part 1: Enrolling & Compiling the Biometric Face Gallery

Part 1 extracts 512-dimensional AdaFace facial embeddings, normalizes them via Unit $L_2$ norm ($V / \|V\|_2$), and serializes them into an $O(\log N)$ FAISS `IndexHNSWFlat` vector database.

### **Method A: Automated 3-Second Guided Webcam Auto-Enrollment (Recommended)**

For enrolling live subjects directly via camera:

```bash
python3 enroll_webcam.py
```

#### **Step-by-Step Instructions**:
1. **Launch Command**: Run `python3 enroll_webcam.py`.
2. **Enter Name**: Type the subject's full name (e.g. `Jane Doe`).
3. **Follow Guided Pose Sequence**:
   - **0.0s – 1.0s**: Look directly at the camera (Frontal).
   - **1.0s – 2.0s**: Turn head slightly left and right.
   - **2.0s – 3.0s**: Tilt head slightly up.
4. **Automated Pipeline Execution**:
   - Captures aligned face frames.
   - YuNet extracts 5 facial landmarks (left eye, right eye, nose tip, mouth corners).
   - Performs 5-point Umeyama similarity transform to warp the face to $112 \times 112$ canonical coordinate space.
   - Extracts unit $L_2$-normalized 512D vectors.
   - Serializes `data/faces.index` and `data/metadata.json`.
   - Syncs profiles to SQLite database (`data/faces.db`).

---

### **Method B: Manual Photo Folder Enrollment**

For enrolling subjects using existing photos (e.g., ID photos or smartphone photos):

#### **Step 1: Create Subject Directory**
Inside `data/known_faces/`, create a subfolder named after the subject:
```
data/known_faces/Jane_Smith/
```

#### **Step 2: Add Face Photos**
Place 3 to 5 clear photos of the subject into that directory:
```
data/known_faces/Jane_Smith/Frontal.jpg
data/known_faces/Jane_Smith/Left_Angle.jpg
data/known_faces/Jane_Smith/Right_Angle.jpg
```

#### **Step 3: Run Biometric Gallery Compiler**
Execute the gallery compiler script:

```bash
python3 enroll.py
```

#### **Step 4: Verification Output**
Verify that the compiler successfully generated the following artifacts:
- `data/faces.index` (FAISS `IndexHNSWFlat` vector database)
- `data/metadata.json` (ID-to-Name metadata map)
- `data/faces.db` (SQLite WAL database)

---

## 🔵 Part 2: Fine-Tuning the YOLO Object Detector (`train_detector.py`)

Part 2 orchestrates PyTorch transfer learning for YOLOv8/YOLO26-Nano to detect custom objects, person bounding boxes, or intruder hands in 1080p CCTV feeds.

---

### **Step 1: Automated Dataset Preparation & Auto-Annotation**

Run the automated dataset builder script to convert existing face photos and event snapshots into a YOLO dataset:

```bash
python3 prepare_yolo_dataset.py
```

#### **What `prepare_yolo_dataset.py` does automatically**:
1. **Scans Images**: Auto-discovers images in `data/known_faces/` and `events/`.
2. **Auto-Generates Labels**:
   - Uses YuNet to detect face bounding boxes (Class `1: face`).
   - Uses YOLO to detect person bounding boxes (Class `0: person`).
   - Formats bounding box labels into normalized YOLO TXT coordinate format:
     `<class_id> <x_center> <y_center> <width> <height>`
3. **Dataset Structure Creation**:
   Organizes the dataset into standard training/validation directories:
   ```
   data/dataset/
   ├── images/train/ (80% of images)
   ├── images/val/   (20% of images)
   ├── labels/train/ (80% of labels)
   └── labels/val/   (20% of labels)
   ```
4. **Generates `custom_dataset.yaml`**:
   Creates the dataset configuration YAML:
   ```yaml
   path: data/dataset
   train: images/train
   val: images/val

   names:
     0: person
     1: face
     2: intruder_hand
   ```

---

### **Step 2: Launch PyTorch Transfer Learning Training**

Start the PyTorch transfer learning loop using `train_detector.py`:

```bash
python3 train_detector.py --data custom_dataset.yaml --epochs 100 --batch 16 --imgsz 640
```

#### **Command Arguments**:
- `--data`: Path to dataset config YAML (`custom_dataset.yaml`).
- `--weights`: Pretrained weights (`yolov8n.pt`).
- `--epochs`: Total training iterations (default: 100).
- `--batch`: Batch size optimized for memory (default: 16).
- `--imgsz`: Target inference resolution (default: 640).

---

### **Step 3: Validation Metrics & Model Deployment**

During and after training, `train_detector.py`:
1. **Hardware Acceleration**: Automatically selects **Apple Silicon MPS (Metal GPU)** on macOS.
2. **Evaluates Metrics**: Evaluates performance on the held-out validation set and outputs:
   - **`mAP@50`**: Mean Average Precision at IoU 0.50.
   - **`mAP@50-95`**: Mean Average Precision across IoU 0.50:0.95.
3. **Model Deployment**: Saves best weights to `runs/detect/cctv_yolo_detector/weights/best.pt` and automatically copies them to `yolov8n_custom.pt` for live CCTV inference.

---

## 🛠️ Maintenance & Troubleshooting Commands

### **Rebuilding Multi-Pose Database Index**
If you update gallery images and need to force a full multi-pose FAISS rebuild:
```bash
python3 -m src.utils.rebuild_db
```

### **Testing Real-Time Pipeline Performance**
To verify biometric logic gates, temporal consensus voting, and spatial uniqueness constraints:
```bash
python3 mock_pipeline_validator.py
```

### **Running Full Stress & HOTA LocA Benchmark Suite**
```bash
python3 run_stress_tests.py
```

---

## 📁 Related Handoff Documents
- [Handoff System Overview](file:///Users/ceaser/Documents/School/CCTV%20Project/handoff/handoff.md)
- [System Problems & Solutions Log](file:///Users/ceaser/Documents/School/CCTV%20Project/handoff/system_problems_and_solutions.md)

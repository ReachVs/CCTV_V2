#!/usr/bin/env python3
"""
HDR & Backlit Multi-Exposure Biometric Gallery Ingestion Utility
-----------------------------------------------------------------
Automates the loading, CLAHE contrast lifting, and multi-exposure registration 
of subject photos in 'data/known_faces/' to harden the system against extreme 
backlit shadows and underexposed lighting conditions.
"""

import os
import sys
import cv2
import numpy as np
import json
import sqlite3
from deepface import DeepFace

# Force single-threaded execution for OpenMP and OpenBLAS to prevent threading conflict crashes (Exit Code 139)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.face_align import YuNetFaceAligner, apply_clahe_contrast_lifting
from src.database.enroll import l2_normalize, DATA_DIR, FACES_DIR, INDEX_PATH, METADATA_PATH

try:
    import faiss
except ImportError:
    print("[ERROR] FAISS vector library not installed. Run: pip install faiss-cpu")
    sys.exit(1)

def extract_hdr_embedding(crop):
    if crop is None or crop.size == 0:
        return None
    try:
        reps = DeepFace.represent(
            img_path=crop,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False
        )
        if reps and "embedding" in reps[0]:
            return l2_normalize(reps[0]["embedding"])
    except Exception as e:
        print(f"[HDR Extraction Warning] {e}")
    return None

def process_hdr_photo_registration():
    print("=" * 80)
    print("🌅 HIGH-DYNAMIC-RANGE (HDR) & BACKLIT GALLERY REGISTRATION ENGINE")
    print("=" * 80)
    
    known_faces_dir = FACES_DIR
    if not os.path.exists(known_faces_dir):
        print(f"[ERROR] Gallery directory not found: {known_faces_dir}")
        sys.exit(1)
        
    aligner = YuNetFaceAligner(score_threshold=0.50)
    
    hdr_crops = []
    metadata_list = []
    
    subjects = [d for d in os.listdir(known_faces_dir) if os.path.isdir(os.path.join(known_faces_dir, d)) and not d.startswith('.')]
    subjects.sort()
    
    total_images_processed = 0
    total_hdr_variants = 0
    
    for subject_folder in subjects:
        display_name = subject_folder.replace("_", " ").strip()
        subject_path = os.path.join(known_faces_dir, subject_folder)
        img_files = [f for f in os.listdir(subject_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        img_files.sort()
        
        print(f"\n[HDR Ingestion] Processing gallery for subject: '{display_name}' ({len(img_files)} base photos)...")
        
        for img_name in img_files:
            img_path = os.path.join(subject_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            total_images_processed += 1
            
            # 1. Base Alignment
            aligned_base = aligner.align_face(img, output_size=(112, 112), strict=True)
            if aligned_base is None:
                aligned_base = cv2.resize(img, (112, 112))
                
            # 2. Standard Exposure Variant
            hdr_crops.append(aligned_base)
            metadata_list.append({"id": len(hdr_crops) - 1, "name": display_name, "exposure": "standard"})
            total_hdr_variants += 1
            
            # 3. CLAHE Contrast-Lifted Exposure Variant (Lifts backlit shadows)
            clahe_lifted = apply_clahe_contrast_lifting(aligned_base, clip_limit=3.5, tile_grid_size=(8, 8))
            hdr_crops.append(clahe_lifted)
            metadata_list.append({"id": len(hdr_crops) - 1, "name": display_name, "exposure": "clahe_lifted"})
            total_hdr_variants += 1

            # 4. Gamma Brightened Variant (Recovers deep shadow features)
            gamma_val = 1.4
            inv_gamma = 1.0 / gamma_val
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            gamma_brightened = cv2.LUT(aligned_base, table)
            hdr_crops.append(gamma_brightened)
            metadata_list.append({"id": len(hdr_crops) - 1, "name": display_name, "exposure": "gamma_boost"})
            total_hdr_variants += 1

    print(f"\n[HDR Ingestion] Extracting 512D ArcFace embeddings for {total_hdr_variants} multi-exposure variants...")
    embeddings = []
    valid_metadata = []
    
    for idx, (crop, meta) in enumerate(zip(hdr_crops, metadata_list)):
        vec = extract_hdr_embedding(crop)
        if vec is not None and vec.size == 512:
            quality_norm = float(np.linalg.norm(vec))
            embeddings.append(vec)
            meta["id"] = len(embeddings) - 1
            meta["quality_norm"] = float(quality_norm)
            valid_metadata.append(meta)

    if not embeddings:
        print("[ERROR] No valid embeddings extracted.")
        sys.exit(1)
        
    embeddings_matrix = np.array(embeddings, dtype=np.float32)
    dim = 512
    num_profiles = embeddings_matrix.shape[0]
    
    print(f"\n[HDR Ingestion] Building FAISS IndexHNSWFlat with {num_profiles} multi-exposure profiles...")
    M = 32
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 16
    index.add(embeddings_matrix)
    
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"[HDR Ingestion] Saved FAISS HNSW Index to '{INDEX_PATH}' ({num_profiles} profiles)")
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(valid_metadata, f, indent=2)
    print(f"[HDR Ingestion] Saved metadata to '{METADATA_PATH}'")
    
    print("\n" + "=" * 80)
    print(f"✅ HDR GALLERY INGESTION COMPLETE! ({total_images_processed} base photos -> {num_profiles} multi-exposure profiles)")
    print("=" * 80)

if __name__ == "__main__":
    process_hdr_photo_registration()

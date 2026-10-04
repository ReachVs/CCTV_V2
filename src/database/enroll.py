import os
import sys
import json
import re
import random
import numpy as np
import faiss

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
from src.utils.face_align import YuNetFaceAligner
from src.utils.fsrcnn_upscaler import FSRCNNUpscaler
from src.inference.face_embedder import ArcFaceEmbedder

aligner = YuNetFaceAligner()
upscaler = FSRCNNUpscaler()
embedder = ArcFaceEmbedder(model_name="ArcFace", face_aligner=aligner, upscaler=upscaler)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FACES_DIR = os.path.join(DATA_DIR, "known_faces")
INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
DIMENSION = 512

def l2_normalize(vector):
    v = np.array(vector, dtype=np.float32)
    dot_val = float(np.dot(v, v))
    if dot_val == 0.0:
        return v
    return v / np.sqrt(dot_val)

def get_image_files(directory):
    valid_extensions = (".jpg", ".jpeg", ".png")
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[Enrollment] Created directory: '{directory}'")
        return []
    
    files = []
    for root, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith(valid_extensions):
                files.append(os.path.join(root, f))
    return sorted(files)

def extract_subject_name(img_path, root_dir="known_faces"):
    relative_path = os.path.relpath(img_path, root_dir)
    parts = relative_path.split(os.sep)
    
    standard_folders = {"frontal", "left_profile", "right_profile", "left", "right", "side_profile"}
    
    if len(parts) == 1:
        raw_name = os.path.splitext(parts[0])[0]
    elif len(parts) == 2:
        parent, filename = parts
        if parent.lower() in standard_folders:
            raw_name = os.path.splitext(filename)[0]
        else:
            raw_name = parent
    else:
        if parts[0].lower() not in standard_folders:
            raw_name = parts[0]
        else:
            raw_name = os.path.splitext(parts[-1])[0]
            
    # Clean trailing numbers (e.g. Ceaser_1 -> Ceaser) and profile tags (e.g. Ceaser_left -> Ceaser)
    raw_name = re.sub(r'[-_]\d+$', '', raw_name)
    raw_name = re.sub(r'[-_](left|right|frontal|profile|side)$', '', raw_name, flags=re.IGNORECASE)
    
    return raw_name.replace("_", " ").title()

def run_enrollment():
    print("[Enrollment] Initializing face enrollment pipeline (AdaFace 512D)...")
    image_files = get_image_files(FACES_DIR)
    
    embeddings_list = []
    names_list = []
    
    if not image_files:
        print(f"[Enrollment] No images found in '{FACES_DIR}' directory.")
        print("[Enrollment] Creating a mock profile for testing purposes...")
        
        # Create a mock 512-dimensional embedding and normalize it
        mock_embedding = [random.uniform(-1.0, 1.0) for _ in range(DIMENSION)]
        normalized_mock = l2_normalize(mock_embedding)
        
        embeddings_list.append(normalized_mock)
        names_list.append("Mock Subject (John Doe)")
        
        print(f"\n=== TO ENROLL REAL FACES ===")
        print(f"1. Place clear face images of individuals in '{FACES_DIR}' (e.g. 'Ceaser.jpg').")
        print("2. Re-run this script: python3 -m src.database.enroll")
        print("============================\n")
    else:
        print(f"[Enrollment] Found {len(image_files)} image(s) to process.")
        
        for img_path in image_files:
            name = extract_subject_name(img_path, FACES_DIR)
            print(f"[Enrollment] Extracting ArcFace biometric embedding for '{name}'...")
            
            try:
                img_cv = cv2.imread(img_path)
                if img_cv is None or img_cv.size == 0:
                    continue

                # Primary YuNet 5-point landmark alignment & embedding
                emb = embedder.embed(img_cv, align=True, strict_alignment=True, upscale=False)
                if emb is None:
                    # Non-strict alignment fallback
                    emb = embedder.embed(img_cv, align=True, strict_alignment=False, upscale=False)

                if emb is not None:
                    embeddings_list.append(emb)
                    names_list.append(name)
                    print(f"[Enrollment] Successfully enrolled profile for: '{name}'")
                else:
                    print(f"[Enrollment Warning] Skipping '{img_path}' for '{name}' - Could not extract embedding.")
                    
            except Exception as e:
                print(f"[Enrollment] Error processing {img_path}: {e}")
                
    # Build FAISS vector database (HNSWFlat with FlatL2 fallback)
    if embeddings_list:
        print(f"[Enrollment] Building FAISS Vector Index with {len(embeddings_list)} profiles...")
        embeddings_matrix = np.vstack(embeddings_list).astype(np.float32)
        
        try:
            # Initialize IndexHNSWFlat for O(log N) approximate nearest-neighbor search
            index = faiss.IndexHNSWFlat(DIMENSION, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            index.add(embeddings_matrix)
            print("[Enrollment] Built FAISS IndexHNSWFlat vector database.")
        except Exception as hnsw_err:
            print(f"[Enrollment Warning] IndexHNSWFlat init failed ({hnsw_err}). Falling back to IndexFlatL2.")
            index = faiss.IndexFlatL2(DIMENSION)
            index.add(embeddings_matrix)
            print("[Enrollment] Built FAISS IndexFlatL2 exact vector database.")
        
        # Save FAISS Index
        faiss.write_index(index, INDEX_PATH)
        print(f"[Enrollment] FAISS index saved to '{INDEX_PATH}'")
        
        # Save Metadata mapping
        existing_meta = {}
        if os.path.exists(METADATA_PATH):
            try:
                with open(METADATA_PATH, "r") as f:
                    existing_meta = json.load(f)
            except Exception:
                existing_meta = {}

        metadata = {}
        for i, name in enumerate(names_list):
            old_info = existing_meta.get(str(i), {})
            if isinstance(old_info, dict) and "name" in old_info:
                info = dict(old_info)
                info["name"] = name
                metadata[str(i)] = info
            else:
                metadata[str(i)] = {
                    "name": name,
                    "occupation": "SECURITY SPECIALIST",
                    "clearance": "LEVEL 1 GRANTED"
                }

        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[Enrollment] Metadata map saved to '{METADATA_PATH}'")

        # Sync SQLite faces.db
        try:
            import sqlite3
            db_path = os.path.join(DATA_DIR, "faces.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL
                )
            """)
            
            # Aggregate per unique subject
            subject_embeddings = {}
            for name, emb in zip(names_list, embeddings_list):
                if name not in subject_embeddings:
                    subject_embeddings[name] = []
                subject_embeddings[name].append(emb)

            cursor.execute("DELETE FROM people;")
            for s_name, s_embs in subject_embeddings.items():
                mean_emb = np.mean(s_embs, axis=0)
                mean_emb = l2_normalize(mean_emb)
                cursor.execute("INSERT INTO people (name, embedding) VALUES (?, ?)", (s_name, json.dumps(mean_emb.tolist())))
            conn.commit()
            conn.close()
            print(f"[Enrollment] Synced {len(subject_embeddings)} unique subjects to SQLite database at '{db_path}'.")
        except Exception as db_err:
            print(f"[Enrollment Warning] SQLite database sync: {db_err}")

        print("[Enrollment] Enrollment completed successfully.")
    else:
        print("[Enrollment] Error: No profiles generated. Database not built.")

if __name__ == "__main__":
    run_enrollment()

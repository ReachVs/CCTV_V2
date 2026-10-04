import os
import sys
import json
import sqlite3
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

from src.utils.face_align import YuNetFaceAligner
from deepface import DeepFace
import faiss

def rebuild_biometric_database():
    print("=" * 65)
    print("  BIOMETRIC DATABASE REBUILD & MULTI-TEMPLATE INDEXER")
    print("=" * 65)
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[ERROR] Directory '{KNOWN_FACES_DIR}' not found.")
        return

    # Initialize ArcFace model & YuNet aligner
    print("[INIT] Loading YuNet face aligner and ArcFace embedding model...")
    aligner = YuNetFaceAligner()
    keras_model = None
    try:
        model_wrapper = DeepFace.build_model("ArcFace")
        keras_model = model_wrapper.model
        print("[INIT] ArcFace Keras model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Could not load Keras model directly: {e}. Fallback to DeepFace.represent.")

    vectors = []
    metadata_dict = {}
    db_people_records = {} # name -> list of vectors for SQLite averaging

    folders = sorted([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
    
    for subject_dir_name in folders:
        subject_path = os.path.join(KNOWN_FACES_DIR, subject_dir_name)
        display_name = subject_dir_name.replace("_", " ").strip()
        
        image_files = sorted([f for f in os.listdir(subject_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        print(f"\nProcessing subject: '{display_name}' ({len(image_files)} image files)")
        
        subject_vectors = []
        for img_name in image_files:
            img_path = os.path.join(subject_path, img_name)
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                continue
                
            # Align face using YuNet + Umeyama transform
            aligned = aligner.align_face(img, output_size=(112, 112), strict=False)
            if aligned is None:
                aligned = cv2.resize(img, (112, 112))
                
            v = None
            if keras_model is not None:
                img_tensor = (aligned.astype(np.float32) / 255.0)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                v = keras_model(img_tensor, training=False).numpy().flatten()
            else:
                try:
                    reps = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
                    if reps:
                        v = np.array(reps[0]["embedding"], dtype=np.float32)
                except Exception as e:
                    print(f"  [Warning] Extracting {img_name}: {e}")
                    
            if v is not None:
                norm_val = np.linalg.norm(v)
                if norm_val > 0:
                    v = v / norm_val
                    vector_idx = len(vectors)
                    vectors.append(v)
                    subject_vectors.append(v)
                    
                    pose_tag = img_name.split("_")[-2] if "_" in img_name else "Frontal"
                    metadata_dict[str(vector_idx)] = {
                        "name": display_name,
                        "file": img_name,
                        "pose": pose_tag,
                        "occupation": "SECURITY SPECIALIST",
                        "clearance": "LEVEL 1 GRANTED"
                    }
                    print(f"  + Added template #{vector_idx}: {img_name} ({pose_tag})")
                    
        if subject_vectors:
            # Mean vector for SQLite single-record sync
            mean_v = np.mean(subject_vectors, axis=0)
            mean_v = mean_v / (np.linalg.norm(mean_v) + 1e-10)
            db_people_records[display_name] = mean_v.tolist()

    if not vectors:
        print("\n[ERROR] No valid facial embedding vectors extracted.")
        return

    # Build multi-template FAISS index
    index = faiss.IndexFlatL2(512)
    index.add(np.array(vectors, dtype=np.float32))
    faiss.write_index(index, INDEX_PATH)
    print(f"\n[SUCCESS] Built FAISS Index '{INDEX_PATH}' with {index.ntotal} multi-pose templates across {len(db_people_records)} subjects.")

    # Write metadata.json
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata_dict, f, indent=4)
    print(f"[SUCCESS] Wrote metadata registry to '{METADATA_PATH}'.")

    # Sync SQLite faces.db people table
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL
            )
        """)
        # Clear out old outdated records
        cursor.execute("DELETE FROM people;")
        for name, emb_list in db_people_records.items():
            cursor.execute(
                "INSERT INTO people (name, embedding) VALUES (?, ?)",
                (name, json.dumps(emb_list))
            )
        conn.commit()
        conn.close()
        print(f"[SUCCESS] Synced {len(db_people_records)} identity records to SQLite database '{DB_PATH}'.")
    except Exception as e:
        print(f"[WARNING] SQLite sync error: {e}")

    # Trigger live engine reload if running
    try:
        import src.inference.live_cctv as live_cctv
        live_cctv.reload_biometric_db()
        print("[SUCCESS] Reloaded live biometric verification engine.")
    except Exception:
        pass

    print("\n" + "=" * 65)
    print("  DATABASE REBUILD COMPLETE - MULTI-POSE SEARCH READY")
    print("=" * 65)

if __name__ == "__main__":
    rebuild_biometric_database(rename_ceo_to_ceaser=True)

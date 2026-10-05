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

from src.inference.face_embedder import ArcFaceEmbedder
import faiss

def rebuild_biometric_database():
    print("=" * 65)
    print("  BIOMETRIC DATABASE REBUILD & MULTI-TEMPLATE INDEXER")
    print("=" * 65)
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"[ERROR] Directory '{KNOWN_FACES_DIR}' not found.")
        return

    # Initialize ArcFace embedder (YuNet aligner + Keras tensor engine)
    print("[INIT] Initializing unified ArcFaceEmbedder...")
    embedder = ArcFaceEmbedder(model_name="ArcFace")

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

            h, w = img.shape[:2]
            if max(h, w) > 1280:
                scale = 1080.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                
            # Unified alignment and ArcFace extraction
            v = embedder.embed(img, align=True, strict_alignment=False, upscale=False)
            if v is not None:
                vector_idx = len(vectors)
                vectors.append(v)
                subject_vectors.append(v)
                stem = os.path.splitext(img_name)[0]
                pose_tag = stem if stem in ["Frontal", "Left_Angle", "Right_Angle", "Side_Left_Angle", "Side_Right_Angle", "Tilt_Up", "Working_Angle"] else (img_name.split("_")[-2] if "_" in img_name else "Frontal")
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
            db_people_records[display_name] = {
                "mean": mean_v.tolist(),
                "vectors": subject_vectors
            }

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

    # Sync SQLite faces.db people and person_vectors tables
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                embedding TEXT NOT NULL,
                occupation TEXT DEFAULT 'SECURITY SPECIALIST',
                clearance TEXT DEFAULT 'LEVEL 1 GRANTED',
                phone TEXT DEFAULT 'N/A',
                dob TEXT DEFAULT 'N/A',
                address TEXT DEFAULT 'SYSTEM DATABASE',
                threat_level TEXT DEFAULT 'LOW',
                created_at TEXT DEFAULT '2026-10-05 00:00:00',
                updated_at TEXT DEFAULT '2026-10-05 00:00:00'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                vector_idx INTEGER NOT NULL,
                embedding TEXT NOT NULL,
                pose TEXT DEFAULT 'frontal',
                image_path TEXT,
                created_at TEXT DEFAULT '2026-10-05 00:00:00',
                FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
            )
        """)
        # Clear out old outdated records
        cursor.execute("DELETE FROM person_vectors;")
        cursor.execute("DELETE FROM people;")
        for name, data in db_people_records.items():
            cursor.execute(
                "INSERT INTO people (name, embedding) VALUES (?, ?)",
                (name, json.dumps(data["mean"]))
            )
            person_id = cursor.lastrowid
            for v_idx, v_arr in enumerate(data["vectors"]):
                cursor.execute(
                    "INSERT INTO person_vectors (person_id, vector_idx, embedding, pose) VALUES (?, ?, ?, ?)",
                    (person_id, v_idx, json.dumps(v_arr.tolist() if hasattr(v_arr, "tolist") else list(v_arr)), f"pose_{v_idx}")
                )
        conn.commit()
        conn.close()
        print(f"[SUCCESS] Synced {len(db_people_records)} identity records and {len(vectors)} template vectors to SQLite database '{DB_PATH}'.")
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
    rebuild_biometric_database()

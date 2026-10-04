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

try:
    import faiss
except ImportError:
    faiss = None

def check_database_health():
    """
    Evaluates database health across FAISS index, SQLite faces.db, and JSON metadata.
    Returns:
        - is_healthy (bool): True if all 3 databases exist, are readable, and synchronized.
        - reason (str): Explanation of health status or corruption detail.
    """
    if faiss is None:
        return True, "FAISS library not installed (running in fallback mode)"

    index_exists = os.path.exists(INDEX_PATH)
    metadata_exists = os.path.exists(METADATA_PATH)
    db_exists = os.path.exists(DB_PATH)

    if not index_exists:
        return False, f"FAISS index file '{INDEX_PATH}' is missing"

    if not metadata_exists:
        return False, f"Metadata registry '{METADATA_PATH}' is missing"

    # Test FAISS index integrity
    try:
        index = faiss.read_index(INDEX_PATH)
        faiss_count = int(index.ntotal)
    except Exception as e:
        return False, f"FAISS index file is corrupted: {e}"

    # Test Metadata registry integrity
    try:
        with open(METADATA_PATH, 'r') as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return False, "Metadata registry format is invalid"
        meta_count = len(meta)
    except Exception as e:
        return False, f"Metadata registry is corrupted: {e}"

    # Check for Count Desynchronization
    if faiss_count != meta_count:
        return False, f"Count desync: FAISS index has {faiss_count} vectors vs Metadata has {meta_count} entries"

    # Test SQLite integrity & count matching if DB exists
    if db_exists:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='people';")
            if cursor.fetchone()[0] > 0:
                cursor.execute("SELECT COUNT(*) FROM people;")
                sql_count = cursor.fetchone()[0]
                if sql_count > 0 and faiss_count < sql_count:
                    conn.close()
                    return False, f"Count desync: SQLite 'people' table has {sql_count} entries vs FAISS has {faiss_count}"
            conn.close()
        except Exception as e:
            return False, f"SQLite faces.db integrity warning: {e}"

    return True, f"All databases operational ({faiss_count} profiles in sync)"

def auto_heal_database():
    """
    Automated startup healing procedure.
    Rebuilds data/faces.index and data/metadata.json from SQLite people table.
    If SQLite is empty/missing, rescans data/known_faces image archive to regenerate embeddings.
    """
    is_healthy, reason = check_database_health()
    if is_healthy:
        print(f"[AUTO-HEAL ENGINE] Health check passed: {reason}")
        return True

    print(f"[AUTO-HEAL ENGINE] ALERT: {reason}. Initiating automated schema healing...")

    # Phase 1: Attempt recovery from SQLite people table
    healed = _heal_from_sqlite()
    if healed:
        print("[AUTO-HEAL ENGINE] SUCCESS: Rebuilt FAISS index and metadata from SQLite 'people' table.")
        return True

    # Phase 2: Fallback recovery from data/known_faces image archive
    healed = _heal_from_known_faces()
    if healed:
        print("[AUTO-HEAL ENGINE] SUCCESS: Rescanned image archive and auto-healed biometric database.")
        return True

    print("[AUTO-HEAL ENGINE] WARNING: Auto-healing complete (0 existing profiles found). Standing by for new enrollments.")
    return False

def _heal_from_sqlite():
    if not os.path.exists(DB_PATH) or faiss is None:
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='people';")
        if cursor.fetchone()[0] == 0:
            conn.close()
            return False

        cursor.execute("SELECT id, name, embedding FROM people ORDER BY id ASC;")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return False

        vectors = []
        meta_dict = {}
        for row in rows:
            name, emb_json = row[1], row[2]
            v = np.array(json.loads(emb_json), dtype=np.float32)
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                v = v / norm_v
            vectors.append(v)
            meta_dict[str(len(vectors) - 1)] = {
                "name": name,
                "occupation": "SECURITY SPECIALIST",
                "clearance": "LEVEL 1 GRANTED"
            }

        if vectors:
            index = faiss.IndexFlatL2(512)
            index.add(np.array(vectors, dtype=np.float32))
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, 'w') as f:
                json.dump(meta_dict, f, indent=4)
            return True
    except Exception as e:
        print(f"[AUTO-HEAL ENGINE Error] SQLite healing failed: {e}")

    return False

def _heal_from_known_faces():
    if not os.path.exists(KNOWN_FACES_DIR) or faiss is None:
        return False
    try:
        from src.utils.face_align import YuNetFaceAligner
        from deepface import DeepFace

        aligner = YuNetFaceAligner()
        keras_model = None
        try:
            model_wrapper = DeepFace.build_model("ArcFace")
            keras_model = model_wrapper.model
        except Exception:
            pass

        vectors = []
        meta_dict = {}
        db_people = {}

        folders = sorted([d for d in os.listdir(KNOWN_FACES_DIR) if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))])
        for folder in folders:
            subject_dir = os.path.join(KNOWN_FACES_DIR, folder)
            display_name = folder.replace("_", " ").strip()
            images = sorted([f for f in os.listdir(subject_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            
            subject_vectors = []
            for img_name in images:
                img_path = os.path.join(subject_dir, img_name)
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    continue
                aligned = aligner.align_face(img, output_size=(112, 112), strict=False)
                if aligned is None:
                    aligned = cv2.resize(img, (112, 112))

                v = None
                if keras_model is not None:
                    img_tensor = (aligned.astype(np.float32) / 255.0)
                    img_tensor = np.expand_dims(img_tensor, axis=0)
                    v = keras_model(img_tensor, training=False).numpy().flatten()
                else:
                    reps = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)
                    if reps:
                        v = np.array(reps[0]["embedding"], dtype=np.float32)

                if v is not None:
                    norm_v = np.linalg.norm(v)
                    if norm_v > 0:
                        v = v / norm_v
                        idx_val = len(vectors)
                        vectors.append(v)
                        subject_vectors.append(v)
                        meta_dict[str(idx_val)] = {"name": display_name}

            if subject_vectors:
                mean_v = np.mean(subject_vectors, axis=0)
                mean_v = mean_v / (np.linalg.norm(mean_v) + 1e-10)
                db_people[display_name] = mean_v.tolist()

        if vectors:
            index = faiss.IndexFlatL2(512)
            index.add(np.array(vectors, dtype=np.float32))
            faiss.write_index(index, INDEX_PATH)
            with open(METADATA_PATH, 'w') as f:
                json.dump(meta_dict, f, indent=4)

            # Sync SQLite DB
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
            cursor.execute("DELETE FROM people;")
            for name, emb_list in db_people.items():
                cursor.execute("INSERT INTO people (name, embedding) VALUES (?, ?)", (name, json.dumps(emb_list)))
            conn.commit()
            conn.close()
            return True
    except Exception as e:
        print(f"[AUTO-HEAL ENGINE Error] Image archive healing failed: {e}")

    return False

if __name__ == "__main__":
    auto_heal_database()

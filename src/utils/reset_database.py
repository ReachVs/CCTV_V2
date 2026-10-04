import os
import sys
import json
import sqlite3
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
DEBUG_CROPS_DIR = os.path.join(DATA_DIR, "debug_crops")
INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

try:
    import faiss
except ImportError:
    faiss = None

def reset_entire_database():
    print("=" * 65)
    print("  PURGING ALL BIOMETRIC DATA & RE-SETTING SYSTEM TO CLEAN STATE")
    print("=" * 65)

    # 1. Purge known_faces directory
    if os.path.exists(KNOWN_FACES_DIR):
        for item in os.listdir(KNOWN_FACES_DIR):
            p = os.path.join(KNOWN_FACES_DIR, item)
            if os.path.isdir(p):
                shutil.rmtree(p)
                print(f"[-] Deleted folder: '{p}'")
            elif os.path.isfile(p) and not item.startswith("."):
                os.remove(p)
                print(f"[-] Deleted image: '{p}'")

    # 2. Purge debug crops
    if os.path.exists(DEBUG_CROPS_DIR):
        for item in os.listdir(DEBUG_CROPS_DIR):
            p = os.path.join(DEBUG_CROPS_DIR, item)
            if os.path.isfile(p):
                os.remove(p)

    # 3. Reset SQLite Database tables
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("DROP TABLE IF EXISTS people;")
            cursor.execute("DROP TABLE IF EXISTS detection_events;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embedding TEXT NOT NULL
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    match_distance REAL NOT NULL,
                    snapshot_path TEXT
                );
            """)
            conn.commit()
            conn.close()
            print(f"[CLEAN] Reset SQLite tables 'people' and 'detection_events' in '{DB_PATH}'")
        except Exception as e:
            print(f"[WARNING] SQLite reset error: {e}")

    # 4. Reset FAISS Index file to empty index
    if faiss is not None:
        index = faiss.IndexFlatL2(512)
        faiss.write_index(index, INDEX_PATH)
        print(f"[CLEAN] Reset FAISS Index '{INDEX_PATH}' (0 profiles)")
    elif os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    # 5. Reset Metadata JSON to empty object
    with open(METADATA_PATH, 'w') as f:
        json.dump({}, f, indent=4)
    print(f"[CLEAN] Reset Metadata Registry '{METADATA_PATH}' ({0} profiles)")

    # 6. Trigger live engine database reload
    try:
        import src.inference.live_cctv as live_cctv
        live_cctv.reload_biometric_db()
        print("[CLEAN] Reloaded live biometric engine.")
    except Exception:
        pass

    print("\n" + "=" * 65)
    print("  ALL BIOMETRIC DATA PURGED SUCCESSFULLY. READY FOR NEW ENROLLMENT.")
    print("=" * 65)

if __name__ == "__main__":
    reset_entire_database()

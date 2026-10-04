import os
import sys
import sqlite3
import json
import random

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "faces.db")
FACES_DIR = os.path.join(DATA_DIR, "known_faces")

def initialize_database():
    print(f"[Database Setup] Initializing SQLite database at: '{DB_PATH}'")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person_name TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            match_distance REAL NOT NULL,
            snapshot_path TEXT
        )
    """)
    conn.commit()
    return conn, cursor

def get_image_files(directory):
    valid_extensions = (".jpg", ".jpeg", ".png")
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[Database Setup] Created directory: '{directory}'")
        return []
    
    files = [
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(valid_extensions)
    ]
    return files

def enroll_faces():
    conn, cursor = initialize_database()
    image_files = get_image_files(FACES_DIR)
    
    if not image_files:
        print(f"\n[Database Setup] No images found in '{FACES_DIR}' directory.")
        print("[Database Setup] Creating a mock profile for testing purposes...")
        # 512-dimensional embedding for AdaFace
        mock_embedding = [random.uniform(-0.1, 0.1) for _ in range(512)]
        try:
            cursor.execute(
                "INSERT INTO people (name, embedding) VALUES (?, ?)",
                ("Mock Subject (John Doe)", json.dumps(mock_embedding))
            )
            conn.commit()
            print("[Database Setup] Successfully inserted mock profile: 'Mock Subject (John Doe)'")
        except sqlite3.IntegrityError:
            print("[Database Setup] Mock profile 'Mock Subject (John Doe)' already exists in database.")
        
        conn.close()
        return

    print(f"[Database Setup] Found {len(image_files)} image(s) to process.")
    
    try:
        from deepface import DeepFace
    except ImportError:
        print("[Database Setup Warning] DeepFace not installed. Skipping live representation extraction.")
        conn.close()
        return

    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0].replace("_", " ").title()
        print(f"[Database Setup] Processing image for '{name}' ({img_path}) using AdaFace 512D model...")
        
        try:
            try:
                representations = DeepFace.represent(
                    img_path=img_path,
                    model_name="AdaFace",
                    detector_backend="retinaface",
                    enforce_detection=False
                )
            except ValueError as ve:
                if "AdaFace" in str(ve) or "not supported" in str(ve).lower():
                    representations = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        detector_backend="retinaface",
                        enforce_detection=False
                    )
                else:
                    raise ve
            
            if representations:
                embedding = representations[0]["embedding"]
                embedding_json = json.dumps(embedding)
                
                cursor.execute(
                    "INSERT INTO people (name, embedding) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding",
                    (name, embedding_json)
                )
                conn.commit()
                print(f"[Database Setup] Successfully enrolled: '{name}'")
            else:
                print(f"[Database Setup] Warning: Could not detect face in '{img_path}'.")
        except Exception as e:
            print(f"[Database Setup] Error processing '{img_path}': {e}")
            
    conn.close()

if __name__ == "__main__":
    enroll_faces()

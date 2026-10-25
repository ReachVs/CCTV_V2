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

def initialize_database(db_path: str = DB_PATH):
    print(f"[Database Setup] Initializing SQLite database at: '{db_path}'")
    conn = sqlite3.connect(db_path)
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS courses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            department TEXT DEFAULT 'GENERAL',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            day_of_week INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            grace_minutes INTEGER DEFAULT 15,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS course_enrollments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
            person_name TEXT NOT NULL,
            enrolled_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(course_id, person_name)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            person_name TEXT NOT NULL,
            check_in_time TEXT NOT NULL,
            status TEXT NOT NULL,
            track_id INTEGER DEFAULT 0,
            match_distance REAL DEFAULT 0.0,
            snapshot_path TEXT DEFAULT '',
            course_id INTEGER DEFAULT NULL,
            is_manual INTEGER DEFAULT 0,
            override_reason TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, person_name)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            attendance_id INTEGER,
            person_name TEXT NOT NULL,
            old_status TEXT NOT NULL,
            new_status TEXT NOT NULL,
            reason TEXT NOT NULL,
            changed_by TEXT DEFAULT 'TEACHER',
            changed_at TEXT DEFAULT CURRENT_TIMESTAMP
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

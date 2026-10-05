"""
Biometric Repository & Synchronization Manager.
Authoritative source of truth for biometric subjects, identities, and multi-pose face vectors.
Maintains atomic synchronization between SQLite (faces.db), FAISS (faces.index), and in-memory cache (metadata.json).
"""
import os
import json
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DB_PATH = os.path.join(DATA_DIR, "faces.db")
DEFAULT_INDEX_PATH = os.path.join(DATA_DIR, "faces.index")
DEFAULT_METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    v = np.array(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        return v / norm
    return v


class BiometricRepository:
    """
    Authoritative Biometric Repository managing persistent SQLite profiles
    and real-time FAISS vector synchronization.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH
        self._lock = threading.RLock()
        self._ensure_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Ensures all tables and required columns exist without dropping data."""
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            conn = self._get_connection()
            cur = conn.cursor()

            # Ensure people table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    embedding TEXT,
                    occupation TEXT DEFAULT 'SECURITY SPECIALIST',
                    clearance TEXT DEFAULT 'LEVEL 1 GRANTED',
                    phone TEXT DEFAULT 'N/A',
                    dob TEXT DEFAULT 'N/A',
                    address TEXT DEFAULT 'SYSTEM DATABASE',
                    threat_level TEXT DEFAULT 'LOW',
                    created_at TEXT DEFAULT '2026-10-05 00:00:00',
                    updated_at TEXT DEFAULT '2026-10-05 00:00:00'
                );
            """)

            # Ensure columns exist in case table was created with older schema
            cur.execute("PRAGMA table_info(people);")
            cols = {row["name"] for row in cur.fetchall()}
            column_defs = [
                ("occupation", "TEXT DEFAULT 'SECURITY SPECIALIST'"),
                ("clearance", "TEXT DEFAULT 'LEVEL 1 GRANTED'"),
                ("phone", "TEXT DEFAULT 'N/A'"),
                ("dob", "TEXT DEFAULT 'N/A'"),
                ("address", "TEXT DEFAULT 'SYSTEM DATABASE'"),
                ("threat_level", "TEXT DEFAULT 'LOW'"),
                ("created_at", "TEXT DEFAULT '2026-10-05 00:00:00'"),
                ("updated_at", "TEXT DEFAULT '2026-10-05 00:00:00'")
            ]
            for col_name, col_type in column_defs:
                if col_name not in cols:
                    cur.execute(f"ALTER TABLE people ADD COLUMN {col_name} {col_type};")

            # Ensure person_vectors table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS person_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    vector_idx INTEGER NOT NULL,
                    embedding TEXT NOT NULL,
                    pose TEXT DEFAULT 'frontal',
                    image_path TEXT,
                    created_at TEXT DEFAULT '2026-10-05 00:00:00',
                    FOREIGN KEY (person_id) REFERENCES people(id) ON DELETE CASCADE
                );
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_person_vectors_person_id ON person_vectors(person_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_person_vectors_idx ON person_vectors(vector_idx);")

            conn.commit()
            conn.close()

    def get_all_subjects(self) -> List[Dict[str, Any]]:
        """Returns all enrolled subject profiles and vector counts."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.id, p.name, p.occupation, p.clearance, p.phone, p.dob, p.address, p.threat_level,
                   COUNT(pv.id) AS vector_count
            FROM people p
            LEFT JOIN person_vectors pv ON p.id = pv.person_id
            GROUP BY p.id
            ORDER BY p.name ASC;
        """)
        results = [dict(row) for row in cur.fetchall()]
        conn.close()
        return results

    def get_subject_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Finds a subject profile by name (case-insensitive)."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT id, name, occupation, clearance, phone, dob, address, threat_level, created_at, updated_at
            FROM people
            WHERE LOWER(name) = LOWER(?);
        """, (name.strip(),))
        row = cur.fetchone()
        conn.close()
        return dict(row) if row else None

    def add_subject(
        self,
        name: str,
        embeddings: List[Any],
        profile_data: Optional[Dict[str, Any]] = None,
        poses: Optional[List[str]] = None,
        image_paths: Optional[List[str]] = None,
        trigger_reload: bool = True
    ) -> int:
        """
        Enrolls or updates a subject with multiple pose embeddings.
        Synchronizes to SQLite, FAISS index, and in-memory metadata cache.
        """
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Subject name cannot be empty.")

        if not embeddings:
            raise ValueError("At least one face embedding is required for enrollment.")

        cleaned_name = name.strip()

        # Security check: validate finite float values and dimension (defense-in-depth against FAISS SIGSEGV)
        norm_embeddings = []
        for idx, e in enumerate(embeddings):
            arr = np.array(e, dtype=np.float32).flatten()
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Embedding vector at index {idx} contains non-finite values (NaN or Inf).")
            if arr.size != 512:
                raise ValueError(f"Embedding vector at index {idx} has dimension {arr.size} (expected 512).")
            norm_embeddings.append(_l2_normalize(arr))

        profile_data = profile_data or {}
        occupation = str(profile_data.get("occupation", "SECURITY SPECIALIST")).strip()
        clearance = str(profile_data.get("clearance", "LEVEL 1 GRANTED")).strip()
        phone = str(profile_data.get("phone", "N/A")).strip()
        dob = str(profile_data.get("dob", "N/A")).strip()
        address = str(profile_data.get("address", "SYSTEM DATABASE")).strip()
        threat_level = str(profile_data.get("threat_level", "LOW")).strip()

        # Mean embedding for backward compatibility
        mean_vec = _l2_normalize(np.mean(norm_embeddings, axis=0))

        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()

            # Insert or update in people
            cur.execute("""
                INSERT INTO people (name, embedding, occupation, clearance, phone, dob, address, threat_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    embedding = excluded.embedding,
                    occupation = excluded.occupation,
                    clearance = excluded.clearance,
                    phone = excluded.phone,
                    dob = excluded.dob,
                    address = excluded.address,
                    threat_level = excluded.threat_level;
            """, (cleaned_name, json.dumps(mean_vec.tolist()), occupation, clearance, phone, dob, address, threat_level))

            cur.execute("SELECT id FROM people WHERE LOWER(name) = LOWER(?);", (cleaned_name,))
            person_id = cur.fetchone()["id"]

            # Remove previous vectors for this subject to replace with new set
            cur.execute("DELETE FROM person_vectors WHERE person_id = ?;", (person_id,))

            for i, vec in enumerate(norm_embeddings):
                pose = poses[i] if (poses and i < len(poses)) else f"pose_{i}"
                img_path = image_paths[i] if (image_paths and i < len(image_paths)) else None
                cur.execute("""
                    INSERT INTO person_vectors (person_id, vector_idx, embedding, pose, image_path)
                    VALUES (?, ?, ?, ?, ?);
                """, (person_id, 0, json.dumps(vec.tolist()), pose, img_path))

            conn.commit()
            conn.close()

            # Resequence vector indices and rebuild FAISS + metadata cache
            self.sync_faiss_and_cache(trigger_reload=trigger_reload)
            return person_id

    def remove_subject(self, name: str, trigger_reload: bool = True) -> bool:
        """
        Removes a subject and all associated vectors from SQLite,
        rebuilds FAISS index, regenerates metadata cache, and triggers hot-reload.
        """
        if not name or not isinstance(name, str) or not name.strip():
            return False

        cleaned_name = name.strip()

        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM people WHERE LOWER(name) = LOWER(?);", (cleaned_name,))
            deleted = cur.rowcount > 0
            cur.execute("DELETE FROM person_vectors WHERE person_id NOT IN (SELECT id FROM people);")
            conn.commit()
            conn.close()

            if deleted:
                self.sync_faiss_and_cache(trigger_reload=trigger_reload)
            return deleted

    def sync_faiss_and_cache(self, trigger_reload: bool = True) -> Tuple[int, int]:
        """
        Exports all vectors from SQLite into FAISS (faces.index)
        and regenerates the fast in-memory dictionary and metadata.json.
        """
        with self._lock:
            conn = self._get_connection()
            cur = conn.cursor()

            # Fetch all vectors with their parent people profiles ordered stably
            cur.execute("""
                SELECT pv.id AS vector_row_id, pv.embedding, pv.pose, pv.image_path,
                       p.id AS person_id, p.name, p.occupation, p.clearance, p.phone, p.dob, p.address, p.threat_level
                FROM person_vectors pv
                JOIN people p ON pv.person_id = p.id
                ORDER BY p.name ASC, pv.id ASC;
            """)
            rows = cur.fetchall()

            vectors_list = []
            metadata_dict: Dict[str, Any] = {}

            for new_idx, row in enumerate(rows):
                # Update vector_idx in SQLite for consistency
                cur.execute("UPDATE person_vectors SET vector_idx = ? WHERE id = ?;", (new_idx, row["vector_row_id"]))

                vec = np.array(json.loads(row["embedding"]), dtype=np.float32)
                vec = _l2_normalize(vec)
                vectors_list.append(vec)

                metadata_dict[str(new_idx)] = {
                    "name": row["name"],
                    "occupation": row["occupation"] or "SECURITY SPECIALIST",
                    "clearance": row["clearance"] or "LEVEL 1 GRANTED",
                    "phone": row["phone"] or "N/A",
                    "dob": row["dob"] or "N/A",
                    "address": row["address"] or "SYSTEM DATABASE",
                    "threat_level": row["threat_level"] or "LOW",
                    "pose": row["pose"] or "frontal",
                    "image_path": row["image_path"] or ""
                }

            conn.commit()
            conn.close()

            # Build FAISS index
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            if vectors_list and faiss is not None:
                matrix = np.array(vectors_list, dtype=np.float32)
                faiss.normalize_L2(matrix)
                index = faiss.IndexFlatL2(matrix.shape[1])
                index.add(matrix)
                faiss.write_index(index, self.index_path)
                index_count = index.ntotal
            elif faiss is not None:
                index = faiss.IndexFlatL2(512)
                faiss.write_index(index, self.index_path)
                index_count = 0
            else:
                index_count = len(vectors_list)

            # Export metadata.json cache
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=4)

            # Hot reload live CCTV engine if running
            if trigger_reload:
                try:
                    from src.inference.live_cctv import reload_biometric_db
                    reload_biometric_db()
                except Exception:
                    pass

            return index_count, len(metadata_dict)


# Singleton biometric repository
biometric_repo = BiometricRepository()

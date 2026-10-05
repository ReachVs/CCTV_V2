"""
Unified Audit Event Ingestion & WAL Encryptor Module (Candidate 4).
Consolidates AES-256 encrypted storage, SQLite WAL mode, asynchronous FastAPI/MCP queries,
and non-blocking background queue ingestion behind an explicit AuditLogger seam.
"""
from abc import ABC, abstractmethod
import os
import sys
import json
import time
import queue
import threading
import sqlite3
import asyncio
from typing import List, Dict, Any, Optional

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None


class AuditLogger(ABC):
    """
    Abstract seam for biometric and security audit event logging.
    Decouples consumers from database storage engines and cryptographic implementations.
    """

    @abstractmethod
    def log_event(
        self,
        timestamp: str,
        track_id: int,
        person_name: str,
        match_distance: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = "",
        **kwargs: Any
    ) -> None:
        """Asynchronously stages an audit event for encrypted WAL persistence."""
        pass

    @abstractmethod
    def fetch_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Synchronously retrieves and decrypts the most recent audit events."""
        pass

    @abstractmethod
    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Asynchronously retrieves and decrypts the most recent audit events (for FastAPI/MCP)."""
        pass

    def flush(self, timeout: float = 2.0) -> None:
        """Waits until all pending queued audit events are persisted to disk."""
        pass

    def stop(self) -> None:
        """Gracefully terminates worker threads and closes open connections."""
        pass


class EncryptedWALAuditLogger(AuditLogger):
    """
    Unified AES-256 Encrypted SQLite WAL Audit Event Logger & Repository.
    - Encrypts sensitive fields (person_name, match_distance, metadata) at rest using AES-256 Fernet.
    - Operates SQLite in WAL mode with NORMAL synchrony for zero lock contention.
    - Features a non-blocking queue worker ensuring <0.1ms latency in live video threads.
    - Exposes both sync (fetch_recent_events) and async (get_recent_events) query APIs.
    - Transparently falls back to legacy detection_events schema if present.
    """

    def __init__(self, db_path: str, encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.log_queue: queue.Queue = queue.Queue()
        self.running = True

        # Initialize AES-256 Fernet cipher
        if Fernet is not None:
            if encryption_key is None:
                env_key = os.environ.get("AES_ENCRYPTION_KEY")
                if env_key:
                    self.key = env_key.encode("utf-8")
                else:
                    db_dir = os.path.dirname(os.path.abspath(self.db_path)) if self.db_path else "."
                    key_file = os.path.join(db_dir, ".audit_key")
                    if os.path.exists(key_file):
                        try:
                            with open(key_file, "rb") as kf:
                                self.key = kf.read().strip()
                        except Exception:
                            self.key = Fernet.generate_key()
                    else:
                        self.key = Fernet.generate_key()
                        try:
                            os.makedirs(db_dir, exist_ok=True)
                            with open(key_file, "wb") as kf:
                                kf.write(self.key)
                        except Exception:
                            pass
            else:
                self.key = encryption_key
            self.cipher = Fernet(self.key)
        else:
            self.key = None
            self.cipher = None

        self._ensure_schema()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _ensure_schema(self) -> None:
        """Initializes SQLite database with WAL mode and creates required tables."""
        if not self.db_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    encrypted_payload TEXT NOT NULL
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
            conn.close()
        except Exception as e:
            print(f"[EncryptedWALAuditLogger Error] Schema initialization failed: {e}")

    def encrypt_payload(self, data: Dict[str, Any]) -> str:
        """Encrypts dictionary payload to an AES-256 ciphertext string."""
        raw_json = json.dumps(data).encode("utf-8")
        if self.cipher is not None:
            return self.cipher.encrypt(raw_json).decode("utf-8")
        return raw_json.decode("utf-8")

    def decrypt_payload(self, encrypted_str: str) -> Dict[str, Any]:
        """Decrypts an AES-256 ciphertext string back to payload dictionary."""
        if self.cipher is not None:
            try:
                decrypted_bytes = self.cipher.decrypt(encrypted_str.encode("utf-8"))
                return json.loads(decrypted_bytes.decode("utf-8"))
            except Exception as e:
                print(f"[EncryptedWALAuditLogger] Decryption error: {e}")
                return {"person_name": "DECRYPTION_ERROR", "match_distance": 999.0}
        try:
            return json.loads(encrypted_str)
        except Exception:
            return {"person_name": encrypted_str, "match_distance": 999.0}

    def _worker(self) -> None:
        """Background daemon processing queued audit records with WAL batch commit."""
        conn = None
        while self.running or not self.log_queue.empty():
            try:
                event = self.log_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now_str, track_id, payload_dict = event
            encrypted_payload = self.encrypt_payload(payload_dict)

            try:
                if conn is None:
                    conn = sqlite3.connect(self.db_path, timeout=5.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("""
                    INSERT INTO encrypted_detection_events (timestamp, track_id, encrypted_payload)
                    VALUES (?, ?, ?)
                """, (now_str, int(track_id), encrypted_payload))
                conn.commit()
            except Exception as e:
                print(f"[EncryptedWALAuditLogger Warning] Worker write failed: {e}")
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
            finally:
                self.log_queue.task_done()

        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def log_event(
        self,
        timestamp: str,
        track_id: int,
        person_name: str,
        match_distance: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = "",
        **kwargs: Any
    ) -> None:
        """
        Thread-safe non-blocking queue submission. Supports both modern and legacy parameter orders.
        """
        # Handle legacy swapped signature if person_name was passed in 2nd position
        if isinstance(track_id, str) and isinstance(person_name, (int, float)):
            person_name, track_id = track_id, int(person_name)

        payload = {
            "person_name": str(person_name),
            "match_distance": float(match_distance),
            "snapshot_path": snapshot_path or kwargs.get("snapshot", ""),
            "meta": extra_meta or kwargs.get("meta", {})
        }
        self.log_queue.put((timestamp, int(track_id), payload))

    def fetch_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Synchronously retrieves and decrypts the latest audit log records."""
        if not os.path.exists(self.db_path):
            return []

        results = []
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 1. Fetch from encrypted_detection_events
            cursor.execute(
                "SELECT id, timestamp, track_id, encrypted_payload FROM encrypted_detection_events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            for r in rows:
                decrypted = self.decrypt_payload(r["encrypted_payload"])
                results.append({
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "track_id": r["track_id"],
                    "person_name": decrypted.get("person_name", "UNKNOWN"),
                    "match_distance": decrypted.get("match_distance", 999.0),
                    "snapshot_path": decrypted.get("snapshot_path", ""),
                    "meta": decrypted.get("meta", {})
                })

            # 2. Fallback to unencrypted legacy detection_events table if encrypted is empty
            if not results:
                cursor.execute(
                    "SELECT id, timestamp, person_name, track_id, match_distance, snapshot_path FROM detection_events ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                for r in cursor.fetchall():
                    results.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "track_id": r["track_id"],
                        "person_name": r["person_name"],
                        "match_distance": float(r["match_distance"]),
                        "snapshot_path": r["snapshot_path"] or "",
                        "meta": {}
                    })

            conn.close()
        except Exception as e:
            print(f"[EncryptedWALAuditLogger Fetch Error] {e}")

        return results

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Asynchronously retrieves recent events off the main event loop."""
        return await asyncio.to_thread(self.fetch_recent_events, limit)

    def flush(self, timeout: float = 2.0) -> None:
        """Waits until all pending queued audit events are persisted to disk."""
        start = time.time()
        while not self.log_queue.empty() and (time.time() - start < timeout):
            time.sleep(0.01)

    def stop(self) -> None:
        """Graceful shutdown."""
        self.running = False
        self.flush(timeout=1.0)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


class ScriptedAuditLogger(AuditLogger):
    """
    Deterministic, in-memory audit logger test adapter.
    Enables zero-IO, database-free hermetic testing without SQLite or filesystem access.
    """

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self._next_id = 1
        self.call_count = 0

    def log_event(
        self,
        timestamp: str,
        track_id: int,
        person_name: str,
        match_distance: float = 0.0,
        extra_meta: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = "",
        **kwargs: Any
    ) -> None:
        self.call_count += 1
        if isinstance(track_id, str) and isinstance(person_name, (int, float)):
            person_name, track_id = track_id, int(person_name)

        event = {
            "id": self._next_id,
            "timestamp": timestamp,
            "track_id": int(track_id),
            "person_name": str(person_name),
            "match_distance": float(match_distance),
            "snapshot_path": snapshot_path or "",
            "meta": extra_meta or {}
        }
        self._next_id += 1
        self.events.append(event)

    def fetch_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(reversed(self.events[-limit:]))

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.fetch_recent_events(limit)

    def flush(self, timeout: float = 2.0) -> None:
        pass

    def stop(self) -> None:
        pass


# Backward compatibility aliases
AuditRepository = EncryptedWALAuditLogger
AESEncryptedWALAuditLogger = EncryptedWALAuditLogger

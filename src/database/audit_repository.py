"""
Audit Repository for asynchronous and synchronous SQLite (WAL Mode) audit event operations.
Encapsulates all database connection handling, row mapping, and query logic.
"""
import os
import sqlite3
from typing import List, Dict, Any, Optional

try:
    import aiosqlite
except ImportError:
    aiosqlite = None


class AuditRepository:
    """
    Data Access Layer for CCTV Biometric Security Audit Events.
    Standardizes WAL mode journal execution and provides async/sync operations.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Synchronously initializes schema if database does not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
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
        finally:
            conn.close()

    async def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Asynchronously retrieves recent biometric identification events using aiosqlite,
        with graceful fallback to synchronous sqlite3.
        """
        if not os.path.exists(self.db_path):
            return []

        try:
            from src.utils.privacy_compliance import AESEncryptedWALAuditLogger
            logger = AESEncryptedWALAuditLogger(db_path=self.db_path)
            events = logger.fetch_recent_events(limit=limit)
            if events:
                return events
        except Exception:
            pass

        if aiosqlite is not None:
            try:
                async with aiosqlite.connect(self.db_path) as conn:
                    conn.row_factory = aiosqlite.Row
                    await conn.execute("PRAGMA journal_mode=WAL;")
                    async with conn.execute(
                        "SELECT id, timestamp, person_name, track_id, match_distance FROM detection_events ORDER BY id DESC LIMIT ?",
                        (limit,)
                    ) as cursor:
                        rows = await cursor.fetchall()
                        return [dict(row) for row in rows]
            except Exception:
                pass

        # Synchronous fallback
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, timestamp, person_name, track_id, match_distance FROM detection_events ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    async def log_event(
        self,
        timestamp: str,
        person_name: str,
        track_id: int,
        match_distance: float,
        snapshot_path: Optional[str] = None
    ) -> None:
        """Asynchronously writes a detection event to the audit log."""
        if aiosqlite is not None:
            try:
                async with aiosqlite.connect(self.db_path) as conn:
                    await conn.execute("PRAGMA journal_mode=WAL;")
                    await conn.execute(
                        "INSERT INTO detection_events (timestamp, person_name, track_id, match_distance, snapshot_path) VALUES (?, ?, ?, ?, ?)",
                        (timestamp, person_name, track_id, match_distance, snapshot_path)
                    )
                    await conn.commit()
                    return
            except Exception:
                pass

        # Synchronous fallback
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO detection_events (timestamp, person_name, track_id, match_distance, snapshot_path) VALUES (?, ?, ?, ?, ?)",
                (timestamp, person_name, track_id, match_distance, snapshot_path)
            )
            conn.commit()
        finally:
            conn.close()

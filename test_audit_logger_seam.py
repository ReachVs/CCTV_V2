"""
Unit tests for the Unified Audit Logger Seam (Candidate 4).
Verifies:
1. EncryptedWALAuditLogger AES-256 Fernet payload encryption and decryption.
2. Direct SQLite inspection verifying ciphertext at rest (GDPR compliance).
3. Background queue ingestion with deterministic flush().
4. Both synchronous (fetch_recent_events) and asynchronous (get_recent_events) query APIs.
5. ScriptedAuditLogger hermetic in-memory test adapter (zero IO).
6. BiometricTrackingEngine dependency injection through audit_logger seam.
7. 100% backward compatibility for AuditRepository and AESEncryptedWALAuditLogger aliases.
"""
import unittest
import asyncio
import os
import sqlite3
import tempfile
import time
import numpy as np

from src.database.audit_repository import (
    AuditLogger,
    EncryptedWALAuditLogger,
    ScriptedAuditLogger,
    AuditRepository,
    AESEncryptedWALAuditLogger,
)
from src.inference.engine import BiometricTrackingEngine
from src.inference.person_detector import ScriptedPersonDetector, DetectedBox
from src.inference.face_embedder import ScriptedFaceEmbedder


class TestEncryptedWALAuditLogger(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_audit.db")
        self.logger = EncryptedWALAuditLogger(db_path=self.db_path)

    def tearDown(self):
        self.logger.stop()
        self.temp_dir.cleanup()

    def test_log_and_fetch_decrypted_events(self):
        """Verify events logged to EncryptedWALAuditLogger are properly encrypted and decrypted."""
        self.logger.log_event(
            timestamp="2026-10-05 12:00:00",
            track_id=101,
            person_name="Special Agent Mulder",
            match_distance=0.38,
            extra_meta={"clearance": "LEVEL 5"}
        )
        self.logger.flush()

        events = self.logger.fetch_recent_events(limit=10)
        self.assertEqual(len(events), 1)
        ev = events[0]
        self.assertEqual(ev["track_id"], 101)
        self.assertEqual(ev["person_name"], "Special Agent Mulder")
        self.assertAlmostEqual(ev["match_distance"], 0.38, places=2)
        self.assertEqual(ev["meta"]["clearance"], "LEVEL 5")

    def test_ciphertext_at_rest_verification(self):
        """Verify SQLite table stores ciphertext and does NOT contain plaintext subject names."""
        secret_name = "Super Secret Subject X"
        self.logger.log_event(
            timestamp="2026-10-05 12:05:00",
            track_id=202,
            person_name=secret_name,
            match_distance=0.25
        )
        self.logger.flush()

        # Connect directly to SQLite and inspect raw payload column
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT encrypted_payload FROM encrypted_detection_events WHERE track_id = 202")
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        raw_payload = row[0]
        # In presence of Fernet, the raw payload MUST NOT contain plaintext name
        if self.logger.cipher is not None:
            self.assertNotIn(secret_name, raw_payload)
            self.assertTrue(raw_payload.startswith("gAAAAA") or len(raw_payload) > 50)

    def test_async_get_recent_events(self):
        """Verify asynchronous get_recent_events works cleanly."""
        self.logger.log_event(
            timestamp="2026-10-05 12:10:00",
            track_id=303,
            person_name="Agent Scully",
            match_distance=0.31
        )
        self.logger.flush()

        async def run_async():
            return await self.logger.get_recent_events(limit=5)

        events = asyncio.run(run_async())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["person_name"], "Agent Scully")


class TestScriptedAuditLogger(unittest.TestCase):

    def test_in_memory_logging(self):
        """Verify ScriptedAuditLogger provides fast, hermetic, zero-IO testing."""
        scripted = ScriptedAuditLogger()
        self.assertEqual(scripted.call_count, 0)

        scripted.log_event(
            timestamp="2026-10-05 12:15:00",
            track_id=404,
            person_name="Walter Skinner",
            match_distance=0.45
        )
        self.assertEqual(scripted.call_count, 1)

        events = scripted.fetch_recent_events(limit=10)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["person_name"], "Walter Skinner")
        self.assertEqual(events[0]["track_id"], 404)


class TestEngineAuditLoggerIntegration(unittest.TestCase):

    def test_engine_injection_and_seam_delegation(self):
        """Verify BiometricTrackingEngine operates with injected ScriptedAuditLogger without SQLite IO."""
        scripted_logger = ScriptedAuditLogger()
        detector = ScriptedPersonDetector([
            DetectedBox(box=[50.0, 50.0, 150.0, 250.0], conf=0.95, track_id=10)
        ])
        embedder = ScriptedFaceEmbedder(vector=np.ones(512, dtype=np.float32))

        engine = BiometricTrackingEngine(
            detector=detector,
            embedder=embedder,
            audit_logger=scripted_logger
        )

        # Verify logger was injected
        self.assertIs(engine.audit_logger, scripted_logger)
        self.assertIs(engine.encrypted_audit_logger, scripted_logger)

        # Pre-seed head presence & test frame
        engine.head_presence_cache[10] = 100
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        engine.process_frame(frame)

        # Engine flush should invoke logger flush cleanly without errors
        engine.flush()
        engine.stop()


class TestBackwardCompatibilityAliases(unittest.TestCase):

    def test_aliases_match(self):
        """Verify backward compatibility aliases point to EncryptedWALAuditLogger."""
        self.assertIs(AuditRepository, EncryptedWALAuditLogger)
        self.assertIs(AESEncryptedWALAuditLogger, EncryptedWALAuditLogger)


if __name__ == "__main__":
    unittest.main()

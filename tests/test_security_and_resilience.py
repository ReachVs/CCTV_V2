"""
Security, Boundary Validation, and Resilience Test Suite.
Verifies SQL injection prevention, Directory Traversal defense, FAISS vector poisoning guards,
and concurrent thread-safety across the Biometric Repository.
"""
import os
import sys
import unittest
import tempfile
import threading
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.biometric_repository import BiometricRepository
from src.api.cctv_server import get_safe_subject_dir, KNOWN_FACES_DIR


class TestSecurityAndResilience(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_faces.db")
        self.index_path = os.path.join(self.temp_dir.name, "test_faces.index")
        self.metadata_path = os.path.join(self.temp_dir.name, "test_metadata.json")
        self.repo = BiometricRepository(
            db_path=self.db_path,
            index_path=self.index_path,
            metadata_path=self.metadata_path
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sql_injection_defense(self):
        """Verify repository safely handles malicious SQL injection payload names."""
        malicious_names = [
            "Robert'); DROP TABLE people;--",
            "' OR '1'='1",
            "admin' --",
            "test_user'; DELETE FROM person_vectors;--"
        ]
        
        v = np.random.randn(512).astype(np.float32)

        for evil_name in malicious_names:
            person_id = self.repo.add_subject(
                name=evil_name,
                embeddings=[v],
                profile_data={"occupation": "HACKER'; DROP TABLE people;--"},
                trigger_reload=False
            )
            self.assertGreater(person_id, 0)

            # Query back - should match literal string safely
            record = self.repo.get_subject_by_name(evil_name)
            self.assertIsNotNone(record)
            self.assertEqual(record["name"], evil_name)

        # Confirm tables and all enrolled subjects still exist unharmed
        subjects = self.repo.get_all_subjects()
        self.assertEqual(len(subjects), len(malicious_names))

    def test_directory_traversal_defense(self):
        """Verify directory traversal attack payloads are completely blocked."""
        traversal_payloads = [
            "../../etc/passwd",
            "../../../secret",
            "..\\..\\windows\\system32",
            "/absolute/root/path",
            "",
            "   ",
            ".",
            "..",
            "../../data/known_faces"
        ]

        for payload in traversal_payloads:
            result = get_safe_subject_dir(payload)
            if result is not None:
                # If a path was returned, verify it strictly lives inside KNOWN_FACES_DIR
                real_target = os.path.abspath(result)
                real_root = os.path.abspath(KNOWN_FACES_DIR)
                self.assertTrue(
                    real_target.startswith(real_root) and real_target != real_root,
                    f"Path traversal breach detected with payload: {payload} -> {real_target}"
                )

    def test_vector_poisoning_defense(self):
        """Verify invalid, NaN, Inf, and wrong dimension embeddings are rejected before FAISS."""
        # 1. NaN in vector
        nan_vec = np.random.randn(512).astype(np.float32)
        nan_vec[10] = np.nan
        with self.assertRaises(ValueError):
            self.repo.add_subject("Eve", [nan_vec], trigger_reload=False)

        # 2. Inf in vector
        inf_vec = np.random.randn(512).astype(np.float32)
        inf_vec[50] = np.inf
        with self.assertRaises(ValueError):
            self.repo.add_subject("Eve", [inf_vec], trigger_reload=False)

        # 3. Wrong dimension (128 instead of 512)
        wrong_dim = np.random.randn(128).astype(np.float32)
        with self.assertRaises(ValueError):
            self.repo.add_subject("Eve", [wrong_dim], trigger_reload=False)

        # 4. Empty name
        good_vec = np.random.randn(512).astype(np.float32)
        with self.assertRaises(ValueError):
            self.repo.add_subject("", [good_vec], trigger_reload=False)
        with self.assertRaises(ValueError):
            self.repo.add_subject("   ", [good_vec], trigger_reload=False)

    def test_concurrent_thread_safety(self):
        """Verify multi-threaded concurrent enrollment and queries do not deadlock or corrupt state."""
        errors = []

        def worker(idx):
            try:
                v = np.random.randn(512).astype(np.float32)
                self.repo.add_subject(f"User_{idx}", [v], trigger_reload=False)
                _ = self.repo.get_subject_by_name(f"User_{idx}")
                _ = self.repo.get_all_subjects()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Thread errors encountered: {errors}")
        subjects = self.repo.get_all_subjects()
        self.assertEqual(len(subjects), 10)

    def test_audit_key_file_permissions(self):
        """Verify generated .audit_key is created with restricted 0o600 permissions."""
        import stat
        from src.database.audit_repository import EncryptedWALAuditLogger
        test_db = os.path.join(self.temp_dir.name, "perms_test.db")
        logger = EncryptedWALAuditLogger(db_path=test_db)
        key_file = os.path.join(self.temp_dir.name, ".audit_key")
        self.assertTrue(os.path.exists(key_file))
        mode = stat.S_IMODE(os.stat(key_file).st_mode)
        self.assertEqual(mode, 0o600)
        logger.stop()

    def test_security_headers_middleware(self):
        """Verify comprehensive defensive security headers (CSP, X-Frame-Options, etc.)."""
        import asyncio
        from unittest.mock import AsyncMock
        from fastapi import Response
        from src.api.cctv_server import add_security_headers

        mock_req = AsyncMock()
        mock_req.url.path = "/dashboard"

        async def dummy_call_next(req):
            return Response(content="ok")

        resp = asyncio.run(add_security_headers(mock_req, dummy_call_next))
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(resp.headers.get("X-XSS-Protection"), "1; mode=block")
        self.assertEqual(resp.headers.get("Referrer-Policy"), "strict-origin-when-cross-origin")
        self.assertIn("Content-Security-Policy", resp.headers)
        self.assertIn("default-src 'self'", resp.headers["Content-Security-Policy"])

    def test_prefix_collision_defense(self):
        """Verify prefix collisions outside KNOWN_FACES_DIR are rejected."""
        root_base = os.path.basename(KNOWN_FACES_DIR)
        # Attempt to target a sibling directory like known_faces_leak
        payload = f"{root_base}_leak"
        res = get_safe_subject_dir(payload)
        # Result must either be None or live strictly inside KNOWN_FACES_DIR
        if res is not None:
            self.assertEqual(os.path.commonpath([KNOWN_FACES_DIR, res]), KNOWN_FACES_DIR)


if __name__ == "__main__":
    unittest.main()

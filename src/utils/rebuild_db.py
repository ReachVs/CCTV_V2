"""
Biometric Database Rebuild CLI Adapter.
Delegates multi-template indexing and full archive rebuilding to BiometricRepository.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.biometric_repository import biometric_repo


def rebuild_biometric_database(known_faces_dir=None):
    """Rebuilds the entire biometric database (SQLite, FAISS, and metadata.json) from archive."""
    print("=" * 65)
    print("  BIOMETRIC DATABASE REBUILD & MULTI-TEMPLATE INDEXER")
    print("=" * 65)
    count, meta_count = biometric_repo.rebuild_from_archive(known_faces_dir=known_faces_dir)
    print(f"\n[SUCCESS] Rebuilt {count} templates across metadata registry ({meta_count} entries).")
    print("=" * 65)
    return count, meta_count


if __name__ == "__main__":
    rebuild_biometric_database()

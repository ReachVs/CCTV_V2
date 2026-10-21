"""
Database Auto-Heal CLI Adapter.
Delegates multi-store synchronization and health diagnostics to BiometricRepository.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database.biometric_repository import biometric_repo


def check_database_health():
    """Evaluates database health across FAISS, SQLite, and metadata.json."""
    return biometric_repo.check_health()


def auto_heal_database():
    """Executes automated self-healing across the multi-store biometric database."""
    return biometric_repo.auto_heal()


if __name__ == "__main__":
    is_healthy, reason = check_database_health()
    if is_healthy:
        print(f"[AUTO-HEAL] Database healthy: {reason}")
    else:
        print(f"[AUTO-HEAL] Initiating recovery ({reason})...")
        success = auto_heal_database()
        print(f"[AUTO-HEAL] Recovery result: {'SUCCESS' if success else 'FAILED'}")

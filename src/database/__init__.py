"""
Database & Persistence Package for CCTV Pipeline.
"""
from src.database.audit_repository import (
    AuditLogger,
    EncryptedWALAuditLogger,
    ScriptedAuditLogger,
    AuditRepository,
    AESEncryptedWALAuditLogger,
)
from src.database.vector_store import (
    VectorIndexAdapter,
    FAISSIndexAdapter,
    InMemoryIndexAdapter
)
from src.database.biometric_repository import (
    BiometricRepository,
    biometric_repo
)

__all__ = [
    "AuditLogger",
    "EncryptedWALAuditLogger",
    "ScriptedAuditLogger",
    "AuditRepository",
    "AESEncryptedWALAuditLogger",
    "VectorIndexAdapter",
    "FAISSIndexAdapter",
    "InMemoryIndexAdapter",
    "BiometricRepository",
    "biometric_repo"
]

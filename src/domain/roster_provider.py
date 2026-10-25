"""
Domain Roster Provider Abstraction
===================================
Defines the seam for attendee roster sources.
Decouples daily checklist generation from biometric vector storage,
enabling pluggable university SIS, CSV roster files, SQLite, or LDAP registries.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class RosterProvider(ABC):
    """
    Abstract Seam for attendee roster sources.
    Decouples daily checklist generation from biometric vector storage,
    enabling pluggable university SIS, CSV roster files, or LDAP registries.
    """
    @abstractmethod
    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        """Returns list of enrolled attendee dicts: [{'name': str, 'occupation': str, 'clearance': str, ...}]"""
        pass


class BiometricRepositoryRosterProvider(RosterProvider):
    """Default adapter fetching enrolled subjects from BiometricRepository."""
    def __init__(self, biometric_repo: Any):
        self.biometric_repo = biometric_repo

    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        if self.biometric_repo is not None and hasattr(self.biometric_repo, "get_all_subjects"):
            try:
                return self.biometric_repo.get_all_subjects()
            except Exception as e:
                print(f"[RosterProvider Warning] Could not fetch enrolled subjects: {e}")
        return []


BiometricRepoRosterAdapter = BiometricRepositoryRosterProvider


class InMemoryRosterProvider(RosterProvider):
    """In-memory roster test double for deterministic unit/integration testing."""
    def __init__(self, roster: Optional[List[Dict[str, Any]]] = None):
        self.roster = list(roster) if roster else []

    def get_enrolled_roster(self) -> List[Dict[str, Any]]:
        return list(self.roster)

    def set_roster(self, roster: List[Dict[str, Any]]) -> None:
        self.roster = list(roster)

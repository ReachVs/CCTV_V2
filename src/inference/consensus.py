"""
IdentityConsensus module encapsulating multi-frame biometric voting, inter-subject margin gating,
side-profile yaw guardrails, and active identity claim leases behind a cohesive seam.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

from src.domain.subject_session import SubjectSession, TrackStatus


@dataclass
class MatchCandidate:
    """Biometric vector search match candidate."""
    name: str
    dist1: float
    dist2: float
    norm: float = 1.0
    is_side_profile: bool = False


@dataclass
class ConsensusDecision:
    """The outcome of evaluating a candidate match against track history and active claims."""
    display_name: str
    status: TrackStatus
    is_confirmed: bool
    confidence: float
    telemetry: Dict[str, Any] = field(default_factory=dict)


class IdentityConsensus:
    """
    Cohesive domain module responsible for transitioning TrackedSubjects from SCANNING
    to CONFIRMED identity via multi-frame consensus voting and collision arbitration.
    """
    def __init__(
        self,
        consensus_votes: int = 2,
        history_window: int = 5,
        margin_threshold: float = 0.15,
        side_profile_threshold: float = 0.30,
        max_scanning_attempts: int = 12,
        min_unknown_crops: int = 15
    ):
        self.consensus_votes = consensus_votes
        self.history_window = history_window
        self.margin_threshold = margin_threshold
        self.side_profile_threshold = side_profile_threshold
        self.max_scanning_attempts = max_scanning_attempts
        self.min_unknown_crops = min_unknown_crops

        # track_id -> List[Dict[str, Any]] (records of {"name": str, "is_side": bool})
        self._vote_histories: Dict[int, List[Dict[str, Any]]] = {}
        # identity_name -> track_id
        self.active_identity_claims: Dict[str, int] = {}

    def get_history(self, track_id: int) -> List[Dict[str, Any]]:
        """Returns the current vote history list for a track."""
        return self._vote_histories.setdefault(track_id, [])

    def purge_track(self, track_id: int) -> None:
        """Purges history and releases any active claims held by track_id."""
        self._vote_histories.pop(track_id, None)
        stale_names = [name for name, tid in self.active_identity_claims.items() if tid == track_id]
        for name in stale_names:
            self.active_identity_claims.pop(name, None)

    def cleanup_stale_claims(
        self,
        active_track_ids: Optional[List[int]],
        current_frame: int,
        last_seen_lookup: Callable[[int], int],
        current_track_id: Optional[int] = None
    ) -> None:
        """Removes claims held by tracks that have disappeared from the scene."""
        stale_claims = []
        for name, claimant_tid in list(self.active_identity_claims.items()):
            if current_track_id is not None and claimant_tid == current_track_id:
                continue
            last_seen = last_seen_lookup(claimant_tid)
            is_dead_or_stale = (
                (active_track_ids is not None and claimant_tid not in active_track_ids)
                or (current_frame - last_seen > 15)
            )
            if is_dead_or_stale:
                stale_claims.append(name)
        for name in stale_claims:
            self.active_identity_claims.pop(name, None)

    def evaluate(
        self,
        session: SubjectSession,
        candidate: Optional[MatchCandidate],
        current_frame: int,
        active_track_ids: Optional[List[int]] = None,
        last_seen_lookup: Optional[Callable[[int], int]] = None,
        effective_age: Optional[int] = None
    ) -> ConsensusDecision:
        """
        Evaluates a candidate match against the subject's lifecycle, updates the SubjectSession,
        and returns the consensus decision.
        """
        track_id = session.track_id

        # 1. Clean up stale claims
        if last_seen_lookup is not None:
            self.cleanup_stale_claims(active_track_ids, current_frame, last_seen_lookup, current_track_id=track_id)

        # 2. Confirmed Identity Hysteresis Lock-In
        if session.is_confirmed and session.status == TrackStatus.CONFIRMED:
            self.active_identity_claims[session.identity_name] = track_id
            return ConsensusDecision(
                display_name=session.display_name,
                status=session.status,
                is_confirmed=True,
                confidence=session.confidence,
                telemetry={
                    "bypassed": True,
                    "norm": session.best_norm,
                    "cached": session.identity_name,
                    "confirmed": True
                }
            )

        # 3. Smart bypass for confirmed Unknown Person
        if session.is_confirmed and session.status == TrackStatus.UNKNOWN:
            if candidate is not None and candidate.name not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]:
                # Reopen scanning if a fresh valid recognized candidate arrives
                session.status = TrackStatus.SCANNING
                session.attempts = 0
                session.valid_face_crops = 0
                session.failed_attempts = 0
                self._vote_histories.pop(track_id, None)
            else:
                return ConsensusDecision(
                    display_name="Unknown Person",
                    status=TrackStatus.UNKNOWN,
                    is_confirmed=True,
                    confidence=session.confidence,
                    telemetry={"bypassed": True, "norm": 0.0, "cached": "Unknown Person"}
                )

        # 4. Tentative Track Gating
        age_to_check = effective_age if effective_age is not None else max(3, session.age)
        if age_to_check < 3 and not session.is_confirmed:
            return ConsensusDecision(
                display_name=session.display_name,
                status=TrackStatus.SCANNING,
                is_confirmed=False,
                confidence=session.confidence,
                telemetry={"tentative": True, "age": age_to_check}
            )

        # 5. Non-face crop handling
        if candidate is None:
            session.failed_attempts += 1
            if session.failed_attempts >= self.max_scanning_attempts and not session.is_confirmed:
                session.mark_unverified()
            return ConsensusDecision(
                display_name=session.display_name,
                status=session.status,
                is_confirmed=session.is_confirmed,
                confidence=session.confidence,
                telemetry={"failed": True, "confirmed": session.is_confirmed}
            )

        # 6. Process valid face candidate
        session.valid_face_crops += 1
        session.attempts += 1
        matched_name = candidate.name
        is_side = candidate.is_side_profile
        norm = candidate.norm

        # Identity claim collision arbitration
        if matched_name not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]:
            existing_claimant = self.active_identity_claims.get(matched_name)
            if existing_claimant is not None and existing_claimant != track_id:
                claimant_seen = last_seen_lookup(existing_claimant) if last_seen_lookup else 0
                is_claimant_live = (
                    (active_track_ids is None or existing_claimant in active_track_ids)
                    and (current_frame - claimant_seen <= 15)
                )
                if is_claimant_live:
                    matched_name = "Ambiguous Match"
                else:
                    self.active_identity_claims[matched_name] = track_id
            else:
                self.active_identity_claims[matched_name] = track_id

        # Update vote history
        history = self._vote_histories.setdefault(track_id, [])
        if norm >= session.best_norm * 0.85 and matched_name not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]:
            history.append({"name": matched_name, "is_side": is_side})
            if len(history) > self.history_window:
                history.pop(0)
            if norm > session.best_norm:
                session.best_norm = norm

        # 7. Consensus vote counting
        if len(history) >= self.consensus_votes:
            valid_vote_names = [
                item["name"] for item in history
                if item["name"] not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person"]
            ]
            if valid_vote_names:
                most_common = max(set(valid_vote_names), key=valid_vote_names.count)
                vote_count = valid_vote_names.count(most_common)
                if vote_count >= self.consensus_votes:
                    # Side-profile consensus guardrail:
                    # Confirms if at least one frontal/quarter vote exists, or if side-profile matches are
                    # ultra-high confidence (dist1 <= 0.15) or consistent across >= 3 votes.
                    has_frontal_confirmation = any(not item.get("is_side", False) for item in history)
                    is_confident_side = is_side and (candidate.dist1 <= 0.15 or vote_count >= self.consensus_votes + 1)
                    if has_frontal_confirmation or not is_side or is_confident_side:
                        session.confirm_identity(most_common, distance=candidate.dist1, norm=norm)
                        self.active_identity_claims[most_common] = track_id
                        return ConsensusDecision(
                            display_name=session.display_name,
                            status=session.status,
                            is_confirmed=True,
                            confidence=0.85,
                            telemetry={
                                "dist1": candidate.dist1,
                                "dist2": candidate.dist2,
                                "norm": norm,
                                "confirmed": True,
                                "votes": vote_count
                            }
                        )

        # 8. Check for Unknown Person transition:
        # Categorize as Unknown Person ONLY if unmatched valid crops reach threshold without any match,
        # AND there are NO valid known-identity votes in history!
        is_unmatched_crop = matched_name in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person", ""]
        has_known_votes = any(
            item.get("name") not in ["Unknown", "Ambiguous Match", "Scanning", "Unknown Person", ""]
            for item in history
        )
        if not session.is_confirmed and not has_known_votes and (
            (is_unmatched_crop and session.valid_face_crops >= self.min_unknown_crops)
            or session.attempts >= self.max_scanning_attempts
        ):
            session.mark_unknown()
            return ConsensusDecision(
                display_name=session.display_name,
                status=session.status,
                is_confirmed=True,
                confidence=0.50,
                telemetry={"valid_face_crops": session.valid_face_crops, "confirmed": True}
            )

        # Return scanning state
        return ConsensusDecision(
            display_name=session.display_name,
            status=session.status,
            is_confirmed=False,
            confidence=session.confidence,
            telemetry={
                "dist1": candidate.dist1,
                "dist2": candidate.dist2,
                "norm": norm,
                "attempts": session.attempts
            }
        )

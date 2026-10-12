"""
Deep Module: BiometricMatcher
=============================
Encapsulates vector similarity search, pose-aware distance thresholding,
and top-k inter-subject margin gating behind a cohesive biometric matching seam.
"""
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np

from src.database.vector_store import VectorIndexAdapter
from src.inference.consensus import MatchCandidate


class BiometricMatcher:
    """
    Deep Domain Module: Executes vector similarity search against FAISS/in-memory indexes,
    enforces strict side-profile L2 thresholds, computes nearest distinct-subject margins,
    and produces structured MatchCandidates.
    """

    def __init__(
        self,
        index_adapter: VectorIndexAdapter,
        metadata: Optional[Dict[str, Any]] = None,
        l2_threshold: float = 0.68,
        margin_threshold: float = 0.15,
    ):
        self.index_adapter = index_adapter
        self.metadata = metadata or {}
        self.l2_threshold = l2_threshold
        self.margin_threshold = margin_threshold

    def get_identity(self, idx: int) -> str:
        """Resolves vector index position to subject identity string."""
        if not self.metadata:
            return "Unknown"
        if isinstance(self.metadata, dict):
            val = self.metadata.get(str(idx), self.metadata.get(idx, "Unknown"))
            if isinstance(val, dict):
                return val.get("name", "Unknown")
            return str(val) if val is not None else "Unknown"
        elif isinstance(self.metadata, list) and 0 <= idx < len(self.metadata):
            val = self.metadata[idx]
            return val.get("name", "Unknown") if isinstance(val, dict) else str(val)
        return "Unknown"

    def match(
        self,
        embedding: Optional[np.ndarray],
        face_center_ratio: float = 0.5,
        is_side_profile: bool = False
    ) -> MatchCandidate:
        """
        Executes vector search, evaluates pose-aware thresholding, and returns a MatchCandidate.
        """
        if embedding is None:
            return MatchCandidate(
                name="Unknown", dist1=999.0, dist2=999.0, norm=0.0, is_side_profile=is_side_profile
            )

        emb = np.array(embedding, dtype=np.float32).flatten()
        if emb.size != 512:
            return MatchCandidate(
                name="Unknown", dist1=999.0, dist2=999.0, norm=0.0, is_side_profile=is_side_profile
            )

        if face_center_ratio < 0.15 or face_center_ratio > 0.85:
            return MatchCandidate(
                name="Ambiguous Match", dist1=999.0, dist2=999.0, norm=0.0, is_side_profile=is_side_profile
            )

        quality_norm = float(np.linalg.norm(emb))
        if quality_norm < 0.3:
            return MatchCandidate(
                name="Unknown", dist1=999.0, dist2=999.0, norm=quality_norm, is_side_profile=is_side_profile
            )

        if self.index_adapter is None or self.index_adapter.ntotal == 0:
            return MatchCandidate(
                name="Unknown", dist1=999.0, dist2=999.0, norm=quality_norm, is_side_profile=is_side_profile
            )

        norm_emb = emb / (quality_norm + 1e-10)
        query_vector = np.expand_dims(norm_emb, axis=0)

        k = min(10, self.index_adapter.ntotal)
        distances, indices = self.index_adapter.search(query_vector, k=k)

        if indices is None or len(indices) == 0 or len(indices[0]) == 0 or indices[0][0] < 0:
            return MatchCandidate(
                name="Unknown",
                dist1=999.0,
                dist2=999.0,
                norm=quality_norm,
                is_side_profile=is_side_profile
            )

        dist1 = float(distances[0][0])
        idx1 = int(indices[0][0])
        name1 = self.get_identity(idx1)

        # Pose-aware strict threshold: side profile crops require tighter distance (<0.30 vs default 0.68)
        effective_threshold = min(self.l2_threshold, 0.30) if is_side_profile else self.l2_threshold
        if dist1 > effective_threshold or name1 in ["Unknown", ""]:
            return MatchCandidate(
                name="Unknown",
                dist1=dist1,
                dist2=999.0,
                norm=quality_norm,
                is_side_profile=is_side_profile
            )

        # Find first nearest neighbor belonging to a DIFFERENT subject (Inter-Subject Margin Gating)
        dist_diff_subject = 999.0
        diff_subject_found = False

        for j in range(1, distances.shape[1]):
            idx_j = int(indices[0][j])
            if idx_j < 0:
                continue
            name_j = self.get_identity(idx_j)
            if name_j not in ["Unknown", ""] and name_j.lower() != name1.lower():
                dist_diff_subject = float(distances[0][j])
                diff_subject_found = True
                break

        effective_margin = max(self.margin_threshold, 0.20) if is_side_profile else self.margin_threshold
        if diff_subject_found:
            margin = abs(dist_diff_subject - dist1)
            if margin < effective_margin:
                return MatchCandidate(
                    name="Ambiguous Match",
                    dist1=dist1,
                    dist2=dist_diff_subject,
                    norm=quality_norm,
                    is_side_profile=is_side_profile
                )
            dist2 = dist_diff_subject
        else:
            dist2 = 999.0

        return MatchCandidate(
            name=name1,
            dist1=dist1,
            dist2=dist2,
            norm=quality_norm,
            is_side_profile=is_side_profile
        )

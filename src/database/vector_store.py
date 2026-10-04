"""
Vector Index Store & Adapters for FAISS and Hermetic In-Memory Search.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class VectorIndexAdapter(ABC):
    """Seam abstraction for vector similarity search."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (distances, indices) as numpy arrays with shape (1, k)."""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Adds normalized vector embeddings to the index."""
        pass

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total vectors registered in the index."""
        pass


class FAISSIndexAdapter(VectorIndexAdapter):
    """Production vector search adapter wrapping FAISS L2/HNSW."""

    def __init__(self, index_obj: Any = None, dimension: int = 512):
        if index_obj is not None:
            self._index = index_obj
        elif faiss is not None:
            self._index = faiss.IndexFlatL2(dimension)
        else:
            self._index = None

    def search(self, query_vector: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        if self._index is None or getattr(self._index, 'ntotal', 0) == 0:
            return np.array([[999.0] * k], dtype=np.float32), np.array([[-1] * k], dtype=np.int64)
        q = np.ascontiguousarray(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        return self._index.search(q, k)

    def add(self, vectors: np.ndarray) -> None:
        if self._index is not None:
            v = np.ascontiguousarray(vectors, dtype=np.float32)
            if v.ndim == 1:
                v = np.expand_dims(v, axis=0)
            self._index.add(v)

    @property
    def ntotal(self) -> int:
        return int(getattr(self._index, 'ntotal', 0)) if self._index is not None else 0

    @property
    def raw_index(self) -> Any:
        return self._index


class InMemoryIndexAdapter(VectorIndexAdapter):
    """Hermetic in-memory test adapter avoiding external C++ libraries."""

    def __init__(self, vectors: Optional[np.ndarray] = None, dimension: int = 512):
        self._dim = dimension
        if vectors is not None and len(vectors) > 0:
            v = np.array(vectors, dtype=np.float32)
            if v.ndim == 1:
                v = np.expand_dims(v, axis=0)
            self._vectors = v
        else:
            self._vectors = np.empty((0, dimension), dtype=np.float32)

    def search(self, query_vector: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        if len(self._vectors) == 0:
            return np.array([[999.0] * k], dtype=np.float32), np.array([[-1] * k], dtype=np.int64)
        
        q = np.array(query_vector, dtype=np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        results = []
        for idx, v in enumerate(self._vectors):
            v_flat = v.flatten().astype(np.float32)
            v_norm = np.linalg.norm(v_flat)
            if v_norm > 0:
                v_flat = v_flat / v_norm
            dist_sq = float(np.sum((v_flat - q) ** 2))
            results.append((dist_sq, idx))

        results.sort(key=lambda x: x[0])
        k_val = min(k, len(results))
        distances = np.array([[r[0] for r in results[:k_val]]], dtype=np.float32)
        indices = np.array([[r[1] for r in results[:k_val]]], dtype=np.int64)

        if k_val < k:
            pad_d = np.full((1, k - k_val), 999.0, dtype=np.float32)
            pad_i = np.full((1, k - k_val), -1, dtype=np.int64)
            distances = np.hstack([distances, pad_d])
            indices = np.hstack([indices, pad_i])

        return distances, indices

    def add(self, vectors: np.ndarray) -> None:
        v = np.array(vectors, dtype=np.float32)
        if v.ndim == 1:
            v = np.expand_dims(v, axis=0)
        if len(self._vectors) == 0:
            self._vectors = v
        else:
            self._vectors = np.vstack([self._vectors, v])

    @property
    def ntotal(self) -> int:
        return len(self._vectors)

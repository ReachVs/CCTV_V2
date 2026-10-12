"""
Deep Module: EmbeddingBatchWorker
=================================
Asynchronously batches face crops, executes GPU/CPU face alignment, upscaling,
and vector embedding, and invokes the engine's embedding result callback.
"""
from typing import Callable, Dict, Any, List, Optional, Tuple
import threading
import time
import numpy as np

from src.inference.face_embedder import FaceEmbedder
from src.utils.privacy_compliance import GDPRMemorySanitizer


class EmbeddingBatchWorker:
    """
    Encapsulates queue management, thread lifecycle, and batch inference
    for face embeddings without polluting the tracking engine domain.
    """

    def __init__(
        self,
        embedder_getter: Callable[[], Optional[FaceEmbedder]],
        on_result: Callable[[int, Optional[np.ndarray], Any, bool], None],
        max_queue_size: int = 16,
    ):
        self._embedder_getter = embedder_getter
        self.on_result = on_result
        self.max_queue_size = max_queue_size
        self._queue: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    @property
    def queue(self) -> List[Dict[str, Any]]:
        return self._queue

    @queue.setter
    def queue(self, val: List[Dict[str, Any]]) -> None:
        self._queue = val

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def submit(self, track_id: int, crop: np.ndarray, frame_num: int) -> None:
        """Submits a crop for embedding, replacing any older pending crop for this track."""
        with self._lock:
            self._queue = [item for item in self._queue if item["track_id"] != track_id]
            if len(self._queue) >= self.max_queue_size:
                self._queue.pop()
            self._queue.insert(0, {
                "track_id": track_id,
                "crop": crop,
                "frame_added": frame_num
            })

    def _worker_loop(self) -> None:
        while self._running:
            self.flush()
            time.sleep(0.005)

    def flush(self) -> None:
        """Processes any queued embedding crops synchronously (for tests and shutdown)."""
        with self._lock:
            if not self._queue:
                return
            batch_to_process = list(self._queue)
            self._queue.clear()

        embedder = self._embedder_getter()
        if embedder is None:
            return

        crops_to_sanitize = []
        embeddings: List[Optional[np.ndarray]] = []
        ratios: List[Optional[Tuple[float, float, float, float]]] = []
        is_sides: List[bool] = []

        for item in batch_to_process:
            crop = item["crop"]
            crops_to_sanitize.append(crop)
            is_side = False
            if hasattr(embedder, "is_side_profile"):
                try:
                    is_side = embedder.is_side_profile(crop)
                except Exception:
                    is_side = False
            emb = embedder.embed(crop, align=True, strict_alignment=True, upscale=True)
            embeddings.append(emb)
            is_sides.append(is_side)
            ratios.append((0.1, 0.1, 0.8, 0.8) if emb is not None else None)

        for i, item in enumerate(batch_to_process):
            self.on_result(item["track_id"], embeddings[i], ratios[i], is_side_profile=is_sides[i])

        GDPRMemorySanitizer.sanitize_crop_list(crops_to_sanitize)

    def stop(self) -> None:
        """Gracefully terminates worker thread."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

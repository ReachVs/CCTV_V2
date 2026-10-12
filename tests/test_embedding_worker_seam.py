"""
Unit tests for the deep EmbeddingBatchWorker module.
Validates crop queue deduplication, thread lifecycle, and batch flushing.
"""
import unittest
import numpy as np

from src.inference.embedding_worker import EmbeddingBatchWorker
from src.inference.face_embedder import ScriptedFaceEmbedder


class TestEmbeddingWorkerSeam(unittest.TestCase):

    def setUp(self):
        self.embedder = ScriptedFaceEmbedder(vector=np.ones(512, dtype=np.float32))
        self.results = []
        self.worker = EmbeddingBatchWorker(
            embedder_getter=lambda: self.embedder,
            on_result=self._on_result,
            max_queue_size=4
        )

    def tearDown(self):
        self.worker.stop()

    def _on_result(self, track_id, emb, ratio, is_side_profile):
        self.results.append({
            "track_id": track_id,
            "emb": emb,
            "ratio": ratio,
            "is_side_profile": is_side_profile
        })

    def test_submit_deduplication(self):
        dummy_crop1 = np.zeros((100, 100, 3), dtype=np.uint8)
        dummy_crop2 = np.ones((100, 100, 3), dtype=np.uint8)

        # Submit twice for track 1
        self.worker.submit(1, dummy_crop1, frame_num=10)
        self.worker.submit(1, dummy_crop2, frame_num=11)

        # There should only be 1 item in the queue for track 1 with frame_added=11
        with self.worker.lock:
            self.assertEqual(len(self.worker.queue), 1)
            self.assertEqual(self.worker.queue[0]["frame_added"], 11)

    def test_flush_processes_batch(self):
        dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
        self.worker.submit(1, dummy_crop, frame_num=1)
        self.worker.submit(2, dummy_crop, frame_num=1)

        self.worker.flush()
        # Verify both tracks received callbacks
        received_tids = {r["track_id"] for r in self.results}
        self.assertIn(1, received_tids)
        self.assertIn(2, received_tids)


if __name__ == "__main__":
    unittest.main()

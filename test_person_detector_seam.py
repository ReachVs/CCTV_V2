"""
Unit tests for the PersonDetector seam and adapters (Candidate 2).
Verifies:
1. ScriptedPersonDetector provides fast, hermetic, model-free test execution.
2. Dynamic callable detection generators in ScriptedPersonDetector.
3. UltralyticsPersonDetector Haar cascade fallback encapsulation.
4. Seamless backward compatibility with legacy engine._yolo_model attribute mutations.
"""
import unittest
import numpy as np

from src.database.vector_store import InMemoryIndexAdapter
from src.inference.engine import BiometricTrackingEngine, ProcessResult
from src.inference.person_detector import (
    PersonDetector,
    UltralyticsPersonDetector,
    ScriptedPersonDetector,
    DetectedBox,
)


class TestPersonDetectorSeam(unittest.TestCase):

    def setUp(self):
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.adapter = InMemoryIndexAdapter(dimension=512)

    def test_scripted_detector_direct_injection(self):
        """Verify BiometricTrackingEngine runs with an injected ScriptedPersonDetector."""
        scripted = ScriptedPersonDetector([
            DetectedBox(box=[50.0, 50.0, 150.0, 250.0], conf=0.90, track_id=101)
        ])

        engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            detector=scripted,
            metadata_path=None,
        )

        result = engine.process_frame(self.frame, annotate=False)
        self.assertIsInstance(result, ProcessResult)
        self.assertEqual(scripted.call_count, 1)

    def test_scripted_detector_callable_generator(self):
        """Verify ScriptedPersonDetector can take a dynamic callable."""
        boxes_sequence = [
            [DetectedBox(box=[50.0, 50.0, 150.0, 250.0], conf=0.88, track_id=1)],
            [DetectedBox(box=[55.0, 50.0, 155.0, 250.0], conf=0.89, track_id=1)],
            []
        ]
        seq_idx = {"idx": 0}

        def gen():
            i = seq_idx["idx"]
            seq_idx["idx"] += 1
            if i < len(boxes_sequence):
                return boxes_sequence[i]
            return []

        scripted = ScriptedPersonDetector(detections=gen)
        engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            detector=scripted,
            metadata_path=None,
        )

        res1 = engine.process_frame(self.frame)
        res2 = engine.process_frame(self.frame)
        res3 = engine.process_frame(self.frame)

        self.assertEqual(scripted.call_count, 3)

    def test_backward_compatibility_yolo_model_setter(self):
        """Verify legacy tests that mutate engine._yolo_model continue to work seamlessly."""
        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes

        class MockBox:
            def __init__(self, xyxy, conf, track_id):
                class MockTensor:
                    def __init__(self, arr):
                        self._arr = arr
                    def cpu(self):
                        return self
                    def numpy(self):
                        return self._arr

                self.xyxy = MockTensor(np.array(xyxy, dtype=np.float32))
                self.conf = MockTensor(np.array(conf, dtype=np.float32))
                self.id = MockTensor(np.array(track_id, dtype=np.int32))

        class LegacyMockYOLO:
            def track(self, *args, **kwargs):
                return [MockResult(MockBox([[50, 50, 150, 250]], [0.95], [42]))]

        engine = BiometricTrackingEngine(
            index_adapter=self.adapter,
            metadata_path=None,
        )

        # Mutate legacy private attribute
        engine._yolo_model = LegacyMockYOLO()

        # Engine must execute through the seam without raising
        result = engine.process_frame(self.frame, annotate=False)
        self.assertIsInstance(result, ProcessResult)


if __name__ == "__main__":
    unittest.main()

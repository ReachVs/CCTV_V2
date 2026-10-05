import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import numpy as np

from src.rendering.annotator import (
    Annotator,
    CyberpunkHUDAnnotator,
    FrameAnnotator,
    NoOpAnnotator,
    ScriptedAnnotator
)
from src.domain.subject_session import SubjectSession, TrackStatus
from src.inference.engine import (
    BiometricTrackingEngine,
    InMemoryIndexAdapter
)
from src.inference.person_detector import ScriptedPersonDetector, DetectedBox
from src.inference.face_embedder import ScriptedFaceEmbedder


class TestHUDAnnotatorSeam(unittest.TestCase):
    def setUp(self):
        self.dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_noop_annotator_passes_frame_untouched(self):
        """NoOpAnnotator must return the exact same frame without mutating or copying."""
        annotator = NoOpAnnotator()
        subjects = [
            SubjectSession(track_id=1, first_seen=1, last_seen=1, box=(50, 50, 150, 150)).to_tracked_subject()
        ]
        res = annotator.annotate(self.dummy_frame, subjects, fps=30.0)
        self.assertIs(res, self.dummy_frame)
        self.assertEqual(int(np.sum(res)), 0)

    def test_scripted_annotator_records_calls(self):
        """ScriptedAnnotator must capture calls and subjects for headless test verification."""
        scripted = ScriptedAnnotator()
        subjects = [
            SubjectSession(track_id=42, first_seen=1, last_seen=1, box=(20, 20, 100, 100)).to_tracked_subject()
        ]
        res = scripted.annotate(self.dummy_frame, subjects, fps=28.5)
        self.assertEqual(len(scripted.calls), 1)
        self.assertEqual(scripted.calls[0], (1, 360, 28.5))
        self.assertEqual(len(scripted.last_subjects), 1)
        self.assertEqual(scripted.last_subjects[0].track_id, 42)

    def test_cyberpunk_hud_annotator_draws_visuals(self):
        """CyberpunkHUDAnnotator must draw corner brackets, telemetry bar, and status banners."""
        annotator = CyberpunkHUDAnnotator(show_telemetry_bar=True)
        session = SubjectSession(track_id=7, first_seen=1, last_seen=1, box=(40, 40, 140, 140))
        session.confirm_identity("Ceaser", 0.1, 1.0)
        subjects = [session.to_tracked_subject()]

        frame = self.dummy_frame.copy()
        res = annotator.annotate(frame, subjects, fps=30.0)
        self.assertIsNotNone(res)
        # Verify that pixels were drawn (sum > 0)
        self.assertGreater(int(np.sum(res)), 0)

    def test_engine_injection_with_scripted_annotator(self):
        """BiometricTrackingEngine must accept an injected Annotator and delegate annotation to it."""
        scripted_annotator = ScriptedAnnotator()
        scripted_detector = ScriptedPersonDetector([
            DetectedBox(box=(40, 40, 140, 140), conf=0.90, track_id=9)
        ])
        dummy_adapter = InMemoryIndexAdapter(np.zeros((1, 512), dtype=np.float32))

        engine = BiometricTrackingEngine(
            index_adapter=dummy_adapter,
            detector=scripted_detector,
            embedder=ScriptedFaceEmbedder(np.zeros(512, dtype=np.float32)),
            annotator=scripted_annotator
        )

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = engine.process_frame(frame, annotate=True)

        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(scripted_annotator.calls), 1)
        self.assertEqual(scripted_annotator.calls[-1][1], 360)


if __name__ == "__main__":
    unittest.main()

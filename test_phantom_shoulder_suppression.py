import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference.engine import (
    BiometricTrackingEngine,
    compute_box_iou,
    compute_box_containment,
    InMemoryIndexAdapter
)

def test_box_containment_math():
    # Box 8: Full person body & face
    box_person = (115, 135, 685, 675)
    # Box 9: Bare shoulder + chair
    box_phantom_shoulder = (550, 350, 800, 675)

    iou = compute_box_iou(box_person, box_phantom_shoulder)
    containment = compute_box_containment(box_person, box_phantom_shoulder)
    
    print(f"Computed IoU: {iou:.4f}, Containment: {containment:.4f}")
    assert iou < 0.35, f"Expected IoU < 0.35, got {iou}"
    assert containment > 0.45, f"Expected Containment > 0.45, got {containment}"
    print("[PASS] Containment correctly detects the nested shoulder box where IoU failed.")

def test_engine_phantom_suppression():
    # Initialize engine with in-memory vector index
    adapter = InMemoryIndexAdapter(dimension=512)
    meta = {0: {"name": "New Test Person", "role": "User"}}
    # Dummy embedding for "New Test Person"
    dummy_vec = np.ones((1, 512), dtype=np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    adapter.add(dummy_vec)

    engine = BiometricTrackingEngine(
        index_adapter=adapter,
        metadata_path=None,
        l2_threshold=0.68,
        margin_threshold=0.15
    )
    engine.metadata = meta

    # Simulate Track 8 as confirmed "New Test Person"
    with engine.lock:
        engine.active_tracks[8] = "New Test Person"
        engine.track_cache[8] = {
            "name": "New Test Person",
            "is_confirmed": True,
            "best_dist": 0.20,
            "best_norm": 1.0,
            "history": ["New Test Person"]
        }
        engine.recent_track_history[8] = {
            "box": (115, 135, 685, 675),
            "last_seen": 1,
            "name": "New Test Person",
            "confirmed": True
        }

    # Create dummy blank frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Mock YOLO boxes injection:
    # Small frame is 640x360 (scale_x=2.0, scale_y=2.0 from 1280x720)
    # So small frame box coordinates should be divided by 2.0
    box_person = [115 / 2.0, 135 / 2.0, 685 / 2.0, 675 / 2.0]
    box_shoulder = [550 / 2.0, 350 / 2.0, 800 / 2.0, 675 / 2.0]
    box_other_person = [900 / 2.0, 200 / 2.0, 1150 / 2.0, 680 / 2.0]

    # Mock _yolo_model.track to return these 3 detections
    class MockBox:
        def __init__(self, xyxy, conf, tid):
            import torch
            self.xyxy = torch.tensor(xyxy, dtype=torch.float32)
            self.conf = torch.tensor(conf, dtype=torch.float32)
            self.id = torch.tensor(tid, dtype=torch.int32)

        def __len__(self):
            return len(self.xyxy)

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class MockYOLO:
        def track(self, *args, **kwargs):
            boxes_data = MockBox(
                [box_person, box_shoulder, box_other_person],
                [0.85, 0.40, 0.80],
                [8, 9, 10]
            )
            return [MockResult(boxes_data)]

    engine._yolo_model = MockYOLO()
    engine.use_cuda = False

    result = engine.process_frame(frame, annotate=True)
    active_tids = [s.track_id for s in result.active_tracks]
    print(f"Active Track IDs in result: {active_tids}")

    # Track 8 (confirmed person) MUST be present
    assert 8 in active_tids, "Track 8 must be present"
    # Track 10 (separate distinct person) MUST be present
    assert 10 in active_tids, "Track 10 must be present"
    # Track 9 (phantom shoulder inside Track 8) MUST BE SUPPRESSED!
    assert 9 not in active_tids, f"Track 9 should be suppressed, but was found in active_tracks: {active_tids}"
    # Track 9 must NOT linger in engine.active_tracks
    assert 9 not in engine.active_tracks, f"Track 9 lingered in engine.active_tracks: {engine.active_tracks}"

    print("[PASS] Phantom shoulder track 9 was successfully suppressed, while real persons (8 and 10) were retained.")

if __name__ == "__main__":
    test_box_containment_math()
    test_engine_phantom_suppression()

import os
import sys
import time
import tracemalloc
import numpy as np
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.engine import BiometricTrackingEngine, InMemoryIndexAdapter
from src.inference.person_detector import DetectedBox, ScriptedPersonDetector
from src.inference.face_embedder import ScriptedFaceEmbedder


def run_stress_diagnostic(num_frames: int = 250):
    print("=" * 65)
    print(f"DIAGNOSTIC BENCHMARK: Multi-Subject Hand/Limb Occlusion Stress Test")
    print(f"Frames to simulate: {num_frames} | Target Subjects: 4 simultaneous tracks")
    print("=" * 65)

    adapter = InMemoryIndexAdapter(dimension=512)
    # Add a mock profile into adapter
    mock_emb = np.random.randn(512).astype(np.float32)
    mock_emb /= np.linalg.norm(mock_emb)
    adapter.add(mock_emb.reshape(1, -1))

    scripted_detector = ScriptedPersonDetector()
    engine = BiometricTrackingEngine(
        index_adapter=adapter,
        detector=scripted_detector,
        metadata_path=None
    )
    engine.metadata = {"0": "Test Subject"}

    # Generate synthetic 720p base frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 50

    # Draw simulated persons
    # Box 1: Track 1 - clear face upper region (x: 100, y: 150, w: 120, h: 300)
    # Box 2: Track 2 - hand-blocked face (x: 350, y: 150, w: 120, h: 300)
    # Box 3: Track 3 - fully covered face / phantom limb (x: 600, y: 150, w: 120, h: 300)
    # Box 4: Track 4 - moving in and out (x: 850 + ..., y: 150, w: 120, h: 300)

    tracemalloc.start()
    start_memory = tracemalloc.get_traced_memory()[0]

    latencies = {
        "pre_filter_head_presence": [],
        "total_process_frame": [],
    }

    track_id_history = set()
    purged_count = 0
    start_wall_time = time.time()

    for f in range(1, num_frames + 1):
        # Set up simulated detected boxes for frame f
        detections = []
        
        # Track 1: Present throughout
        detections.append(DetectedBox(box=(100, 150, 220, 450), conf=0.88, track_id=1))
        
        # Track 2: Hand/limb blocking face throughout
        detections.append(DetectedBox(box=(350, 150, 470, 450), conf=0.85, track_id=2))
        
        # Track 3: Occluded face throughout (valid_face_crops == 0)
        detections.append(DetectedBox(box=(600, 150, 720, 450), conf=0.82, track_id=3))

        # Track 4: Enters at f=20, leaves at f=120, re-enters at f=160
        if 20 <= f <= 120 or f >= 160:
            detections.append(DetectedBox(box=(850, 150, 970, 450), conf=0.79, track_id=4))

        scripted_detector.set_detections(detections)

        # Measure process_frame execution
        t0 = time.time()
        res = engine.process_frame(frame, annotate=True)
        t_total = (time.time() - t0) * 1000.0
        latencies["total_process_frame"].append(t_total)

        # Collect telemetry on active tracks
        current_active = {s.track_id for s in res.active_tracks}
        track_id_history.update(current_active)

    total_duration = time.time() - start_wall_time
    end_memory = tracemalloc.get_traced_memory()[0]
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    avg_frame_ms = np.mean(latencies["total_process_frame"])
    p95_frame_ms = np.percentile(latencies["total_process_frame"], 95)
    effective_fps = num_frames / total_duration

    print("\n" + "=" * 65)
    print("DIAGNOSTIC EVIDENCE REPORT")
    print("=" * 65)
    print(f"Total simulated frames     : {num_frames}")
    print(f"Total wall time elapsed    : {total_duration:.2f} s")
    print(f"Effective throughput (FPS) : {effective_fps:.2f} FPS")
    print(f"Mean latency per frame     : {avg_frame_ms:.2f} ms")
    print(f"95th percentile latency    : {p95_frame_ms:.2f} ms")
    print("-" * 65)
    print("INTERNAL STATE & MEMORY RETENTION:")
    print(f"Start Heap Memory          : {start_memory / 1024 / 1024:.2f} MB")
    print(f"End Heap Memory            : {end_memory / 1024 / 1024:.2f} MB")
    print(f"Peak Heap Memory           : {peak_memory / 1024 / 1024:.2f} MB")
    print(f"Active sessions count      : {len(engine._sessions)}")
    print(f"Track cache entries        : {len(engine.track_cache)}")
    print(f"Track last seen entries    : {len(engine.track_last_seen)}")
    print(f"Recent track history size  : {len(engine.pre_filter.recent_track_history)}")
    print(f"Head presence cache size   : {len(engine.pre_filter.head_presence_cache)}")
    print(f"Track IDs observed         : {sorted(list(track_id_history))}")
    print("-" * 65)
    print("TRACK PURGING / STATUS AUDIT:")
    for tid in sorted(list(engine._sessions.keys())):
        sess = engine._sessions[tid]
        tc = engine.track_cache.get(tid, {})
        print(f" - Track {tid}: status={sess.status.value}, valid_face_crops={tc.get('valid_face_crops', 0)}, display_name={sess.display_name}")

    print("=" * 65)


if __name__ == "__main__":
    run_stress_diagnostic(num_frames=200)

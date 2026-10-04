import threading
from typing import Dict, Any, Tuple, Optional


class TemporalSubsampler:
    """
    Temporal Sub-sampling Module for Zero-Latency Multi-Target Biometric Pipeline.
    Guarantees spatial bounding-box tracking (YOLO + ByteTrack) runs on EVERY frame,
    while heavy YuNet face alignment and ArcFace 512D feature extraction run strictly
    every N-th frame per tracked subject ID.
    """
    def __init__(self, subsample_interval: int = 3):
        self.subsample_interval = max(1, subsample_interval)
        self.track_frame_counters: Dict[int, int] = {}
        self.lock = threading.Lock()

    def should_process_biometrics(self, track_id: int) -> bool:
        """
        Determines whether heavy face detection/alignment and biometric feature extraction
        should execute on the current frame for the given track_id.
        Returns True on frame 1, frame N+1, frame 2N+1, etc.
        """
        with self.lock:
            count = self.track_frame_counters.get(track_id, 0) + 1
            self.track_frame_counters[track_id] = count
            return (count % self.subsample_interval) == 1 or self.subsample_interval == 1

    def reset_track(self, track_id: int):
        with self.lock:
            if track_id in self.track_frame_counters:
                del self.track_frame_counters[track_id]

    def clear(self):
        with self.lock:
            self.track_frame_counters.clear()

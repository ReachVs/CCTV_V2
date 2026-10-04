import time
import threading
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable


class BufferlessVideoCapture:
    """
    Background daemon frame capture thread draining OpenCV buffers with 0ms queue latency.
    Guarantees only the latest raw camera frame is held in memory (max-buffers=1, drop=true).
    """
    def __init__(self, source: Any = 0, width: int = 854, height: int = 480):
        self.source = source
        self.width = width
        self.height = height
        try:
            self.cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(self.source)
        except Exception:
            self.cap = None
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = False
        self.thread: Optional[threading.Thread] = None

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()

    def _reader_loop(self):
        while self.running:
            with self.lock:
                if not self.running:
                    break
                if self.cap is None or not self.cap.isOpened():
                    try:
                        self.cap = cv2.VideoCapture(self.source)
                        if self.cap.isOpened():
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            for _ in range(3):
                                self.cap.read()
                    except Exception as e:
                        print(f"[BufferlessVideoCapture Error] Source {self.source}: {e}")

                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.25)
                    continue

                cap_handle = self.cap

            try:
                ret, frame = cap_handle.read()
            except Exception:
                ret, frame = False, None

            if not ret or frame is None:
                time.sleep(0.005)
                continue

            with self.lock:
                self.latest_frame = frame

            time.sleep(0.001)

        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def isOpened(self) -> bool:
        with self.lock:
            if self.cap is not None:
                try:
                    return self.cap.isOpened()
                except Exception:
                    pass
            return self.running

    def stop(self):
        with self.lock:
            self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def release(self):
        self.stop()


class EventAggregator:
    """
    Central Thread-Safe Event Aggregator correlating tracked spatial bounding boxes across cameras.
    Deduplicates track states and maintains independent `active_tracks`, `track_cache`,
    `track_votes`, and `track_last_seen` dictionaries per camera stream ID.
    """
    def __init__(self):
        self.lock = threading.RLock()
        self.stream_active_tracks: Dict[str, Dict[int, str]] = {}
        self.stream_track_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.stream_track_votes: Dict[str, Dict[int, List[str]]] = {}
        self.stream_track_last_seen: Dict[str, Dict[int, int]] = {}

    def ensure_stream(self, stream_id: str):
        with self.lock:
            if stream_id not in self.stream_active_tracks:
                self.stream_active_tracks[stream_id] = {}
                self.stream_track_cache[stream_id] = {}
                self.stream_track_votes[stream_id] = {}
                self.stream_track_last_seen[stream_id] = {}

    def update_track_state(
        self,
        stream_id: str,
        track_id: int,
        name: str,
        cache_data: Optional[Dict[str, Any]] = None,
        frame_num: int = 0
    ):
        with self.lock:
            self.ensure_stream(stream_id)
            self.stream_active_tracks[stream_id][track_id] = name
            self.stream_track_last_seen[stream_id][track_id] = frame_num
            if cache_data is not None:
                if track_id not in self.stream_track_cache[stream_id]:
                    self.stream_track_cache[stream_id][track_id] = {}
                self.stream_track_cache[stream_id][track_id].update(cache_data)

    def record_vote(self, stream_id: str, track_id: int, name: str):
        with self.lock:
            self.ensure_stream(stream_id)
            if track_id not in self.stream_track_votes[stream_id]:
                self.stream_track_votes[stream_id][track_id] = []
            self.stream_track_votes[stream_id][track_id].append(name)

    def remove_track(self, stream_id: str, track_id: int):
        with self.lock:
            if stream_id in self.stream_active_tracks:
                self.stream_active_tracks[stream_id].pop(track_id, None)
                self.stream_track_cache[stream_id].pop(track_id, None)
                self.stream_track_votes[stream_id].pop(track_id, None)
                self.stream_track_last_seen[stream_id].pop(track_id, None)

    def get_stream_tracks(self, stream_id: str) -> Dict[int, str]:
        with self.lock:
            if stream_id in self.stream_active_tracks:
                return dict(self.stream_active_tracks[stream_id])
            return {}

    def get_stream_cache(self, stream_id: str) -> Dict[int, Dict[str, Any]]:
        with self.lock:
            if stream_id in self.stream_track_cache:
                return {k: dict(v) for k, v in self.stream_track_cache[stream_id].items()}
            return {}

    def get_all_active_identity_claims(self) -> Dict[str, Tuple[str, int]]:
        """Returns map of verified identity name -> (stream_id, track_id) across all active camera streams."""
        claims = {}
        with self.lock:
            for st_id, tracks in self.stream_active_tracks.items():
                for tr_id, name in tracks.items():
                    if name not in ["Unknown", "Scanning", "Ambiguous Match", "Unknown Person"]:
                        claims[name] = (st_id, tr_id)
        return claims


class CameraIngestionPool:
    """
    Unified Camera Ingestion Pool.
    Consolidates on-demand reference-counted single camera acquisition (zero-leak deferred shutdown,
    pause/resume hardware arbitration) and multi-camera RTSP/USB bufferless streams.
    """
    def __init__(
        self,
        source: Any = 0,
        width: int = 854,
        height: int = 480,
        frame_processor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        deferred_shutdown_seconds: float = 3.0,
    ):
        self.source = source
        self.default_source = source
        self.width = width
        self.height = height
        self.frame_processor = frame_processor
        self.deferred_shutdown_seconds = deferred_shutdown_seconds

        self.lock = threading.RLock()
        self.cameras: Dict[str, BufferlessVideoCapture] = {}
        self.event_aggregator = EventAggregator()

        # Reference counting & streaming state
        self.client_count = 0
        self.running = False
        self.paused = False
        self.thread: Optional[threading.Thread] = None
        self.shutdown_timer: Optional[threading.Timer] = None

        # Processed & raw frame caches
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_processed_frame: Optional[np.ndarray] = None

        # FPS metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

    # Multi-camera interface
    def add_camera(self, stream_id: str, source: Any, width: int = 854, height: int = 480) -> bool:
        with self.lock:
            if stream_id in self.cameras:
                return False
            cam = BufferlessVideoCapture(source=source, width=width, height=height)
            cam.start()
            self.cameras[stream_id] = cam
            self.event_aggregator.ensure_stream(stream_id)
            print(f"[CameraIngestionPool] Added camera stream '{stream_id}' (Source: {source})")
            return True

    def remove_camera(self, stream_id: str) -> bool:
        with self.lock:
            if stream_id not in self.cameras:
                return False
            cam = self.cameras.pop(stream_id)
            cam.stop()
            print(f"[CameraIngestionPool] Removed camera stream '{stream_id}'")
            return True

    def get_frame(self, stream_id: str = "primary") -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            cam = self.cameras.get(stream_id)
        if cam is not None:
            return cam.read()
        return False, None

    def active_stream_ids(self) -> List[str]:
        with self.lock:
            return list(self.cameras.keys())

    # Reference-counted lifecycle
    def acquire(self):
        with self.lock:
            if self.shutdown_timer is not None:
                self.shutdown_timer.cancel()
                self.shutdown_timer = None

            self.client_count += 1
            if self.client_count >= 1 and not self.running:
                self._start_thread()

    def release(self):
        with self.lock:
            self.client_count = max(0, self.client_count - 1)
            if self.client_count == 0 and self.running:
                if self.shutdown_timer is not None:
                    self.shutdown_timer.cancel()
                self.shutdown_timer = threading.Timer(self.deferred_shutdown_seconds, self._deferred_shutdown)
                self.shutdown_timer.daemon = True
                self.shutdown_timer.start()

    def _deferred_shutdown(self):
        with self.lock:
            self.shutdown_timer = None
            if self.client_count == 0 and self.running:
                self.running = False
                self._release_hardware_locked()

    def pause(self):
        with self.lock:
            self.paused = True
            if "primary" in self.cameras:
                self.cameras["primary"].stop()
                self.cameras.pop("primary", None)

    def resume(self):
        with self.lock:
            self.paused = False

    def _start_thread(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            with self.lock:
                if "primary" not in self.cameras:
                    cam = BufferlessVideoCapture(source=self.default_source, width=self.width, height=self.height)
                    cam.start()
                    self.cameras["primary"] = cam

            ret, frame = self.get_frame("primary")
            if not ret or frame is None:
                synthetic = self._generate_synthetic_frame()
                with self.lock:
                    self.latest_raw_frame = synthetic
                    self.latest_processed_frame = synthetic
                time.sleep(0.05)
                continue

            raw_copy = frame.copy()
            if self.frame_processor is not None:
                try:
                    proc = self.frame_processor(frame)
                except Exception as e:
                    print(f"[CameraIngestionPool Warning] Frame processor error: {e}")
                    proc = frame
            else:
                proc = raw_copy

            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = round(self.fps_counter / (now - self.last_fps_time), 1)
                self.fps_counter = 0
                self.last_fps_time = now

            with self.lock:
                self.latest_raw_frame = raw_copy
                self.latest_processed_frame = proc

            time.sleep(0.002)

        with self.lock:
            self._release_hardware_locked()

    def _release_hardware_locked(self):
        if "primary" in self.cameras:
            self.cameras["primary"].stop()
            self.cameras.pop("primary", None)

    def _generate_synthetic_frame(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.putText(canvas, "CCTV SYSTEM ENGINE ONLINE", (int(self.width * 0.25), int(self.height * 0.45)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(canvas, f"Time: {time.strftime('%H:%M:%S')} | Standby Mode", (int(self.width * 0.28), int(self.height * 0.55)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return canvas

    def get_raw_frame(self, stream_id: str = "primary") -> np.ndarray:
        with self.lock:
            if stream_id == "primary" and self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            ret, frame = self.get_frame(stream_id)
            if ret and frame is not None:
                return frame
            return self._generate_synthetic_frame()

    def get_processed_frame(self, stream_id: str = "primary") -> np.ndarray:
        with self.lock:
            if stream_id == "primary" and self.latest_processed_frame is not None:
                return self.latest_processed_frame.copy()
            ret, frame = self.get_frame(stream_id)
            if ret and frame is not None:
                if self.frame_processor is not None:
                    try:
                        return self.frame_processor(frame)
                    except Exception:
                        return frame
                return frame
            return self._generate_synthetic_frame()

    def stop(self):
        self.running = False
        with self.lock:
            self._release_hardware_locked()
            for stream_id, cam in list(self.cameras.items()):
                cam.stop()
            self.cameras.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def stop_all(self):
        self.stop()


# Backward compatibility aliases
MultiCameraIngestionManager = CameraIngestionPool
GlobalCameraStream = CameraIngestionPool

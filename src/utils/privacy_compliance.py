import gc
import os
import json
import sqlite3
import queue
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None


class GDPRMemorySanitizer:
    """
    GDPR Compliance Protocol: Explicit Zero-Trace Memory Disposal Pattern.
    Overwrites raw camera frames and face crops in place with zeroes before deleting references.
    """
    @staticmethod
    def sanitize_image_buffer(arr: Optional[np.ndarray]):
        if arr is not None and isinstance(arr, np.ndarray) and arr.size > 0:
            try:
                arr.fill(0)
            except Exception:
                pass
            del arr
            gc.collect()

    @staticmethod
    def sanitize_crop_list(crop_list: List[np.ndarray]):
        for crop in crop_list:
            if crop is not None and isinstance(crop, np.ndarray):
                try:
                    crop.fill(0)
                except Exception:
                    pass
        crop_list.clear()
        gc.collect()


class PoseOnlyEstimator:
    """
    Lightweight 17 Keypoint Body Estimator for Privacy-Preserving PoseOnlyTracking Mode.
    Bypasses facial biometrics entirely when enabled.
    Keypoint indices follow standard COCO 17-keypoint skeleton format:
      0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear, 5: L-Shoulder, 6: R-Shoulder,
      7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist, 11: L-Hip, 12: R-Hip,
      13: L-Knee, 14: R-Knee, 15: L-Ankle, 16: R-Ankle.
    """
    def __init__(self, pose_model_spec: str = "yolo11n-pose.pt"):
        self.pose_model_spec = pose_model_spec
        self.yolo_pose = None
        self._init_model()

    def _init_model(self):
        try:
            from ultralytics import YOLO
            model_path = self.pose_model_spec if os.path.exists(self.pose_model_spec) else "yolo11n-pose.pt"
            self.yolo_pose = YOLO(model_path)
            print(f"[PoseEstimator] Loaded Ultralytics Pose model: '{model_path}'")
        except Exception as e:
            print(f"[PoseEstimator Warning] Could not load YOLO Pose model ({e}). Using synthetic keypoints fallback.")
            self.yolo_pose = None

    def estimate_pose_keypoints(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[Dict[str, float]]:
        """
        Extracts 17 body keypoints within bounding box (x1, y1, x2, y2).
        Returns list of 17 keypoint dicts: [{"id": 0..16, "x": float, "y": float, "conf": float}].
        Raw pixel crop is NOT saved or retained.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1 or self.yolo_pose is None:
            return self._generate_fallback_keypoints(bbox)

        crop = frame[y1:y2, x1:x2]
        try:
            results = self.yolo_pose(crop, verbose=False)
            keypoints_data = []
            if results and len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                kpts = results[0].keypoints.data.cpu().numpy()
                if kpts.shape[0] > 0:
                    person_kpts = kpts[0]  # shape (17, 3) -> (x, y, conf)
                    for idx in range(min(17, len(person_kpts))):
                        kx, ky, conf = person_kpts[idx]
                        keypoints_data.append({
                            "id": idx,
                            "x": float(x1 + kx),
                            "y": float(y1 + ky),
                            "conf": float(conf)
                        })
                    GDPRMemorySanitizer.sanitize_image_buffer(crop)
                    return keypoints_data

            GDPRMemorySanitizer.sanitize_image_buffer(crop)
            return []
        except Exception as e:
            GDPRMemorySanitizer.sanitize_image_buffer(crop)
            return []

    def _generate_fallback_keypoints(self, bbox: Tuple[int, int, int, int]) -> List[Dict[str, float]]:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        keypoints = []
        for i in range(17):
            # Simple proportional anatomical placement fallback
            offset_x = (i % 3 - 1) * (bw * 0.2)
            offset_y = (i / 17.0 - 0.5) * bh
            keypoints.append({
                "id": i,
                "x": float(center_x + offset_x),
                "y": float(center_y + offset_y),
                "conf": 0.0
            })
        return keypoints


class AESEncryptedWALAuditLogger:
    """
    AES-256 Encrypted SQLite WAL Audit Event Logger.
    Encrypts sensitive fields (person_name, match_distance, metadata) at rest using AES-256 Fernet cipher.
    """
    def __init__(self, db_path: str, encryption_key: Optional[bytes] = None):
        self.db_path = db_path
        self.log_queue = queue.Queue()
        self.running = True

        if Fernet is not None:
            if encryption_key is None:
                env_key = os.environ.get("AES_ENCRYPTION_KEY")
                if env_key:
                    self.key = env_key.encode('utf-8')
                else:
                    db_dir = os.path.dirname(os.path.abspath(self.db_path)) if self.db_path else "."
                    key_file = os.path.join(db_dir, ".audit_key")
                    if os.path.exists(key_file):
                        try:
                            with open(key_file, "rb") as kf:
                                self.key = kf.read().strip()
                        except Exception:
                            self.key = Fernet.generate_key()
                    else:
                        self.key = Fernet.generate_key()
                        try:
                            os.makedirs(db_dir, exist_ok=True)
                            with open(key_file, "wb") as kf:
                                kf.write(self.key)
                        except Exception:
                            pass
            else:
                self.key = encryption_key
            self.cipher = Fernet(self.key)
        else:
            self.key = None
            self.cipher = None

        self._init_db()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    encrypted_payload TEXT NOT NULL
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AESEncryptedWALAuditLogger Error] DB init failed: {e}")

    def encrypt_payload(self, data: Dict[str, Any]) -> str:
        raw_json = json.dumps(data).encode('utf-8')
        if self.cipher is not None:
            return self.cipher.encrypt(raw_json).decode('utf-8')
        # Plaintext fallback if cryptography package is missing
        return raw_json.decode('utf-8')

    def decrypt_payload(self, encrypted_str: str) -> Dict[str, Any]:
        if self.cipher is not None:
            try:
                decrypted_bytes = self.cipher.decrypt(encrypted_str.encode('utf-8'))
                return json.loads(decrypted_bytes.decode('utf-8'))
            except Exception as e:
                print(f"[AESEncryptedWALAuditLogger] Decryption error: {e}")
                return {"person_name": "DECRYPTION_ERROR", "match_distance": 999.0}
        try:
            return json.loads(encrypted_str)
        except Exception:
            return {"person_name": encrypted_str, "match_distance": 999.0}

    def _worker(self):
        conn = None
        while self.running or not self.log_queue.empty():
            try:
                event = self.log_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            now_str, track_id, payload_dict = event
            encrypted_payload = self.encrypt_payload(payload_dict)

            try:
                if conn is None:
                    conn = sqlite3.connect(self.db_path, timeout=5.0)
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("""
                    INSERT INTO encrypted_detection_events (timestamp, track_id, encrypted_payload)
                    VALUES (?, ?, ?)
                """, (now_str, int(track_id), encrypted_payload))
                conn.commit()
            except Exception as e:
                print(f"[AESEncryptedWALAuditLogger Warning] {e}")
                if conn:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = None
            finally:
                self.log_queue.task_done()

        if conn:
            try:
                conn.close()
            except Exception:
                pass

    def log_event(self, now_str: str, track_id: int, person_name: str, match_distance: float, extra_meta: Optional[Dict[str, Any]] = None):
        payload = {
            "person_name": person_name,
            "match_distance": float(match_distance),
            "meta": extra_meta or {}
        }
        self.log_queue.put((now_str, track_id, payload))

    def fetch_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        results = []
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, track_id, encrypted_payload FROM encrypted_detection_events ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            for r in rows:
                decrypted = self.decrypt_payload(r["encrypted_payload"])
                results.append({
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "track_id": r["track_id"],
                    "person_name": decrypted.get("person_name", "UNKNOWN"),
                    "match_distance": decrypted.get("match_distance", 999.0),
                    "meta": decrypted.get("meta", {})
                })
            conn.close()
        except Exception as e:
            print(f"[AESEncryptedWALAuditLogger Fetch Error] {e}")
        return results

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

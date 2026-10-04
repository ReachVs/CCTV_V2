import math
import numpy as np


class ConfidenceKalmanFilter:
    """
    Confidence-Based Kalman Filter (CBKF).
    Dynamically adjusts measurement noise covariance matrix R based on detection confidence score conf.
    Non-Linear Scaling Formula: R(conf) = R_base * exp(-beta * (conf - c_0))
    Prevents identity drift during partial occlusions and turns by trusting motion state trajectory.
    """
    def __init__(self, dt=1.0, std_weight_position=1.0/20.0, std_weight_velocity=1.0/160.0, beta=3.0, c0=0.5):
        self.dt = float(dt)
        self.std_weight_position = std_weight_position
        self.std_weight_velocity = std_weight_velocity
        self.beta = float(beta)
        self.c0 = float(c0)

        # State vector x = [cx, cy, w, h, v_cx, v_cy, v_w, v_h]^T
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0

        # State transition matrix F
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        # Measurement matrix H (observes cx, cy, w, h)
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Base process noise Q
        self.Q = np.eye(8, dtype=np.float32) * 0.01

        # Base measurement noise R_base
        self.R_base = np.eye(4, dtype=np.float32) * 1.0

        # Pre-allocate identity matrix I
        self.I = np.eye(8, dtype=np.float32)

    def initiate(self, measurement: tuple):
        cx, cy, w, h = measurement
        self.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 1.0

    def predict(self):
        self.x[:4] += self.dt * self.x[4:]
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_box()

    def update(self, measurement: tuple, confidence: float = 1.0):
        cx, cy, w, h = measurement
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        
        conf = float(confidence)
        scale_factor = float(math.exp(-self.beta * (conf - self.c0)))
        scale_factor = max(0.1, min(scale_factor, 50.0))
        
        R_k = self.R_base * scale_factor
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R_k
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.get_box()

    def get_box(self):
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = int(cx - w / 2.0)
        y1 = int(cy - h / 2.0)
        x2 = int(cx + w / 2.0)
        y2 = int(cy + h / 2.0)
        return (x1, y1, x2, y2)


class JitterFilterEngine:
    """
    Zero-Jitter Bounding Box Filter Engine with Confidence-Based Kalman Filtering (CBKF)
    and Velocity Displacement Filtering for Limb Movement Suppression.
    Velocity formula: velocity = sqrt((x_t - x_{t-1})^2 + (y_t - y_{t-1})^2)
    """
    def __init__(self, deadzone_velocity=3.5, min_alpha=0.08, max_alpha=0.85, vel_scale=16.0, erratic_threshold=45.0):
        self.deadzone_velocity = deadzone_velocity
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.vel_scale = vel_scale
        self.erratic_threshold = erratic_threshold
        self.smoothed_states = {}   # track_id -> (x1, y1, x2, y2)
        self.kalman_filters = {}    # track_id -> ConfidenceKalmanFilter
        self.track_velocities = {}  # track_id -> float (pixels/frame displacement velocity)
        self.last_centroids = {}    # track_id -> (cx, cy)

    def filter_box(self, track_id: int, target_box: tuple, confidence: float = 1.0) -> tuple:
        """
        Applies CBKF Kalman filtering, displacement velocity computation, and EMA smoothing.
        """
        tx1, ty1, tx2, ty2 = target_box
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        tw = float(tx2 - tx1)
        th = float(ty2 - ty1)

        # Calculate displacement velocity
        if track_id in self.last_centroids:
            prev_cx, prev_cy = self.last_centroids[track_id]
            velocity = math.hypot(tcx - prev_cx, tcy - prev_cy)
        else:
            velocity = 0.0

        self.last_centroids[track_id] = (tcx, tcy)
        self.track_velocities[track_id] = velocity

        # 1. Update Confidence-Based Kalman Filter (CBKF)
        if track_id not in self.kalman_filters:
            kf = ConfidenceKalmanFilter()
            kf.initiate((tcx, tcy, tw, th))
            self.kalman_filters[track_id] = kf
        else:
            kf = self.kalman_filters[track_id]
            kf.predict()
            kf.update((tcx, tcy, tw, th), confidence=confidence)

        kx1, ky1, kx2, ky2 = kf.get_box()

        # 2. Velocity-Adaptive EMA & Zero-Jitter Deadzone Lock
        old_state = self.smoothed_states.get(track_id)
        if old_state is None:
            filtered = (int(kx1), int(ky1), int(kx2), int(ky2))
            self.smoothed_states[track_id] = filtered
            return filtered

        ox1, oy1, ox2, oy2 = old_state

        if velocity < self.deadzone_velocity:
            filtered = (ox1, oy1, ox2, oy2)
        else:
            alpha = min(self.max_alpha, max(self.min_alpha, (velocity - 1.5) / self.vel_scale))
            fx1 = int((1.0 - alpha) * ox1 + alpha * kx1)
            fy1 = int((1.0 - alpha) * oy1 + alpha * ky1)
            fx2 = int((1.0 - alpha) * ox2 + alpha * kx2)
            fy2 = int((1.0 - alpha) * oy2 + alpha * ky2)
            filtered = (fx1, fy1, fx2, fy2)

        self.smoothed_states[track_id] = filtered
        return filtered

    def get_velocity(self, track_id: int) -> float:
        """Returns current displacement velocity for track_id in pixels/frame."""
        return self.track_velocities.get(track_id, 0.0)

    def is_erratic_movement(self, track_id: int, threshold: float = 45.0) -> bool:
        """
        Returns True if displacement velocity exceeds threshold (indicating erratic hand/arm waving).
        """
        thresh = threshold or self.erratic_threshold
        vel = self.get_velocity(track_id)
        return vel > thresh

    def reset_track(self, track_id: int):
        self.smoothed_states.pop(track_id, None)
        self.kalman_filters.pop(track_id, None)
        self.track_velocities.pop(track_id, None)
        self.last_centroids.pop(track_id, None)

    def clear(self):
        self.smoothed_states.clear()
        self.kalman_filters.clear()
        self.track_velocities.clear()
        self.last_centroids.clear()

"""
Deep Module: BiometricIngestionScheduler
========================================
Encapsulates biometric temporal subsampling, track-level rate limiting,
cooldown periods, erratic movement gating, and face crop quality verification.
Callers simply query `should_extract(session, crop_shape, current_frame)`
without managing timers, cooldown counters, or frame modulos directly.
"""

from typing import Dict, Any, Optional, Tuple
import threading

from src.domain.subject_session import SubjectSession, TrackStatus


class BiometricIngestionScheduler:
    """
    Deep Module: Governs admission cadence for biometric embedding extraction.
    Hides erratic motion checks, limb cooldown counters, unconfirmed 2-frame vs
    confirmed 15-frame intervals, and temporal subsampler integration.
    """

    def __init__(
        self,
        jitter_engine: Optional[Any] = None,
        temporal_subsampler: Optional[Any] = None,
    ):
        self.jitter_engine = jitter_engine
        self.temporal_subsampler = temporal_subsampler
        self.lock = threading.RLock()

        self.limb_cooldowns: Dict[int, int] = {}
        self.track_last_submitted_frame: Dict[int, int] = {}

    def set_cooldown(self, track_id: int, until_frame: int) -> None:
        """Sets a cooldown frame threshold before which no biometrics will be extracted."""
        with self.lock:
            self.limb_cooldowns[track_id] = until_frame

    def record_submission(self, track_id: int, frame_num: int) -> None:
        """Records that a biometric crop was submitted for extraction at the given frame."""
        with self.lock:
            self.track_last_submitted_frame[track_id] = frame_num

    def purge_track(self, track_id: int) -> None:
        """Purges cooldowns and submission timestamps for a terminated track."""
        with self.lock:
            self.limb_cooldowns.pop(track_id, None)
            self.track_last_submitted_frame.pop(track_id, None)

    def should_extract(
        self,
        session: SubjectSession,
        crop_shape: Optional[Tuple[int, int]],
        current_frame: int,
        is_contended: bool = False,
    ) -> bool:
        """
        Determines whether a face crop should be submitted for embedding extraction.

        Rules encapsulated:
        1. Bounding box & crop dimensions:
           - Body box height >= 90 and width >= 45 (avoid tiny distant noise)
           - Crop dimensions height >= 80 and width >= 80
        2. Movement stability:
           - Not erratic (via jitter_engine)
        3. Rate limiting & cooldown:
           - Not in active limb cooldown
           - Not in crossover contention
        4. Cadence:
           - Unconfirmed scanning tracks: requires track_age >= 2, submits every 2 frames
             for fast initial identification (<0.3s).
           - Confirmed tracks: submits every 15 frames to conserve compute while maintaining
             continuity.
        5. Subsampler readiness:
           - Confirmed tracks must also satisfy temporal_subsampler gating.
        """
        track_id = session.track_id
        box = session.box
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]

        # 1. Distance & Crop Dimension Check
        if box_h < 90 or box_w < 45:
            return False
        if crop_shape is None:
            return False
        crop_h, crop_w = crop_shape[:2]
        if crop_h < 80 or crop_w < 80:
            return False

        with self.lock:
            # 2. Movement Stability Check
            if self.jitter_engine is not None and hasattr(self.jitter_engine, "is_erratic_movement"):
                if self.jitter_engine.is_erratic_movement(track_id, threshold=80.0):
                    return False

            # 3. Cooldown & Contention Check
            if current_frame < self.limb_cooldowns.get(track_id, 0):
                return False
            if is_contended or session.status == TrackStatus.CONTENDED:
                return False

            # 4. Cadence & Track Age Check
            is_confirmed = session.is_confirmed
            track_age = session.track_age(current_frame)
            last_sub_frame = self.track_last_submitted_frame.get(track_id, 0)

            if not is_confirmed:
                if track_age < 2:
                    return False
                interval = 2
            else:
                interval = 15

            if last_sub_frame != 0 and (current_frame - last_sub_frame < interval):
                return False

            # 5. Temporal Subsampler Check for confirmed tracks
            if is_confirmed and self.temporal_subsampler is not None:
                if hasattr(self.temporal_subsampler, "should_process_biometrics"):
                    if not self.temporal_subsampler.should_process_biometrics(track_id):
                        return False

        return True

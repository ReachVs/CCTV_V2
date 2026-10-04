"""
Face Embedder Seam & Adapters.
Encapsulates face super-resolution, alignment, ArcFace/DeepFace neural inference,
and L2-normalization behind a unified, testable interface.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable, Dict, Union, List
import cv2
import numpy as np

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

from src.utils.face_align import YuNetFaceAligner
from src.utils.fsrcnn_upscaler import FSRCNNUpscaler


class FaceEmbedder(ABC):
    """
    Abstract seam for extracting normalized 512D biometric embedding vectors from face crops or images.
    """

    @abstractmethod
    def embed(
        self,
        img_or_crop: np.ndarray,
        align: bool = True,
        strict_alignment: bool = False,
        upscale: bool = False,
    ) -> Optional[np.ndarray]:
        """
        Extract an L2-normalized 512D biometric vector.

        Args:
            img_or_crop: Raw facial image or bounding box crop (BGR uint8).
            align: Whether to perform 5-point landmark geometric face alignment.
            strict_alignment: If True, fails (returns None) if landmarks cannot be detected.
            upscale: If True, runs super-resolution upscaling (e.g. FSRCNN) prior to alignment.

        Returns:
            Unit-normalized np.ndarray of shape (512,), or None if face detection/extraction fails.
        """
        pass

    @property
    def keras_model(self) -> Any:
        return None

    @keras_model.setter
    def keras_model(self, model: Any) -> None:
        pass


class ArcFaceEmbedder(FaceEmbedder):
    """
    Production adapter using ArcFace 512D embeddings.
    Combines FSRCNN super-resolution upscaling, YuNet landmark alignment,
    fast Keras tensor inference (with DeepFace fallback), and L2 normalization.
    """

    def __init__(
        self,
        model_name: str = "ArcFace",
        keras_model: Optional[Any] = None,
        face_aligner: Optional[YuNetFaceAligner] = None,
        upscaler: Optional[FSRCNNUpscaler] = None,
        lazy_init: bool = False,
    ):
        self.model_name = model_name
        self.face_aligner = face_aligner or YuNetFaceAligner()
        self.upscaler = upscaler or FSRCNNUpscaler()
        self._keras_model = keras_model

        if self._keras_model is None and not lazy_init:
            self._init_model()

    def _init_model(self):
        try:
            if DeepFace is not None:
                model_wrapper = DeepFace.build_model(self.model_name)
                self._keras_model = getattr(model_wrapper, "model", None)
                if self._keras_model is not None:
                    dummy_in = np.zeros((1, 112, 112, 3), dtype=np.float32)
                    _ = self._keras_model(dummy_in, training=False)
                    print("[ArcFaceEmbedder] Fast ArcFace Tensor Engine initialized.")
        except Exception as e:
            print(f"[ArcFaceEmbedder Warning] Embedder model initialization: {e}")

    @property
    def keras_model(self) -> Any:
        return self._keras_model

    @keras_model.setter
    def keras_model(self, model: Any) -> None:
        self._keras_model = model

    def embed(
        self,
        img_or_crop: np.ndarray,
        align: bool = True,
        strict_alignment: bool = False,
        upscale: bool = False,
    ) -> Optional[np.ndarray]:
        if img_or_crop is None or img_or_crop.size == 0:
            return None

        # 1. Optional Super-Resolution Upscaling (for small stream crops)
        if upscale and self.upscaler is not None:
            enhanced = self.upscaler.upscale(img_or_crop)
            crop_h, crop_w = enhanced.shape[:2]
            if crop_w > 320 or crop_h > 320:
                scale = 320.0 / max(crop_w, crop_h)
                new_w = max(1, int(crop_w * scale))
                new_h = max(1, int(crop_h * scale))
                enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            enhanced = img_or_crop

        # 2. Geometric Face Alignment via YuNet 5-Point Landmarks
        aligned = enhanced
        if align and self.face_aligner is not None:
            try:
                aligned = self.face_aligner.align_face(enhanced, output_size=(112, 112), strict=strict_alignment)
            except Exception:
                aligned = None

            if aligned is None:
                if strict_alignment:
                    return None
                # Fallback to direct resize if non-strict
                aligned = cv2.resize(enhanced, (112, 112), interpolation=cv2.INTER_LINEAR)
        else:
            if aligned.shape[:2] != (112, 112):
                aligned = cv2.resize(aligned, (112, 112), interpolation=cv2.INTER_LINEAR)

        if aligned is None or aligned.size == 0:
            return None

        # 3. Neural Embedding Extraction
        if self._keras_model is not None:
            try:
                img_tensor = (aligned.astype(np.float32) / 255.0)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                emb_raw = self._keras_model(img_tensor, training=False).numpy().flatten()
                norm_val = float(np.linalg.norm(emb_raw))
                if norm_val > 0.3:
                    return emb_raw / norm_val
                return None
            except Exception:
                pass

        if DeepFace is not None:
            try:
                try:
                    reps = DeepFace.represent(
                        img_path=aligned,
                        model_name=self.model_name,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                except Exception:
                    reps = DeepFace.represent(
                        img_path=aligned,
                        model_name=self.model_name,
                        detector_backend="opencv",
                        enforce_detection=False,
                    )
                if reps and len(reps) > 0:
                    raw_e = reps[0].get("embedding") if isinstance(reps[0], dict) else None
                    if raw_e is not None:
                        emb = np.array(raw_e, dtype=np.float32)
                        norm = float(np.linalg.norm(emb))
                        return emb / norm if norm > 0 else emb
            except Exception:
                pass

        return None


class ScriptedFaceEmbedder(FaceEmbedder):
    """
    Deterministic test adapter for face embedding extraction.
    Returns pre-configured vectors or callable responses without loading DeepFace or TensorFlow.
    """

    def __init__(self, vector: Optional[Any] = None):
        self._vector = vector
        self.call_count = 0
        self._keras_model = None

    def set_vector(self, vector: Any) -> None:
        self._vector = vector

    @property
    def keras_model(self) -> Any:
        return self._keras_model

    @keras_model.setter
    def keras_model(self, model: Any) -> None:
        self._keras_model = model

    def embed(
        self,
        img_or_crop: np.ndarray,
        align: bool = True,
        strict_alignment: bool = False,
        upscale: bool = False,
    ) -> Optional[np.ndarray]:
        self.call_count += 1
        if img_or_crop is None or img_or_crop.size == 0:
            return None

        val = self._vector
        if callable(val):
            val = val(img_or_crop)

        if val is None:
            return None

        arr = np.array(val, dtype=np.float32).flatten()
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 0 else arr

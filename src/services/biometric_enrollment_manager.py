"""
Deep Module: BiometricEnrollmentManager
======================================
Encapsulates zero-friction subject face enrollment behind two high-leverage seams:
1. `preview_and_curate`: Decodes uploaded images, aligns faces with YuNet (Umeyama transform),
   computes Laplacian sharpness metrics, and returns clean base64 previews with recommendation tags.
2. `commit_enrollment`: Enforces OWASP-compliant filesystem isolation, writes raw crops to disk,
   extracts normalized 512D ArcFace embeddings, indexes multi-exemplars into FAISS and SQLite,
   and hot-reloads the live inference engine without requiring manual folder management.
"""

from dataclasses import dataclass, field
import base64
import os
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class CuratedFacePreview:
    crop_id: str
    image_b64: str
    sharpness_score: float
    confidence: float
    is_recommended: bool
    status_label: str = "Optimal Quality"


@dataclass
class EnrollmentResult:
    status: str
    subject_id: int
    name: str
    exemplar_count: int
    total_profiles: int
    message: str


class BiometricEnrollmentManager:
    """
    Deep Module: Manages end-to-end face enrollment lifecycle.
    Hides image decoding, facial alignment, sharpness scoring, path sanitization,
    multi-exemplar embedding generation, and atomic FAISS/SQLite hot-reloading.
    """

    def __init__(
        self,
        known_faces_dir: str,
        face_aligner: Optional[Any] = None,
        embedder: Optional[Any] = None,
        biometric_repo: Optional[Any] = None,
        engine: Optional[Any] = None,
    ):
        self.known_faces_dir = os.path.abspath(known_faces_dir)
        self.face_aligner = face_aligner
        self.embedder = embedder
        self.biometric_repo = biometric_repo
        self.engine = engine

    def _compute_sharpness(self, img: np.ndarray) -> float:
        """Computes variance of the Laplacian as a metric of focus/sharpness."""
        try:
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 0.0

    def _get_safe_subject_dir(self, name: str) -> Optional[str]:
        """
        Safely resolves a directory path within known_faces_dir for a subject.
        Guards against path traversal attacks (OWASP Top 10 A01 / CWE-22).
        """
        if not name or not isinstance(name, str):
            return None
        # Explicitly reject directory traversal attempts
        if ".." in name or "/" in name or "\\" in name:
            return None

        clean = os.path.basename(name.strip())
        if not clean or clean in (".", ".."):
            return None

        clean_subfolder = clean.replace(" ", "_")
        cand_path = os.path.abspath(os.path.join(self.known_faces_dir, clean_subfolder))
        if cand_path.startswith(self.known_faces_dir):
            return cand_path
        return None

    def preview_and_curate(self, image_bytes_list: List[bytes]) -> List[CuratedFacePreview]:
        """
        Processes uploaded raw image bytes, detects and aligns faces,
        evaluates image focus sharpness, and returns preview descriptors for user validation.
        """
        previews: List[CuratedFacePreview] = []

        for raw_bytes in image_bytes_list:
            if not raw_bytes:
                continue

            try:
                nparr = np.frombuffer(raw_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue

                # 1. Align face using YuNet (Umeyama transform to 112x112)
                aligned_crop = None
                if self.face_aligner is not None and hasattr(self.face_aligner, "align_face"):
                    aligned_crop = self.face_aligner.align_face(img, output_size=(112, 112), strict=False)
                elif self.face_aligner is None:
                    # Fallback: if no aligner is provided, check if img is already a crop
                    h, w = img.shape[:2]
                    if 80 <= h <= 600 and 80 <= w <= 600 and (0.6 <= float(w)/float(h) <= 1.4):
                        aligned_crop = cv2.resize(img, (112, 112))

                if aligned_crop is None or aligned_crop.size == 0:
                    continue

                # 2. Focus sharpness assessment
                sharpness = self._compute_sharpness(aligned_crop)
                is_recommended = sharpness >= 20.0
                if sharpness >= 80.0:
                    status_label = "Optimal Sharpness"
                elif sharpness >= 30.0:
                    status_label = "Good Quality"
                elif sharpness >= 15.0:
                    status_label = "Acceptable (Low Focus)"
                else:
                    status_label = "Blurry / Soft"

                # 3. Base64 encoding for instantaneous UI thumbnail rendering
                success, encoded_buf = cv2.imencode(".jpg", aligned_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not success:
                    continue

                b64_str = base64.b64encode(encoded_buf.tobytes()).decode("utf-8")
                b64_uri = f"data:image/jpeg;base64,{b64_str}"

                crop_id = str(uuid.uuid4())[:8]
                previews.append(CuratedFacePreview(
                    crop_id=crop_id,
                    image_b64=b64_uri,
                    sharpness_score=round(sharpness, 1),
                    confidence=0.95 if is_recommended else 0.70,
                    is_recommended=is_recommended,
                    status_label=status_label
                ))
            except Exception as e:
                print(f"[BiometricEnrollmentManager] Warning during image curation: {e}")
                continue

        return previews

    def commit_enrollment(
        self,
        name: str,
        profile_data: Dict[str, Any],
        approved_crops: List[bytes],
    ) -> EnrollmentResult:
        """
        Commits approved face crops into storage, extracts 512D ArcFace embeddings,
        saves multi-exemplars into BiometricRepository/SQLite, and hot-reloads the FAISS index.
        """
        clean_name = name.strip()
        if not clean_name:
            return EnrollmentResult(
                status="error",
                subject_id=-1,
                name="",
                exemplar_count=0,
                total_profiles=getattr(self.engine, "ntotal", 0),
                message="Subject full name is required."
            )

        faces_dir = self._get_safe_subject_dir(clean_name)
        if not faces_dir:
            return EnrollmentResult(
                status="error",
                subject_id=-1,
                name=clean_name,
                exemplar_count=0,
                total_profiles=getattr(self.engine, "ntotal", 0),
                message="Invalid subject name or illegal path characters."
            )

        os.makedirs(faces_dir, exist_ok=True)
        clean_folder = os.path.basename(faces_dir)

        embeddings: List[np.ndarray] = []
        timestamp = int(time.time())

        for idx, crop_bytes in enumerate(approved_crops):
            if not crop_bytes:
                continue

            try:
                # Support either raw binary bytes or base64 data URLs
                if isinstance(crop_bytes, str) and crop_bytes.startswith("data:"):
                    crop_bytes = base64.b64decode(crop_bytes.split(",")[1])
                elif isinstance(crop_bytes, str):
                    crop_bytes = base64.b64decode(crop_bytes)

                nparr = np.frombuffer(crop_bytes, np.uint8)
                crop_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if crop_img is None or crop_img.size == 0:
                    continue

                # 1. Save image file to dedicated subject folder
                filename = f"{clean_folder}_Pose_{idx+1}_{timestamp}_{idx}.jpg"
                save_path = os.path.join(faces_dir, filename)
                cv2.imwrite(save_path, crop_img)

                # 2. Extract 512D ArcFace embedding via FaceEmbedder seam
                emb_v = None
                if self.embedder is not None:
                    if hasattr(self.embedder, "embed"):
                        emb_v = self.embedder.embed(crop_img, align=False)
                    elif hasattr(self.embedder, "extract_face_embedding"):
                        emb_v = self.embedder.extract_face_embedding(crop_img, align=False)

                if emb_v is None and self.engine is not None and hasattr(self.engine, "extract_face_embedding"):
                    emb_v = self.engine.extract_face_embedding(crop_img, align=False)

                if emb_v is not None and len(emb_v) == 512:
                    norm = np.linalg.norm(emb_v)
                    normalized_emb = emb_v / (norm + 1e-10)
                    embeddings.append(normalized_emb.astype(np.float32))
            except Exception as e:
                print(f"[BiometricEnrollmentManager] Warning processing approved crop #{idx+1}: {e}")

        if not embeddings:
            return EnrollmentResult(
                status="error",
                subject_id=-1,
                name=clean_name,
                exemplar_count=0,
                total_profiles=getattr(self.engine, "ntotal", 0),
                message="No valid face embeddings could be generated from the approved crops."
            )

        # 3. Store multi-exemplars in BiometricRepository and atomically reload live FAISS index
        try:
            if self.biometric_repo is not None and hasattr(self.biometric_repo, "add_subject"):
                self.biometric_repo.add_subject(
                    name=clean_name,
                    embeddings=embeddings,
                    profile_data=profile_data,
                    trigger_reload=False
                )

            if self.engine is not None:
                lock = getattr(self.engine, "lock", None)
                if lock is not None:
                    with lock:
                        if hasattr(self.engine, "reload_index"):
                            self.engine.reload_index()
                elif hasattr(self.engine, "reload_index"):
                    self.engine.reload_index()

            total_profiles = getattr(self.engine, "ntotal", len(embeddings))
            return EnrollmentResult(
                status="success",
                subject_id=0,
                name=clean_name,
                exemplar_count=len(embeddings),
                total_profiles=total_profiles,
                message=f"Successfully enrolled '{clean_name}' with {len(embeddings)} multi-exemplar vectors."
            )
        except Exception as e:
            return EnrollmentResult(
                status="error",
                subject_id=-1,
                name=clean_name,
                exemplar_count=0,
                total_profiles=getattr(self.engine, "ntotal", 0),
                message=f"Database synchronization error: {str(e)}"
            )

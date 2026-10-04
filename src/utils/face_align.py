import cv2
import numpy as np
import os

import threading

# Standard 112x112 canonical facial landmarks (ArcFace / AdaFace reference target)
CANONICAL_5_POINTS_112 = np.array([
    [38.2946, 51.6963],  # Left Eye
    [73.5318, 51.5014],  # Right Eye
    [56.0252, 71.7366],  # Nose Tip
    [41.5493, 92.3655],  # Left Mouth Corner
    [70.7299, 92.2041]   # Right Mouth Corner
], dtype=np.float32)


def umeyama_similarity_transform(src_pts, dst_pts=CANONICAL_5_POINTS_112):
    """
    Computes 2D similarity transform matrix M (rotation, scale, translation) 
    mapping src_pts to dst_pts via Umeyama algorithm.
    """
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    
    num_pts, dim = src.shape
    if num_pts != dst.shape[0] or dim != 2:
        return None
        
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    
    src_var = np.mean(np.sum(src_demean ** 2, axis=1))
    if src_var == 0:
        return None
        
    sigma = (dst_demean.T @ src_demean) / num_pts
    
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(2, dtype=np.float32)
    
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
        
    R = U @ S @ Vt
    scale = 1.0 / src_var * np.trace(np.diag(D) @ S)
    
    t = dst_mean - scale * (R @ src_mean)
    
    M = np.zeros((2, 3), dtype=np.float32)
    M[:2, :2] = scale * R
    M[:2, 2] = t
    return M


class YuNetFaceAligner:
    """
    Strict YuNet 5-landmark facial alignment engine using 5-point Umeyama similarity transform.
    Normalizes off-angle profile faces into 112x112 canonical frontal coordinate space.
    Thread-safe synchronization protects OpenCV FaceDetectorYN state across concurrent workers.
    """
    def __init__(self, score_threshold=0.60, nms_threshold=0.3):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.lock = threading.Lock()
        self.detector = None
        self.cascade = None
        self._init_yunet()
        self._init_cascade()

    def _init_cascade(self):
        try:
            self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception as e:
            print(f"[YuNetFaceAligner Warning] Cascade init: {e}")

    def _init_yunet(self):
        try:
            if hasattr(cv2, 'FaceDetectorYN'):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                model_paths = [
                    os.path.join(project_root, "data", "face_detection_yunet_2023mar.onnx"),
                    "data/face_detection_yunet_2023mar.onnx",
                    "face_detection_yunet_2023mar.onnx"
                ]
                model_file = next((p for p in model_paths if os.path.exists(p)), None)
                if model_file:
                    self.detector = cv2.FaceDetectorYN.create(
                        model_file, "", (320, 320), self.score_threshold, self.nms_threshold
                    )
        except Exception as e:
            print(f"[YuNetFaceAligner Warning] FaceDetectorYN init: {e}")

    def align_face(self, crop, output_size=(112, 112), strict=True):
        """
        Aligns raw face crop into output_size canonical frontal image using Umeyama similarity transform.
        If strict=True and 0 facial landmarks detected, falls back to Haar detection or returns None.
        """
        if crop is None or crop.size == 0:
            return None
            
        h, w = crop.shape[:2]
        if h < 25 or w < 25:
            return None
        
        landmarks = None
        if self.detector is not None:
            try:
                with self.lock:
                    self.detector.setInputSize((w, h))
                    _, faces = self.detector.detect(crop)
                if faces is not None and len(faces) > 0:
                    crop_cx, crop_cy = w / 2.0, h / 2.0
                    centered_faces = []
                    for f in faces:
                        fcx = f[0] + f[2] / 2.0
                        fcy = f[1] + f[3] / 2.0
                        if 0.10 * w <= fcx <= 0.90 * w and 0.05 * h <= fcy <= 0.95 * h:
                            dist_to_center = (fcx - crop_cx)**2 + (fcy - crop_cy)**2
                            centered_faces.append((f, dist_to_center))
                    if centered_faces:
                        best_face = min(centered_faces, key=lambda item: item[1])[0]
                        landmarks = best_face[4:14].reshape((5, 2))
                    else:
                        best_face = max(faces, key=lambda f: f[2] * f[3])
                        landmarks = best_face[4:14].reshape((5, 2))
            except Exception:
                landmarks = None

        if landmarks is not None:
            dst_pts = CANONICAL_5_POINTS_112
            if output_size != (112, 112):
                scale_w = output_size[0] / 112.0
                scale_h = output_size[1] / 112.0
                dst_pts = CANONICAL_5_POINTS_112 * np.array([scale_w, scale_h])

            M = umeyama_similarity_transform(landmarks, dst_pts)
            if M is not None:
                aligned = cv2.warpAffine(crop, M, output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                return aligned

        # Fallback to Haar Cascade if landmarks could not be computed
        head_region = crop[:int(h * 0.90), :]
        if head_region.size > 0 and self.cascade is not None:
            try:
                gray = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(25, 25))
                if len(faces) > 0:
                    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                    face_sub = head_region[fy:fy+fh, fx:fx+fw]
                    return cv2.resize(face_sub, output_size, interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        if strict:
            return None

        return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)


def apply_clahe_contrast_lifting(crop_bgr, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Applies CLAHE to the L-channel of LAB color space to lift dark shadow details."""
    if crop_bgr is None or crop_bgr.size == 0:
        return crop_bgr
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_chan)
        lab_enhanced = cv2.merge((l_clahe, a_chan, b_chan))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    except Exception:
        return crop_bgr


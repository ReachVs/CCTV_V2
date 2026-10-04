import cv2
import numpy as np
import os

class FSRCNNUpscaler:
    """
    Fast Super-Resolution CNN (FSRCNN) Pre-Processor.
    Restores high-frequency facial details from low-resolution crops (< 60px)
    before landmarks alignment and AdaFace embedding extraction.
    """
    def __init__(self, target_min_dim=112, scale=2):
        self.target_min_dim = target_min_dim
        self.scale = scale
        self.sr_engine = None
        self._init_fsrcnn()

    def _init_fsrcnn(self):
        """Initializes cv2.dnn_superres if model weights exist."""
        try:
            if hasattr(cv2, 'dnn_superres'):
                model_paths = [
                    "data/FSRCNN_x2.pb",
                    "FSRCNN_x2.pb"
                ]
                model_file = next((p for p in model_paths if os.path.exists(p)), None)
                if model_file:
                    sr = cv2.dnn_superres.DnnSuperResImpl_create()
                    sr.readModel(model_file)
                    sr.setModel("fsrcnn", self.scale)
                    self.sr_engine = sr
        except Exception as e:
            print(f"[FSRCNNUpscaler Warning] SuperRes init: {e}")

    def upscale(self, crop):
        """
        Upscales crop if resolution is below target_min_dim.
        Uses FSRCNN DNN model when available, falling back to high-frequency detail sharpening.
        """
        if crop is None or crop.size == 0:
            return crop
            
        h, w = crop.shape[:2]
        if min(h, w) >= self.target_min_dim:
            return crop
            
        if self.sr_engine is not None:
            try:
                upscaled = self.sr_engine.upsample(crop)
                return upscaled
            except Exception:
                pass
                
        # High-frequency reconstruction fallback: Lanczos4 upscaling + Unsharp Masking filter
        scale_factor = float(self.target_min_dim) / float(min(h, w))
        new_w = int(round(w * scale_factor))
        new_h = int(round(h * scale_factor))
        
        upscaled = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply high-frequency unsharp mask kernel to enhance facial edges
        gaussian_blur = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
        sharpened = cv2.addWeighted(upscaled, 1.4, gaussian_blur, -0.4, 0)
        return sharpened

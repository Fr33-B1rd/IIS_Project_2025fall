#!/usr/bin/env python
"""
perception.py

Emotion perception for Furhat TRPG DM:
- Uses DINOv3 ViT-S/16 for visual features
- Uses 3-class SVM (excited / nervous / confused)
- Uses temporal smoothing over last N frames
- Provides EmotionDetector.get_emotion() for main loop

Requires:
  - dinov3_svm_3class.joblib (trained via train_dinov3_svm_3class.py)
"""

from collections import deque
from pathlib import Path
from typing import Dict, Tuple, Optional

import cv2
import joblib
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

PRETRAINED_MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


class EmotionPerception:
    """
    Low-level: given a single BGR frame, return smoothed emotion + probabilities.
    """

    def __init__(self, model_path: str = "dinov3_svm_3class.joblib", smooth_window: int = 15):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Cannot find model file: {model_path}. "
                f"Make sure you ran train_dinov3_svm_3class.py first."
            )

        # Load SVM + encoder
        bundle = joblib.load(model_path)
        self.svm = bundle["svm"]
        self.label_encoder = bundle["label_encoder"]

        # Load DINOv3 backbone
        print(f"[EmotionPerception] Loading DINOv3 backbone: {PRETRAINED_MODEL_NAME}")
        self.processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
        self.model = AutoModel.from_pretrained(
            PRETRAINED_MODEL_NAME, device_map="auto", dtype="auto"
        )
        self.model.eval()

        # Temporal smoothing buffer
        self.smooth_window = max(1, smooth_window)
        self.prob_buffer: deque[np.ndarray] = deque(maxlen=self.smooth_window)

        print("[EmotionPerception] Classes:", list(self.label_encoder.classes_))
        print(f"[EmotionPerception] Smoothing window: {self.smooth_window} frames")

    def _center_crop_square(self, img_bgr: np.ndarray) -> np.ndarray:
        """Simple square center crop (assumes face roughly centered)."""
        h, w, _ = img_bgr.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        return img_bgr[y0:y0 + side, x0:x0 + side]

    @torch.inference_mode()
    def _extract_feature(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Extract DINOv3 pooled feature vector from a BGR frame."""
        crop_bgr = self._center_crop_square(frame_bgr)
        img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=[img_rgb], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled = outputs.pooler_output.detach().cpu().numpy()[0]
        return pooled

    def predict(self, frame_bgr: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Predict emotion from a single BGR frame (with temporal smoothing).

        Returns:
            label_str: smoothed label ("excited", "nervous", "confused")
            prob_dict: class_name -> smoothed probability
        """
        feat = self._extract_feature(frame_bgr)
        feat_2d = feat.reshape(1, -1)

        if hasattr(self.svm, "predict_proba"):
            probs = self.svm.predict_proba(feat_2d)[0]
        else:
            raw = self.svm.decision_function(feat_2d)[0]
            exps = np.exp(raw - np.max(raw))
            probs = exps / np.sum(exps)

        # Update buffer and compute smoothed probs
        self.prob_buffer.append(probs)
        avg_probs = probs if len(self.prob_buffer) == 1 else np.mean(
            np.stack(list(self.prob_buffer), axis=0), axis=0
        )

        smoothed_id = int(np.argmax(avg_probs))
        label_str = self.label_encoder.inverse_transform([smoothed_id])[0]

        prob_dict = {
            cls_name: float(p)
            for cls_name, p in zip(self.label_encoder.classes_, avg_probs)
        }
        return label_str, prob_dict


class EmotionDetector:
    """
    High-level wrapper used by main.py.

    - Opens webcam
    - Each call to get_emotion() grabs a few frames and returns the current
      smoothed emotion label (string), and optionally the last frame.
    """

    def __init__(
        self,
        model_path: str = "dinov3_svm_3class.joblib",
        smooth_window: int = 15,
        camera_index: int = 0,
        frames_per_call: int = 5,
    ):
        self.perception = EmotionPerception(model_path=model_path, smooth_window=smooth_window)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {camera_index}.")
        self.frames_per_call = max(1, frames_per_call)
        print(f"[EmotionDetector] Webcam opened at index {camera_index}.")

    def get_emotion(self, return_frame: bool = False) -> (str, Optional[np.ndarray]):
        """
        Capture a few frames, run through EmotionPerception, and return label.
        If return_frame=True, also return the last captured frame (BGR).

        Returns:
            label: "excited", "nervous", "confused"
            frame: np.ndarray or None (if return_frame=True)
        """
        last_label = "confused"
        last_frame = None

        for _ in range(self.frames_per_call):
            ret, frame = self.cap.read()
            if not ret:
                continue

            label, _ = self.perception.predict(frame)
            last_label = label
            last_frame = frame

        if return_frame:
            return last_label, last_frame
        else:
            return last_label, None

    def close(self):
        """Release webcam and destroy any OpenCV windows."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[EmotionDetector] Webcam released.")
        cv2.destroyAllWindows()

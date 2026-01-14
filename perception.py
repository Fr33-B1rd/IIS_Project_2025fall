#!/usr/bin/env python
"""
perception.py

Robust emotion perception for Furhat TRPG DM.

Fixes implemented:
- No long temporal smoothing inside the vision model (prevents double-smoothing lag).
- Stable face ROI using OpenCV face detection + EMA-smoothed bounding box.
- Emotion is computed from multiple frames (average probabilities).
- Confidence gating (p_max + margin) to produce "uncertain" when not confident.
- Supports sampling during ASR listening via a continuous capture loop (main.py can run it in a thread).

Model:
- DINOv3 ViT-S/16 features (HuggingFace)
- 3-class SVM (excited / nervous / confused)

Requires:
  - dinov3_svm_3class_mydata.joblib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable
from pathlib import Path
import threading
import time

import numpy as np
import cv2
import joblib
from transformers import AutoImageProcessor, AutoModel
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None


# ----------------------------
# Face detection utilities
# ----------------------------
_HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_FACE_CASCADE = cv2.CascadeClassifier(_HAAR_PATH)


@dataclass
class CaptureConfig:
    fps: float = 10.0
    max_frames: int = 80  # keep ~8s at 10 fps


class EmotionPerception:
    def __init__(
        self,
        model_path: str = "dinov3_svm_3class_mydata.joblib",
        pretrained_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        face_margin: float = 0.15,
        bbox_ema_alpha: float = 0.35,
        input_size: int = 224,
        prob_temperature: float = 1.2,
    ):
        self.model_path = str(model_path)
        self.face_margin = float(face_margin)
        self.bbox_ema_alpha = float(bbox_ema_alpha)
        self.input_size = int(input_size)
        self.prob_temperature = float(prob_temperature)

        # bbox EMA state (x, y, w, h)
        self._bbox_ema: Optional[np.ndarray] = None

        # Load label encoder + svm
        bundle = joblib.load(self.model_path)
        # Expect training bundle to include these keys
        self.svm = bundle["svm"]
        self.label_encoder = bundle["label_encoder"]

        print(f"[EmotionPerception] Loading DINOv3 backbone: {pretrained_model_name}")
        try:
            # Prefer local cache to avoid demo-day network/auth issues
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto", dtype="auto", local_files_only=True)
        except Exception:
            # Fallback to normal download (may require HF auth for gated models)
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
            self.model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto", dtype="auto")
        self.model.eval()

        print("[EmotionPerception] Classes:", list(self.label_encoder.classes_))
        self._mp_face = None
        if mp is not None:
            try:
                self._mp_face = mp.solutions.face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                )
            except Exception:
                self._mp_face = None

    # ---------- ROI ----------
    def _detect_face_bbox(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Return (x, y, w, h) for the largest detected face, or None."""
        if self._mp_face is not None:
            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp_face.process(img_rgb)
            if results and results.detections:
                best = None
                best_conf = 0.0
                for det in results.detections:
                    conf = float(det.score[0]) if det.score else 0.0
                    if conf < 0.5:
                        continue
                    bbox = det.location_data.relative_bounding_box
                    x0 = int(bbox.xmin * w)
                    y0 = int(bbox.ymin * h)
                    bw = int(bbox.width * w)
                    bh = int(bbox.height * h)
                    x0 = max(0, x0)
                    y0 = max(0, y0)
                    bw = max(1, min(w - x0, bw))
                    bh = max(1, min(h - y0, bh))
                    if conf > best_conf:
                        best_conf = conf
                        best = (x0, y0, bw, bh)
                if best is not None:
                    return best

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return int(x), int(y), int(w), int(h)

    def _ema_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """EMA-smooth bbox to reduce jitter across frames."""
        b = np.array(bbox, dtype=np.float32)
        if self._bbox_ema is None:
            self._bbox_ema = b
        else:
            a = self.bbox_ema_alpha
            self._bbox_ema = a * b + (1 - a) * self._bbox_ema
        x, y, w, h = self._bbox_ema
        return int(round(x)), int(round(y)), int(round(w)), int(round(h))

    def _crop_face_with_margin(self, img_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        H, W = img_bgr.shape[:2]
        # expand bbox
        mx = int(w * self.face_margin)
        my = int(h * self.face_margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(W, x + w + mx)
        y1 = min(H, y + h + my)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return self._center_crop_square(img_bgr)
        return crop

    def _center_crop_square(self, img_bgr: np.ndarray) -> np.ndarray:
        H, W = img_bgr.shape[:2]
        s = min(H, W)
        y0 = (H - s) // 2
        x0 = (W - s) // 2
        return img_bgr[y0:y0 + s, x0:x0 + s]

    # ---------- Features ----------
    def _extract_feature(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        bbox = self._detect_face_bbox(frame_bgr)
        if bbox is not None:
            bbox = self._ema_bbox(bbox)
            img_bgr = self._crop_face_with_margin(frame_bgr, bbox)
        else:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)

        inputs = self.processor(images=img_rgb, return_tensors="pt")
        # Move to model device
        for k in inputs:
            inputs[k] = inputs[k].to(self.model.device)

        with torch.no_grad():
                out = self.model(**inputs)
                # IMPORTANT: match training script (uses outputs.pooler_output)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feat = out.pooler_output.detach().float().cpu().numpy()[0]
                elif hasattr(out, "last_hidden_state"):
                    hs = out.last_hidden_state  # (B, T, D)
                    feat = hs[:, 0, :].detach().float().cpu().numpy()[0]
                else:
                    raise RuntimeError("Model output has neither pooler_output nor last_hidden_state.")

        return feat

    def predict_frame(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        """Return per-class probabilities for one frame (no temporal smoothing)."""
        feat = self._extract_feature(frame_bgr)
        if feat is None:
            raise ValueError("No face detected.")
        feat = feat.reshape(1, -1)

        if hasattr(self.svm, "predict_proba"):
            probs = self.svm.predict_proba(feat)[0]
        else:
            raw = self.svm.decision_function(feat)[0]
            exps = np.exp(raw - np.max(raw))
            probs = exps / np.sum(exps)

        probs = np.asarray(probs, dtype=np.float32)
        if self.prob_temperature > 1e-6 and self.prob_temperature != 1.0:
            probs = np.power(probs, 1.0 / self.prob_temperature)
            probs = probs / np.sum(probs)
        prob_dict = {cls: float(p) for cls, p in zip(self.label_encoder.classes_, probs)}
        return prob_dict

    def predict_frames(self, frames_bgr: Iterable[np.ndarray]) -> Tuple[str, Dict[str, float]]:
        """Average probabilities over multiple frames and return (label, prob_dict)."""
        prob_acc = None
        n = 0
        for fr in frames_bgr:
            try:
                pd = self.predict_frame(fr)
            except Exception:
                continue
            vec = np.array([pd[c] for c in self.label_encoder.classes_], dtype=np.float32)
            prob_acc = vec if prob_acc is None else (prob_acc + vec)
            n += 1

        if n == 0 or prob_acc is None:
            # total fallback
            neutral = {c: 1.0 / len(self.label_encoder.classes_) for c in self.label_encoder.classes_}
            return "uncertain", neutral

        prob_avg = (prob_acc / n).astype(np.float32)
        prob_dict = {c: float(p) for c, p in zip(self.label_encoder.classes_, prob_avg)}
        label = max(prob_dict.items(), key=lambda kv: kv[1])[0]
        return label, prob_dict


# torch is only needed after model load; keep import here to avoid heavy import time in some environments
import torch  # noqa: E402


class EmotionDetector:
    """
    Webcam wrapper.

    Key APIs:
      - get_emotion_snapshot(): capture N frames quickly and estimate emotion.
      - start_buffering()/stop_buffering(): continuous capture for "during listening" sampling.
      - get_emotion_from_buffer(): compute emotion from buffered frames.
    """

    def __init__(
        self,
        model_path: str = "dinov3_svm_3class_mydata.joblib",
        camera_index: int = 0,
        frames_per_call: int = 5,
        pmax_threshold: float = 0.45,
        margin_threshold: float = 0.10,
        pretrained_model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    ):
        # Keep backbone consistent with training (see train_classifier.py)
        self.perception = EmotionPerception(model_path=model_path, pretrained_model_name=pretrained_model_name)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"[EmotionDetector] Could not open webcam index {camera_index}")

        self.frames_per_call = int(frames_per_call)
        self.pmax_threshold = float(pmax_threshold)
        self.margin_threshold = float(margin_threshold)

        # Continuous buffering
        self._buffer_lock = threading.Lock()
        self._buffer: List[np.ndarray] = []
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._cap_cfg = CaptureConfig()
        self._latest_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

    # ---------- confidence gate ----------
    def _apply_confidence_gate(self, prob_dict: Dict[str, float]) -> str:
        items = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_p = items[0]
        second_p = items[1][1] if len(items) > 1 else 0.0
        margin = top_p - second_p
        if top_p < self.pmax_threshold or margin < self.margin_threshold:
            return "uncertain"
        return top_label

    # ---------- snapshot ----------
    def get_emotion_snapshot(self) -> Tuple[str, Dict[str, float]]:
        frames = []
        for _ in range(self.frames_per_call):
            ok, fr = self.cap.read()
            if not ok:
                continue
            with self._latest_lock:
                self._latest_frame = fr
            frames.append(fr)

        label, prob_dict = self.perception.predict_frames(frames)
        gated = self._apply_confidence_gate(prob_dict)
        return gated, prob_dict

    # ---------- buffering (for ASR window) ----------
    def start_buffering(self, fps: float = 10.0, max_frames: int = 80):
        """Start continuous frame capture in a background thread."""
        self._cap_cfg = CaptureConfig(fps=float(fps), max_frames=int(max_frames))
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        with self._buffer_lock:
            self._buffer = []

        def _loop():
            delay = 1.0 / max(1e-6, self._cap_cfg.fps)
            while self._stop_event is not None and not self._stop_event.is_set():
                ok, fr = self.cap.read()
                if ok:
                    with self._latest_lock:
                        self._latest_frame = fr
                    with self._buffer_lock:
                        self._buffer.append(fr)
                        # cap the buffer size
                        if len(self._buffer) > self._cap_cfg.max_frames:
                            self._buffer = self._buffer[-self._cap_cfg.max_frames :]
                time.sleep(delay)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop_buffering(self) -> List[np.ndarray]:
        """Stop capture loop and return buffered frames (may be empty)."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._stop_event = None
        self._thread = None
        with self._buffer_lock:
            frames = list(self._buffer)
            self._buffer = []
        return frames

    def get_emotion_from_buffer(self) -> Tuple[str, Dict[str, float]]:
        frames = self.stop_buffering()
        label, prob_dict = self.perception.predict_frames(frames)
        gated = self._apply_confidence_gate(prob_dict)
        return gated, prob_dict

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        """Return the most recent frame captured by this detector."""
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy() if copy else self._latest_frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[EmotionDetector] Webcam released.")
        cv2.destroyAllWindows()

#!/usr/bin/env python
"""
Train and evaluate a 3-class (excited / nervous / confused)
DINOv3 + SVM classifier on DiffusionFER DifussionEmotion_S.
Dataset folder format:
DifussionEmotion_S/
    angry/
    disgust/
    fear/
    happy/
    neutral/
    sad/
    surprise/
"""

import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import torch
from transformers import AutoImageProcessor, AutoModel

# ============================================================
# Label mapping to 3 classes
# ============================================================
FER_TO_PROJECT = {
    "happy": "excited",
    "surprise": "excited",
    "fear": "nervous",
    "sad": "nervous",
    "angry": "nervous",
    "disgust": "nervous",
    "neutral": "confused",
}

CLASSES_3 = ["excited", "nervous", "confused"]

PRETRAINED_MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


# ============================================================
# Dataset loading
# ============================================================
def load_folder_dataset(root_dir: str):
    images_bgr: List[np.ndarray] = []
    labels: List[str] = []

    root = Path(root_dir)
    for class_name in os.listdir(root):
        subdir = root / class_name
        if not subdir.is_dir():
            continue
        for fname in os.listdir(subdir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = subdir / fname
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                images_bgr.append(img)
                labels.append(class_name.lower())

    print(f"Loaded {len(images_bgr)} images from {root_dir}")
    return images_bgr, labels


# ============================================================
# DINOv3 Feature Extraction
# ============================================================
def init_dinov3():
    processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME, device_map="auto", dtype="auto")
    model.eval()
    return processor, model


@torch.inference_mode()
def extract_features(images_bgr: List[np.ndarray], processor, model, batch_size=8):
    feats = []
    n = len(images_bgr)
    print(f"Extracting DINOv3 features for {n} images...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = images_bgr[start:end]
        batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch]
        inputs = processor(images=batch_rgb, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)
        pooled = outputs.pooler_output.detach().cpu().numpy()
        feats.append(pooled)

        if (start // batch_size) % 10 == 0:
            print(f"Processed {end}/{n}")

    features = np.concatenate(feats, axis=0)
    print("Feature shape:", features.shape)
    return features


# ============================================================
# Main training function
# ============================================================
def train_3class(dataset_root: str, test_size=0.2, seed=42):
    imgs, labels_7 = load_folder_dataset(dataset_root)

    # Remap to 3-class labels
    features_to_use = []
    labels_3 = []
    kept_imgs = []

    for img, lab in zip(imgs, labels_7):
        if lab in FER_TO_PROJECT:
            labels_3.append(FER_TO_PROJECT[lab])
            kept_imgs.append(img)

    print(f"Using {len(labels_3)} images after label remap.")

    processor, model = init_dinov3()

    features = extract_features(kept_imgs, processor, model)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels_3)
    print("Classes:", list(le.classes_))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train SVM
    svm = SVC(probability=True)
    svm.fit(X_train, y_train)

    # Eval
    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy (3-class): {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save model
    out_path = Path("dinov3_svm_3class.joblib")
    joblib.dump({"svm": svm, "label_encoder": le}, out_path)
    print(f"Model saved → {out_path}")


if __name__ == "__main__":
    train_3class(dataset_root="./dataset/DiffusionFER/DiffusionEmotion_S/original")

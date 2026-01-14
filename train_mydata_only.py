#!/usr/bin/env python
"""
Train a 3-class (excited / nervous / confused)
DINOv3 + SVM classifier using ONLY personal data.

Expected dataset folder format:
dataset/mydata/
    excited/
    nervous/
    confused/
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import torch
from transformers import AutoImageProcessor, AutoModel


CLASSES_3 = ["excited", "nervous", "confused"]
PRETRAINED_MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"


def load_mydata_dataset(root_dir: str):
    images_bgr: List[np.ndarray] = []
    labels: List[str] = []

    root = Path(root_dir)
    for class_name in root.iterdir():
        if not class_name.is_dir():
            continue
        class_lower = class_name.name.lower()
        if class_lower not in CLASSES_3:
            continue
        for img_path in class_name.iterdir():
            if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            images_bgr.append(img)
            labels.append(class_lower)

    print(f"Loaded {len(images_bgr)} images from {root_dir}")
    return images_bgr, labels


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


def oversample_minority(images: List[np.ndarray], labels: List[str], seed: int = 42) -> Tuple[List[np.ndarray], List[str]]:
    rng = np.random.default_rng(seed)
    class_to_indices = {}
    for i, lab in enumerate(labels):
        class_to_indices.setdefault(lab, []).append(i)

    if not class_to_indices:
        return images, labels

    max_count = max(len(idxs) for idxs in class_to_indices.values())
    new_images: List[np.ndarray] = []
    new_labels: List[str] = []

    for lab, idxs in class_to_indices.items():
        if not idxs:
            continue
        chosen = rng.choice(idxs, size=max_count, replace=True)
        for i in chosen:
            new_images.append(images[i])
            new_labels.append(lab)

    return new_images, new_labels


def plot_tsne(features: np.ndarray, labels: List[str], out_path: Path, seed: int = 42) -> None:
    tsne = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=seed)
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(7, 6))
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        idx = [i for i, v in enumerate(labels) if v == lab]
        pts = emb[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.75, label=lab)
    plt.legend(frameon=False, fontsize=9)
    plt.title("t-SNE of DINOv3 Features (MyData)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train_mydata_only(dataset_root: str, test_size=0.2, seed=42, visualize_tsne: bool = True):
    imgs, labels = load_mydata_dataset(dataset_root)
    if not imgs:
        raise RuntimeError("No images found in personal dataset.")

    processor, model = init_dinov3()
    features = extract_features(imgs, processor, model)
    if visualize_tsne:
        tsne_path = Path("tsne_mydata.png")
        plot_tsne(features, labels, tsne_path, seed=seed)
        print(f"t-SNE saved -> {tsne_path}")

    # Oversample minority classes for balance
    # imgs, labels = oversample_minority(imgs, labels, seed=seed)
    # features = extract_features(imgs, processor, model)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    print("Classes:", list(le.classes_))

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # svm = SVC(probability=True)
    # 增加 C 值，让权重的影响更显著
    class_weight_map = {}
    for idx, name in enumerate(le.classes_):
        if name == "nervous":
            class_weight_map[idx] = 1.0
        elif name == "excited":
            class_weight_map[idx] = 1.0
        else:
            class_weight_map[idx] = 1.0
    print("Class weights:", {le.classes_[k]: v for k, v in class_weight_map.items()})

    svm = SVC(probability=True, C=1.0, class_weight=class_weight_map)
    svm.fit(X_train, y_train)

    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy (mydata only, 3-class): {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

    out_path = Path("dinov3_svm_3class_mydata.joblib")
    joblib.dump({"svm": svm, "label_encoder": le}, out_path)
    print(f"Model saved -> {out_path}")


if __name__ == "__main__":
    train_mydata_only(dataset_root="./dataset/mydata", visualize_tsne=True)

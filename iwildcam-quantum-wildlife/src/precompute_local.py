"""
precompute_local.py
--------------------
Runs YOLO -> ResNet feature extraction on the locally available labeled images
(created by build_local_dataset.py) and saves:
  - precomputed/features.npy   (N, reduced_dim)
  - precomputed/labels.npy     (N,)
  - precomputed/pca_model.pkl

Falls back to full image if YOLO finds no crops.

Usage:
    python -m src.precompute_local
"""

import os
import csv
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import yaml
import joblib

from src.feature_extractor import ResNetFeatureExtractor, get_transform
from src.yolo_infer import YOLOCropper
from src.pca_utils import fit_and_save_pca
from src.utils import ensure_dir

# ── Config ────────────────────────────────────────────────────────────────────
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CSV_PATH   = os.path.join("data", "iwildcam_v2.0", "image_labels.csv")
OUT_DIR    = ensure_dir(cfg["paths"]["precomputed_dir"])
PCA_PATH   = os.path.join(OUT_DIR, "pca_model.pkl")
FEAT_PATH  = os.path.join(OUT_DIR, "features.npy")
LABEL_PATH = os.path.join(OUT_DIR, "labels.npy")
CROPS_DIR  = ensure_dir(cfg["paths"]["crops_dir"])

REDUCED_DIM = cfg["model"]["reduced_dim"]       # 16
RESNET_DIM  = cfg["model"]["resnet_output_dim"]  # 512

# Use CPU to be safe; change to "cuda" if GPU available
DEVICE = torch.device("cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["filepath"], int(row["label"])))
    return rows


def extract_feature(img_path, feat_model, yolo, transform, idx):
    """
    Tries YOLO crop first; if no crop, uses the full image.
    Returns the 512-dim ResNet feature vector or None on error.
    """
    # YOLO crop
    try:
        crops = yolo.crop_from_image_path(img_path, output_dir=CROPS_DIR)
    except Exception:
        crops = []

    # Pick largest crop by area
    best = None
    best_area = -1
    for c in crops:
        try:
            im = Image.open(c)
            area = im.size[0] * im.size[1]
            if area > best_area:
                best_area = area
                best = c
        except Exception:
            continue

    source = best if best else img_path  # fallback to full image

    try:
        img = Image.open(source).convert("RGB")
        x = transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = feat_model(x).cpu().numpy().squeeze(0)  # (512,)
        return feat
    except Exception as e:
        print(f"  [skip] {os.path.basename(img_path)}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        print("   Run first: python -m src.build_local_dataset")
        return

    rows = load_csv(CSV_PATH)
    print(f"✔ Loaded {len(rows)} labeled images from CSV")

    # Init models
    print("⚡ Loading ResNet-50 …")
    feat_model = ResNetFeatureExtractor(output_dim=RESNET_DIM).to(DEVICE).eval()

    print("⚡ Loading YOLOv8n …")
    yolo = YOLOCropper(
        weights_path=cfg["paths"]["yolov8_weights"],
        conf_thresh=0.20,
        device="cpu"
    )

    transform = get_transform()

    raw_features = []
    labels       = []

    print(f"\n🚀 Extracting features from {len(rows)} images …\n")
    for img_path, label in tqdm(rows):
        if not os.path.exists(img_path):
            continue
        feat = extract_feature(img_path, feat_model, yolo, transform, label)
        if feat is not None:
            raw_features.append(feat)
            labels.append(label)

    raw_features = np.array(raw_features, dtype=np.float32)
    labels       = np.array(labels,       dtype=np.int64)

    print(f"\n✨ Extracted {len(raw_features)} feature vectors")
    print(f"   Feature matrix: {raw_features.shape}")
    print(f"   Feature std:    {raw_features.std():.4f}  (>0.1 means real features)")
    print(f"   Classes:        {len(np.unique(labels))}")

    if len(raw_features) < 10:
        print("❌ Too few samples. Aborting.")
        return

    # PCA reduction
    n_components = min(REDUCED_DIM, raw_features.shape[0] - 1, raw_features.shape[1])
    print(f"\n📉 Fitting PCA: {raw_features.shape[1]} → {n_components} dims …")
    pca = fit_and_save_pca(raw_features, n_components=n_components, save_path=PCA_PATH)
    reduced = pca.transform(raw_features)

    # Save
    np.save(FEAT_PATH,  reduced)
    np.save(LABEL_PATH, labels)

    print(f"\n✅ Precomputation complete!")
    print(f"   features.npy  → {FEAT_PATH}  shape={reduced.shape}")
    print(f"   labels.npy    → {LABEL_PATH}  shape={labels.shape}")
    print(f"   pca_model.pkl → {PCA_PATH}")
    print(f"\n   Next: python -m src.train_mlp_local")


if __name__ == "__main__":
    run()

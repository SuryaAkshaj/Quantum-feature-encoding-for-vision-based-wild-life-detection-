"""
train_mlp_local.py
-------------------
Trains the MLP baseline on the REAL precomputed features produced by
precompute_local.py. Saves the trained model to models/mlp_baseline.pth.

Usage:
    python -m src.train_mlp_local
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.baseline_mlp import MLPBaseline

FEAT_PATH   = "precomputed/features.npy"
LABEL_PATH  = "precomputed/labels.npy"
MODEL_PATH  = "models/mlp_baseline.pth"
NAMES_PATH       = "models/class_names.json"        # original, never overwritten
MODEL_NAMES_PATH = "models/class_names_model.json"  # remapped inference labels

BATCH_SIZE  = 16   # smaller batch since fewer samples after filtering
EPOCHS      = 30   # more epochs for smaller dataset
LR          = 1e-3
VAL_SPLIT   = 0.2
DEVICE      = torch.device("cpu")

# ── Classes to EXCLUDE from training ────────────────────────────────────────
# "background" dominates (444/669 = 66%) and causes bias → exclude it
EXCLUDE_CLASS_NAMES = {"background", "empty"}  # both mean "no animal visible"


def load_class_names():
    if not os.path.exists(NAMES_PATH):
        return {}
    with open(NAMES_PATH) as f:
        return json.load(f)


def train():
    # ── Load data ────────────────────────────────────────────────────────────
    if not os.path.exists(FEAT_PATH) or not os.path.exists(LABEL_PATH):
        print("Precomputed features not found.")
        print("   Run first: python -m src.precompute_local")
        return

    X = np.load(FEAT_PATH)
    y = np.load(LABEL_PATH)
    class_names = load_class_names()

    print(f"Loaded features: {X.shape}   std: {X.std():.4f}")

    # ── Filter out excluded classes (e.g. "background") ────────────────────
    keep_mask = np.ones(len(y), dtype=bool)
    for i, label in enumerate(y):
        name = class_names.get(str(label), f"class_{label}")
        if name in EXCLUDE_CLASS_NAMES:
            keep_mask[i] = False

    X = X[keep_mask]
    y = y[keep_mask]
    print(f"After excluding {EXCLUDE_CLASS_NAMES}: {len(X)} samples remain")

    unique_classes = sorted(np.unique(y))
    n_classes  = len(unique_classes)
    input_dim  = X.shape[1]

    # Dense remap → [0, n_classes)
    label_remap = {old: new for new, old in enumerate(unique_classes)}
    y_dense     = np.array([label_remap[yi] for yi in y], dtype=np.int64)

    # Build filtered class names and save
    filtered_names = {}
    print(f"\nTraining classes ({n_classes}):")
    for new_id, old_id in enumerate(unique_classes):
        name = class_names.get(str(old_id), f"class_{old_id}")
        filtered_names[str(new_id)] = name
        count = (y_dense == new_id).sum()
        print(f"   [{new_id}] {name:20s}  {count} samples")

    # Save REMAPPED names to a separate model file — never touch original class_names.json
    os.makedirs("models", exist_ok=True)
    with open(MODEL_NAMES_PATH, "w") as f:
        json.dump(filtered_names, f, indent=2)
    print(f"\nSaved class_names_model.json ({n_classes} animal classes)")

    # ── Class weights (inverse frequency) to handle imbalance ───────────────
    counts  = np.bincount(y_dense, minlength=n_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * n_classes   # normalize
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {[round(w, 2) for w in weights.tolist()]}")

    # ── Train / val split ────────────────────────────────────────────────────
    can_stratify = all((y_dense == c).sum() >= 2 for c in range(n_classes))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_dense,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y_dense if can_stratify else None,
    )

    train_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long)),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                                        torch.tensor(y_val,   dtype=torch.long)),
                          batch_size=BATCH_SIZE)

    print(f"\nTrain: {len(X_train)}   Val: {len(X_val)}\n")

    # ── Model ────────────────────────────────────────────────────────────────
    model     = MLPBaseline(input_dim=input_dim, n_classes=n_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state   = None

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Acc':>7}")
    print("-" * 42)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = t_correct = t_total = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * xb.size(0)
            t_correct += (model(xb).argmax(1) == yb).sum().item()
            t_total   += xb.size(0)

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_correct += (model(xb).argmax(1) == yb).sum().item()
                v_total   += xb.size(0)

        t_acc = t_correct / t_total
        v_acc = v_correct / v_total
        print(f"{epoch:>6}  {t_loss/t_total:>10.4f}  {t_acc:>9.4f}  {v_acc:>7.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Model saved: {MODEL_PATH}")

    # ── Report ───────────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            all_preds.extend(model(xb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    target_names = [filtered_names.get(str(i), f"class_{i}") for i in range(n_classes)]
    print("\nValidation Classification Report:")
    print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))
    print("Next: Restart Flask -> python -m web.app_flask")


if __name__ == "__main__":
    train()

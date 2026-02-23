"""
Flask API: upload image, run YOLO crop -> feature extractor -> MLP classifier -> return JSON
"""
from flask import Flask, request, jsonify
import os
import json
import torch
import numpy as np

from src.yolo_infer import YOLOCropper
from src.feature_extractor import ResNetFeatureExtractor, get_transform, load_image_tensor
from src.vqc_model import VQCClassifier
from src.baseline_mlp import MLPBaseline

app = Flask(__name__)
UPLOAD_DIR = "web/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Load class names ──────────────────────────────────────────────────────────
# train_mlp_local.py writes REMAPPED (model-output) names to class_names_model.json
# class_names.json = original YOLO labels (0=background, 1=bear …) — used for training only
CLASS_NAMES_PATH       = "models/class_names_model.json"  # preferred: remapped
CLASS_NAMES_PATH_FALLBACK = "models/class_names.json"     # fallback
_DEFAULT_LABELS  = {
    "0": "bear", "1": "bird", "2": "cat", "3": "cow", "4": "dog",
    "5": "elephant", "6": "giraffe", "7": "horse", "8": "sheep", "9": "zebra",
}

def load_class_names():
    for path in [CLASS_NAMES_PATH, CLASS_NAMES_PATH_FALLBACK]:
        if os.path.exists(path):
            with open(path) as f:
                names = json.load(f)
            # Skip if it contains background (means it's the original, not remapped)
            if "background" not in names.values() and "empty" not in names.values():
                return names
    print(f"⚠  No valid class_names_model.json found — using default 10-class labels.")
    return _DEFAULT_LABELS

CLASS_NAMES = load_class_names()
N_CLASSES   = len(CLASS_NAMES)


def label_for(class_id: int) -> str:
    return CLASS_NAMES.get(str(class_id), f"Species {class_id}")

# ── Load models ───────────────────────────────────────────────────────────────
device = torch.device("cpu")

feat_model = ResNetFeatureExtractor(output_dim=512)
feat_model.eval()

yolo_cropper = YOLOCropper(weights_path="yolov8n.pt", conf_thresh=0.25, device="cpu")


# ── Detect input_dim from saved model (adapts to any PCA dimension) ───────────
def get_input_dim(model_path: str, fallback: int = 128) -> int:
    if not os.path.exists(model_path):
        return fallback
    state = torch.load(model_path, map_location="cpu")
    # First layer weight has shape [hidden, input_dim]
    first_w = [v for k, v in state.items() if "weight" in k]
    return int(first_w[0].shape[1]) if first_w else fallback

MLP_PATH   = "models/mlp_baseline.pth"
INPUT_DIM  = get_input_dim(MLP_PATH)

mlp = MLPBaseline(input_dim=INPUT_DIM, n_classes=N_CLASSES)
vqc = VQCClassifier(n_qubits=4, n_layers=3, n_outputs=N_CLASSES, input_dim=INPUT_DIM)

if os.path.exists(MLP_PATH):
    try:
        mlp.load_state_dict(torch.load(MLP_PATH, map_location=device))
        mlp.eval()
        print(f"✔ MLP loaded  ({N_CLASSES} classes, input_dim={INPUT_DIM})")
    except Exception as e:
        print("Could not load mlp:", e)
else:
    print("⚠  models/mlp_baseline.pth not found — using random weights")


if os.path.exists("models/vqc_real.pth"):
    try:
        vqc.load_state_dict(torch.load("models/vqc_real.pth", map_location=device))
        vqc.eval()
        print("✔ VQC loaded")
    except Exception as e:
        print("Could not load vqc:", e)

# ── ImageNet zero-shot classifier (primary predictor) ────────────────────────
from src.imagenet_classifier import ImageNetClassifier
imagenet_clf = ImageNetClassifier(top_k=15, device="cpu")
print("✔ ImageNet classifier ready (ResNet-50, 10 COCO classes)")


def reduce_features(feat_np: np.ndarray) -> np.ndarray:
    """Reduce 512-dim feature to 16-dim using PCA or simple slicing."""
    if pca is not None:
        return pca.transform(feat_np.reshape(1, -1)).squeeze(0).astype(np.float32)
    return feat_np[:16].astype(np.float32)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Flask API is running",
                    "n_classes": N_CLASSES, "classes": CLASS_NAMES})


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "no file"}), 400

    save_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(save_path)

    crops = yolo_cropper.crop_from_image_path(save_path, output_dir="web/crops")
    using_full_image = len(crops) == 0
    if using_full_image:
        crops = [save_path]

    results = []

    for c in crops:
        # ── Primary: ImageNet zero-shot via ResNet-50 classification head ────
        all_scores   = imagenet_clf.classify_all(c)     # {class: prob%}
        top_label    = max(all_scores, key=all_scores.get)
        confidence   = all_scores[top_label] / 100.0    # back to [0,1]

        # Build class_scores in standard format
        label_to_id = {v: int(k) for k, v in CLASS_NAMES.items()}
        class_scores = sorted(
            [{"class_id": label_to_id.get(lbl, i),
              "label":    lbl,
              "probability": round(score / 100.0, 4)}
             for i, (lbl, score) in enumerate(all_scores.items())],
            key=lambda x: x["probability"], reverse=True,
        )

        results.append({
            "crop_file":       os.path.basename(c),
            "predicted_label": top_label,
            "confidence":      round(confidence, 4),
            "yolo_crop_used":  not using_full_image,
            "class_scores":    class_scores,
        })

    return jsonify({"predictions": results})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

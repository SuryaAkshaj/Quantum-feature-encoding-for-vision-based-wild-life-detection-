# 🦁 Quantum-Enhanced Wildlife Detection (iWildCam)

An end-to-end **hybrid quantum-classical machine learning pipeline** for wildlife species classification from camera-trap images, combining YOLOv8, ResNet-50, and a Variational Quantum Circuit (VQC).

---

## 🔬 Overview

This project explores whether **quantum machine learning** can improve wildlife classification accuracy compared to classical neural networks. It processes camera-trap images through a complete detection → feature extraction → quantum classification pipeline.

### Pipeline
```
Camera Image → YOLOv8 Detection → Crop ROIs
    → ResNet-50 Features (512-dim)
    → PCA Reduction (16-dim)
    → VQC Classifier (PennyLane) OR Classical MLP
    → Species Probabilities
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | YOLOv8 (Ultralytics) |
| Feature Extraction | ResNet-50 (torchvision, ImageNet pretrained) |
| Dimensionality Reduction | PCA (scikit-learn) |
| Quantum Classifier | PennyLane VQC (4 qubits, 3 layers) |
| Classical Baseline | PyTorch MLP (3-layer) |
| Dataset | iWildCam v2.0 (WILDS) |
| API | Flask (REST, port 8000) |
| UI | Streamlit |

---

## 📁 Project Structure

```
iwildcam-quantum-wildlife/
├── src/
│   ├── feature_extractor.py     # ResNet-50 feature extractor
│   ├── vqc_model.py             # Quantum VQC classifier (PennyLane)
│   ├── baseline_mlp.py          # Classical MLP baseline
│   ├── yolo_infer.py            # YOLOv8 detection & cropping
│   ├── data_loader.py           # iWildCam WILDS dataset loader
│   ├── precompute_features.py   # Pre-extract & cache features
│   ├── train_vqc.py             # Train quantum model
│   ├── train_baseline.py        # Train classical MLP
│   ├── evaluate.py              # Metrics & evaluation
│   ├── pca_utils.py             # PCA dimensionality reduction
│   └── utils.py                 # Shared helpers (JSON, softmax, dirs)
├── web/
│   ├── app_flask.py             # Flask REST API
│   └── app_streamlit.py         # Streamlit demo UI
├── models/
│   ├── vqc_real.pth             # Trained quantum model weights
│   └── mlp_baseline.pth         # Trained MLP weights
├── precomputed/
│   ├── features.npy             # Cached ResNet features
│   ├── labels.npy               # Corresponding labels
│   └── pca_model.pkl            # Fitted PCA transform
├── configs/
│   └── config.yaml              # Model & training configuration
├── data/
│   └── iwildcam_v2.0/           # iWildCam dataset (WILDS)
├── yolov8n.pt                   # YOLOv8 nano weights
├── requirements.txt
└── setup.sh
```

---

## ⚡ Quickstart (Local Demo)

### 1. Set up environment

```bash
python -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Create sample data & train

```bash
# Generate synthetic sample data (for quick testing)
python create_sample_data.py

# Train quantum classifier
python -m src.train_vqc

# (Optional) Train classical baseline
python -m src.train_baseline
```

### 3. Start the Flask API

```bash
python -m web.app_flask
# API running at http://localhost:8000
```

### 4. Start the Streamlit UI (new terminal)

```bash
streamlit run web/app_streamlit.py
```

### 5. Use the app

- Open the Streamlit URL in your browser
- Upload a camera-trap image
- Click **Run prediction**
- View detected animals & species probabilities

---

## 🌐 API Reference

### `GET /`
Health check

```json
{ "status": "ok", "message": "Flask API is running" }
```

### `POST /predict`
Upload an image and get predictions.

**Request:** `multipart/form-data` with `image` field

**Response:**
```json
{
  "predictions": [
    {
      "crop": "image.jpg__0.jpg",
      "probs": [0.12, 0.45, 0.05, 0.30, 0.08]
    }
  ]
}s
```

Each entry in `probs` is the model's confidence for a species class (sums to 1.0).

---

## ⚛️ Quantum Circuit Design

```
Data Encoding:     RY(θ₀) RY(θ₁) RY(θ₂) RY(θ₃)
Variational Layer: RX RY RZ  →  CNOT chain  (×3 layers)
Measurement:       ⟨Z₀⟩ ⟨Z₁⟩ ⟨Z₂⟩ ⟨Z₃⟩
Classical Head:    Linear(4→64) → ReLU → Linear(64→classes)
```

- **4 qubits**, **3 variational layers**
- Angle encoding maps 16-dim PCA features → qubit rotations
- CNOT chain creates entanglement between qubits
- Trained end-to-end via PyTorch backpropagation through PennyLane

---

## 🏋️ Training with Real Data

```bash
# Precompute ResNet features from iWildCam dataset
python -m src.precompute_features

# Train quantum VQC on precomputed features
python -m src.train_vqc

# Evaluate on test set
python -m src.evaluate
```

> **Note:** The full iWildCam dataset is ~11 GB. The WILDS library handles auto-download when `download=True` is set in `data_loader.py`.

---

## ⚙️ Configuration

Edit `configs/config.yaml` to adjust model and training settings:

```yaml
model:
  resnet_output_dim: 512   # ResNet projection size
  reduced_dim: 16          # PCA output dimensions
  n_qubits: 4              # Quantum circuit qubits
  vqc_layers: 3            # Variational layer depth

training:
  epochs_baseline: 10
  epochs_vqc: 10
  lr_vqc: 1e-3
  device: "cpu"            # Change to "cuda" for GPU
```

---

## 📦 Requirements

```
torch>=2.0.0
torchvision
pennylane>=0.27
pennylane-lightning>=0.27
ultralytics>=8.0.0
wilds
flask
streamlit
scikit-learn
opencv-python
pandas
numpy
matplotlib
tqdm
pyyaml
```

---

## 🔬 Research Goal

> **Can a Variational Quantum Circuit (VQC) outperform a classical MLP for wildlife species classification on the same reduced feature set?**

Both models receive identical 16-dimensional PCA-reduced features extracted from ResNet-50. Results are compared using accuracy, precision, recall, and F1-score.

---

## 📄 License

This project is for research and educational purposes.

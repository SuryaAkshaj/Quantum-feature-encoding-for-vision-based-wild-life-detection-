"""
Helpers to fit, save, load PCA used in pipeline.
Uses joblib for serialization.
"""
from sklearn.decomposition import PCA
import joblib
import os
import numpy as np

def fit_and_save_pca(features: np.ndarray, n_components: int, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    joblib.dump(pca, save_path)
    return pca

def load_pca(save_path: str):
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"PCA file not found: {save_path}")
    return joblib.load(save_path)

def apply_pca(pca, features: np.ndarray):
    return pca.transform(features)

"""
ResNet feature extractor + optional PCA/linear reducer to `reduced_dim`.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import numpy as np
from sklearn.decomposition import PCA
import os
from PIL import Image

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # remove fc layer
        modules = list(resnet.children())[:-1]  # remove classifier
        self.backbone = nn.Sequential(*modules)  # outputs [B, 2048, 1,1]
        self.out_dim = 2048
        self.proj = nn.Linear(self.out_dim, output_dim)
    def forward(self, x):
        x = self.backbone(x)  # [B,2048,1,1]
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        return x

def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# utility to reduce features with PCA (scikit-learn)
def fit_pca(features, reduced_dim=16, save_path=None):
    pca = PCA(n_components=reduced_dim)
    pca.fit(features)
    if save_path:
        import joblib
        joblib.dump(pca, save_path)
    return pca

def load_image_tensor(path, transform=None):
    transform = transform or get_transform()
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # 1,C,H,W

if __name__ == "__main__":
    # demo: extract feature from an image path
    import sys
    if len(sys.argv) < 2:
        print("usage: python feature_extractor.py image.jpg")
    else:
        fe = ResNetFeatureExtractor(output_dim=512)
        fe.eval()
        t = get_transform()
        x = load_image_tensor(sys.argv[1], transform=t)
        with torch.no_grad():
            f = fe(x)
            print("feature shape", f.shape)

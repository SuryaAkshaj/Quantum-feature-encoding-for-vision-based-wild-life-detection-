import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.vqc_model import VQCClassifier

def train():
    print("Loading REAL precomputed dataset...")
    X = np.load("precomputed/features.npy")
    y = np.load("precomputed/labels.npy")

    X = torch.tensor(X).float()
    y = torch.tensor(y).long()

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VQCClassifier(
        n_qubits=4,
        n_layers=3,
        n_outputs=len(np.unique(y)),
        input_dim=X.shape[1]
    )

    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        total = 0
        correct = 0
        total_loss = 0

        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        print(f"Epoch {epoch+1} | Loss {total_loss/total:.4f} | Acc {correct/total:.4f}")

    torch.save(model.state_dict(), "models/vqc_real.pth")
    print("✔ Saved: models/vqc_real.pth")

if __name__ == "__main__":
    train()

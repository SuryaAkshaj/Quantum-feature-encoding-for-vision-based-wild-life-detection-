"""
PennyLane VQC wrapped as a PyTorch Module using TorchLayer.
Angle encoding: map reduced_dim features -> rotations.
We measure PauliZ expectation for each readout wire and map to logits.
"""
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
from pennylane.qnn import TorchLayer
import math

class VQCClassifier(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3, n_outputs=5, input_dim=16, device="cpu"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.input_dim = input_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # number of circuit parameters
        self._init_qnode()
        # Create TorchLayer
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}  # 3 rotations per qubit per layer
        self.qlayer = TorchLayer(self.qnode, weight_shapes)
        # map qlayer output (n_qubits values) to class logits
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs)
        )

    def _angle_encoding(self, x):
        # x is a 1D np array of length input_dim
        # map first n_qubits values (or aggregate) -> rotation angles
        # Here, we fold input vector into n_qubits by simple folding/averaging
        folded = np.zeros(self.n_qubits, dtype=np.float64)
        for i in range(self.input_dim):
            folded[i % self.n_qubits] += float(x[i])
        # normalize
        norm = np.linalg.norm(folded + 1e-9)
        folded = folded / norm
        angles = folded * np.pi  # scale into [0,pi]
        return angles

    def _init_qnode(self):
        n_qubits = self.n_qubits
        n_layers = self.n_layers
        input_dim = self.input_dim
        dev = self.dev

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # inputs: shape (input_dim,)
            # weights: shape (n_layers, n_qubits, 3)
            # angle encode
            # convert torch to numpy if needed handled by TorchLayer
            # folding inside qnode expects numpy-like array
            angles = self._angle_encoding(inputs)
            # apply rotations
            for w in range(n_qubits):
                qml.RY(angles[w], wires=w)
            # variational layers
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RX(weights[layer, q, 0], wires=q)
                    qml.RY(weights[layer, q, 1], wires=q)
                    qml.RZ(weights[layer, q, 2], wires=q)
                # entangling chain
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
            # measurements: return expectation of PauliZ on each wire
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]
        self.qnode = circuit

    def forward(self, x):
        # x: [B, input_dim]
        batch = x.shape[0]
        outs = []
        for i in range(batch):
            qout = self.qlayer(x[i])
            outs.append(qout)
        qouts = torch.stack(outs, dim=0)  # [B, n_qubits]
        logits = self.classifier(qouts)
        return logits

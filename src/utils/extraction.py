import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class QuantumFeatureExtractor(nn.Module):
    def __init__(self, quantum_model, output_size=(25, 25), patch_size=4, n_qubits=4):
        super(QuantumFeatureExtractor, self).__init__()
        self.quantum_model = quantum_model 
        self.output_size = output_size
        self.patch_size = patch_size
        self.n_qubits = n_qubits

        self.circuit = self.quantum_model.define_circuit()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  
        x = self.quantum_model.quanvolution_batch(x) 
        x = torch.relu(x)
        x = self.flatten(x)
        return x

class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.resnet = original_model.resnet
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x
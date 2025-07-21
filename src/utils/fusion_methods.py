import numpy as np
import torch
from src.utils.methods import AttentionFusion, SelfAttentionBlock

class FusionMethods:
    def __init__(self, device, attention_dim):
        self.device = device
        self.attention_module = AttentionFusion(attention_dim).to(device)
        self.self_attention_module = SelfAttentionBlock(attention_dim).to(device)

    def fuse_concat(self, quantum_features, classical_features):
        return np.concatenate((quantum_features, classical_features), axis=1)

    def fuse_add(self, quantum_features, classical_features):
        return quantum_features + classical_features

    def fuse_dot(self, quantum_features, classical_features):
        return np.multiply(quantum_features, classical_features)

    def fuse_attention(self, quantum_features, classical_features):
        self.attention_module.train()
        return self.attention_module(torch.tensor(quantum_features, dtype=torch.float32).to(self.device),
                                      torch.tensor(classical_features, dtype=torch.float32).to(self.device)).cpu().numpy()

    def fuse_self_attention(self, quantum_features, classical_features):
        self.self_attention_module.train()
        return self.self_attention_module(torch.tensor(quantum_features, dtype=torch.float32).to(self.device),
                                          torch.tensor(classical_features, dtype=torch.float32).to(self.device)).cpu().numpy()

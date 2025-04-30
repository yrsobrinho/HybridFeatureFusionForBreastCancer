import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, input_dim):
        super(AttentionFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, quantum, classical):
        common_dim = min(quantum.shape[1], classical.shape[1])
        q = quantum[:, :common_dim]
        c = classical[:, :common_dim]

        inter = torch.cat([q, c], dim=1)
        attn = self.gate(inter)
        fused = q * attn + c * (1 - attn)
        return fused


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.shape[-1] ** 0.5))
        output = torch.matmul(attention_scores, V)
        return output


def fuse_features(quantum, classical, method='concat', attention_module=None):
    quantum = torch.tensor(quantum, dtype=torch.float32)
    classical = torch.tensor(classical, dtype=torch.float32)

    assert quantum.shape[0] == classical.shape[0], "Mismatch in sample count"

    if method == 'add':
        min_dim = min(quantum.shape[1], classical.shape[1])
        q = quantum[:, :min_dim]
        c = classical[:, :min_dim]
        fused = q + c

    elif method == 'concat':
        fused = torch.cat([quantum, classical], dim=1)

    elif method == 'attention' and attention_module is not None:
        fused = attention_module(quantum, classical)

    elif method == 'self_attention' and attention_module is not None:
        inter = torch.cat([quantum, classical], dim=1)
        fused = attention_module(inter)

    elif method == 'dot_product':
        min_dim = min(quantum.shape[1], classical.shape[1])
        q = quantum[:, :min_dim]
        c = classical[:, :min_dim]
        q_norm = F.normalize(q, p=2, dim=1)
        c_norm = F.normalize(c, p=2, dim=1)
        sim = torch.sum(q_norm * c_norm, dim=1, keepdim=True)  # [N, 1]
        weight = (sim + 1) / 2  # scale from [-1,1] to [0,1]
        fused = q * weight + c * (1 - weight)

    else:
        raise ValueError(f"Unsupported fusion method: {method}")

    return fused
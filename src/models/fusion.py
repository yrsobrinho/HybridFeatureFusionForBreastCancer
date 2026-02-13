import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    raise TypeError("Input must be a numpy array or torch tensor") 


def _match_dims(a, b):
    a = _to_numpy(a)
    b = _to_numpy(b)
    da = a.shape[1]
    db = b.shape[1]
    if da == db:
        return a, b
    m = max(da, db)
    if da < m:
        pad = np.zeros((a.shape[0], m - da), dtype=a.dtype)
        a = np.concatenate([a, pad], axis=1)
    if db < m:
        pad = np.zeros((b.shape[0], m - db), dtype=b.dtype)
        b = np.concatenate([b, pad], axis=1)
    return a, b


class FusionMethods:
    def __init__(self, device=None, attention_dim=None):
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.attention_dim = attention_dim
        self._proj = None
        self._proj_in = None
        self._att_mlp = None
        self._att_in = None

    def fuse_concat(self, q_feats, c_feats):
        q, c = _match_dims(q_feats, c_feats)
        return np.concatenate([q, c], axis=1)

    def fuse_add(self, q_feats, c_feats):
        q, c = _match_dims(q_feats, c_feats)
        return q + c

    def fuse_dot(self, q_feats, c_feats):

        q, c = _match_dims(q_feats, c_feats)
        return q * c

    def fuse_attention(self, q_feats, c_feats, eps=1e-8):
        q, c = _match_dims(q_feats, c_feats)
        num = np.sum(q * c, axis=1)
        q_norm = np.linalg.norm(q, axis=1)
        c_norm = np.linalg.norm(c, axis=1)
        denom = (q_norm * c_norm) + eps
        cos = num / denom
        alpha = 1.0 / (1.0 + np.exp(-cos))
        alpha = alpha.reshape(-1, 1)
        fused = alpha * q + (1 - alpha) * c
        return fused

    def fuse_self_attention(self, q_feats, c_feats, eps=1e-8):
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape
        fused_list = []
        for i in range(N):
            M = np.stack([q[i], c[i]], axis=0) 
            S = M @ M.T  
            S = S / (np.sqrt(D) + eps)
            expS = np.exp(S - np.max(S, axis=1, keepdims=True))
            W = expS / (np.sum(expS, axis=1, keepdims=True) + eps)  
            C = W @ M
            pooled = C.mean(axis=0)
            fused_list.append(pooled)
        fused = np.stack(fused_list, axis=0)
        return fused

    def _ensure_proj(self, D):
        if self.attention_dim is None:
            return None
        if (self._proj is None) or (self._proj_in != D):
            self._proj = nn.Linear(D, self.attention_dim).to(self.device)
            self._proj_in = D
        return self._proj

    def _ensure_att_mlp(self, D):
        if (self._att_mlp is None) or (self._att_in != D):
            hidden = max(64, D)  # simple heuristic
            self._att_mlp = nn.Sequential(
                nn.Linear(2 * D, hidden),
                nn.ReLU(),
                nn.Linear(hidden, D),
            ).to(self.device)
            self._att_in = D
        return self._att_mlp

    def _to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        raise TypeError("Input must be a numpy array or torch tensor")

    def fuse_learned_attention(self, q_feats, c_feats, per_feature=True):
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape

        tq = self._to_torch(q).float()
        tc = self._to_torch(c).float()

        if per_feature:
            mlp = self._ensure_att_mlp(D)
            inp = torch.cat([tq, tc], dim=1)  
            logits = mlp(inp)                 
            gates = torch.sigmoid(logits)     
            fused_t = gates * tq + (1.0 - gates) * tc
        else:
            mlp = self._ensure_att_mlp(D) 
            inp = torch.cat([tq, tc], dim=1)
            logits = mlp(inp)          
            scalar_gate = torch.sigmoid(logits.mean(dim=1, keepdim=True)) 
            fused_t = scalar_gate * tq + (1.0 - scalar_gate) * tc

        if self.attention_dim is not None:
            proj = self._ensure_proj(fused_t.shape[1])
            fused_t = proj(fused_t)

        return fused_t.detach().cpu().numpy()

    def fuse_cross_attention(self, q_feats, c_feats, eps=1e-8):
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape

        tM = torch.stack([self._to_torch(q), self._to_torch(c)], dim=1).float()

        Q = tM
        K = tM
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(D) + eps)
        attn = torch.softmax(scores, dim=-1)  

        V = tM
        contexts = torch.matmul(attn, V)  
 
        pooled = contexts.mean(dim=1) 

        if self.attention_dim is not None:
            proj = self._ensure_proj(pooled.shape[1])
            pooled = proj(pooled)

        return pooled.detach().cpu().numpy()

__all__ = ["FusionMethods"]

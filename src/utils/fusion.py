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
    """Pad the smaller feature vector in dim=1 with zeros so both have same width."""
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
    """Collection of fusion methods that operate on 2D feature arrays (N x D).

    Methods return a fused numpy array of shape (N, D') depending on the method.

    New features:
      - learned attention (fuse_learned_attention): small MLP computes per-feature gates
        to combine q and c; trainable if you keep the FusionMethods instance and optimize its params.
      - optional projection to attention_dim after fusion (creates a Linear layer lazily).
    """
    def __init__(self, device=None, attention_dim=None):
        # device can be 'cpu' or 'cuda' or a torch.device; used when creating torch modules
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.attention_dim = attention_dim
        # lazy torch modules (created when needed)
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
        # element-wise (Hadamard) product
        q, c = _match_dims(q_feats, c_feats)
        return q * c

    def fuse_attention(self, q_feats, c_feats, eps=1e-8):
        """Compute a scalar attention weight per sample from cosine similarity
        and return a convex combination of q and c: alpha * q + (1-alpha) * c
        """
        q, c = _match_dims(q_feats, c_feats)
        # cosine similarity per sample
        num = np.sum(q * c, axis=1)
        q_norm = np.linalg.norm(q, axis=1)
        c_norm = np.linalg.norm(c, axis=1)
        denom = (q_norm * c_norm) + eps
        cos = num / denom
        # map cos in [-1,1] to alpha in (0,1)
        alpha = 1.0 / (1.0 + np.exp(-cos))
        alpha = alpha.reshape(-1, 1)
        fused = alpha * q + (1 - alpha) * c
        return fused

    def fuse_self_attention(self, q_feats, c_feats, eps=1e-8):
        """Apply a tiny self-attention over the 2-token sequence [q, c].
        For each sample build a 2 x D matrix, compute attention scores, and
        return the pooled context vector (D,).
        """
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape
        fused_list = []
        for i in range(N):
            M = np.stack([q[i], c[i]], axis=0)  # (2, D)
            # similarity matrix
            S = M @ M.T  # (2,2)
            # scale
            S = S / (np.sqrt(D) + eps)
            # softmax rows
            expS = np.exp(S - np.max(S, axis=1, keepdims=True))
            W = expS / (np.sum(expS, axis=1, keepdims=True) + eps)  # (2,2)
            # context: W @ M -> (2, D)
            C = W @ M
            # pool contexts (mean over tokens)
            pooled = C.mean(axis=0)
            fused_list.append(pooled)
        fused = np.stack(fused_list, axis=0)
        return fused

    def _ensure_proj(self, D):
        """Create or reuse a projection Linear(D -> attention_dim)."""
        if self.attention_dim is None:
            return None
        if (self._proj is None) or (self._proj_in != D):
            self._proj = nn.Linear(D, self.attention_dim).to(self.device)
            self._proj_in = D
        return self._proj

    def _ensure_att_mlp(self, D):
        """Create or reuse an MLP that maps 2*D -> D (per-feature gating logits)."""
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
        """Convert numpy array to torch tensor on configured device or pass through tensors."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(self.device)
        raise TypeError("Input must be a numpy array or torch tensor")

    def fuse_learned_attention(self, q_feats, c_feats, per_feature=True):
        """Learned attention fusion.

        If per_feature=True (default) returns per-feature gating: gate * q + (1-gate) * c,
        where gate is produced by an MLP that takes [q, c] (concatenated) and outputs D logits fed
        through sigmoid -> per-feature gates.

        If per_feature=False returns a scalar gate per sample (MLP -> sigmoid -> scalar).
        When attention_dim is set, the fused vector is projected to that dimension before returning.

        This method uses torch and will create torch modules lazily; to train them, reuse the
        same FusionMethods instance and include its parameters in your optimizer.
        """
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape

        # convert to torch
        tq = self._to_torch(q).float()
        tc = self._to_torch(c).float()

        if per_feature:
            # MLP outputs D logits -> sigmoid -> per-feature gates
            mlp = self._ensure_att_mlp(D)
            inp = torch.cat([tq, tc], dim=1)  # (N, 2D)
            logits = mlp(inp)                 # (N, D)
            gates = torch.sigmoid(logits)     # (N, D)
            fused_t = gates * tq + (1.0 - gates) * tc
        else:
            # scalar gate per sample
            mlp = self._ensure_att_mlp(D)  # reuse but average output to scalar
            inp = torch.cat([tq, tc], dim=1)
            logits = mlp(inp)              # (N, D)
            # reduce to scalar per sample (mean) then sigmoid
            scalar_gate = torch.sigmoid(logits.mean(dim=1, keepdim=True))  # (N,1)
            fused_t = scalar_gate * tq + (1.0 - scalar_gate) * tc

        # optional projection to attention_dim
        if self.attention_dim is not None:
            proj = self._ensure_proj(fused_t.shape[1])
            fused_t = proj(fused_t)

        # return numpy for compatibility
        return fused_t.detach().cpu().numpy()

    def fuse_cross_attention(self, q_feats, c_feats, eps=1e-8):
        """Scaled dot-product style cross-attention over the 2-token sequence [q, c].

        For each sample build M=(2,D). Queries = M, Keys = M, compute attention weights and
        return attended representations for the 'pooled' query (e.g., mean over token outputs).
        This is numerically stable and similar to fuse_self_attention but implemented with torch
        when possible for speed; returns numpy array.
        """
        q, c = _match_dims(q_feats, c_feats)
        N, D = q.shape

        # convert to torch for stable batched matmuls
        tM = torch.stack([self._to_torch(q), self._to_torch(c)], dim=1).float()  # (N,2,D)
        # tM: treat tokens as sequence dim=1
        # compute attention scores: Q @ K^T per sample -> (N,2,2)
        Q = tM
        K = tM
        # scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (np.sqrt(D) + eps)  # (N,2,2)
        # softmax over last dim
        attn = torch.softmax(scores, dim=-1)  # (N,2,2)
        # apply to values (V = tM)
        V = tM
        contexts = torch.matmul(attn, V)  # (N,2,D)
        # pool across token outputs (mean)
        pooled = contexts.mean(dim=1)  # (N,D)

        # optional projection
        if self.attention_dim is not None:
            proj = self._ensure_proj(pooled.shape[1])
            pooled = proj(pooled)

        return pooled.detach().cpu().numpy()

__all__ = ["FusionMethods"]

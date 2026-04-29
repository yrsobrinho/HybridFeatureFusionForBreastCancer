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


class OnlineFusionModel(nn.Module):
    def __init__(self, classical_model, quantum_model, feature_dim=128, 
                 num_classes=2, freeze_classical=False, freeze_quantum=False):
        super(OnlineFusionModel, self).__init__()
        
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Freeze models if requested
        if freeze_classical:
            for param in self.classical_model.parameters():
                param.requires_grad = False
        
        if freeze_quantum:
            for param in self.quantum_model.parameters():
                param.requires_grad = False
 
        self.classical_projection = None
        self.quantum_projection = None

        self.fusion_norm = nn.LayerNorm(feature_dim * 2)
        self.fusion_relu = nn.ReLU()
        self.fusion_fc = nn.Linear(feature_dim * 2, num_classes)
        
    def _init_projections(self, classical_dim, quantum_dim, device):
        """Initialize projection heads based on actual feature dimensions."""
        if self.classical_projection is None:
            self.classical_projection = nn.Linear(classical_dim, self.feature_dim).to(device)
        
        if self.quantum_projection is None:
            self.quantum_projection = nn.Linear(quantum_dim, self.feature_dim).to(device)
    
    def load_pretrained_weights(self, classical_path=None, quantum_path=None, device='cpu'):
        """Load pretrained weights for classical and/or quantum models.
        
        Args:
            classical_path: Path to classical model weights (.pth file)
            quantum_path: Path to quantum model weights (.pth file)
            device: Device to load weights to
        """
        if classical_path is not None:
            try:
                state_dict = torch.load(classical_path, map_location=device)
                # Handle different checkpoint formats
                if isinstance(state_dict, dict):
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                self.classical_model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded classical model weights from {classical_path}")
            except Exception as e:
                print(f"Warning: Could not load classical weights: {e}")
        
        if quantum_path is not None:
            try:
                state_dict = torch.load(quantum_path, map_location=device)
                # Handle different checkpoint formats
                if isinstance(state_dict, dict):
                    if 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                
                self.quantum_model.load_state_dict(state_dict, strict=False)
                print(f"✓ Loaded quantum model weights from {quantum_path}")
            except Exception as e:
                print(f"Warning: Could not load quantum weights: {e}")
    
    def extract_features(self, x):

        if hasattr(self.classical_model, 'extract_features'):
            classical_feats = self.classical_model.extract_features(x)
        else:

            classical_feats = self.classical_model(x)
        
        if hasattr(self.quantum_model, 'extract_features'):
            quantum_feats = self.quantum_model.extract_features(x)
        else:
            quantum_feats = self.quantum_model(x)

        if classical_feats.ndim > 2:
            classical_feats = classical_feats.flatten(start_dim=1)
        if quantum_feats.ndim > 2:
            quantum_feats = quantum_feats.flatten(start_dim=1)

        if self.classical_projection is None or self.quantum_projection is None:
            self._init_projections(classical_feats.shape[1], quantum_feats.shape[1], x.device)

        classical_proj = self.classical_projection(classical_feats)
        quantum_proj = self.quantum_projection(quantum_feats)

        classical_norm = F.normalize(classical_proj, p=2, dim=1)
        quantum_norm = F.normalize(quantum_proj, p=2, dim=1)
        
        fused = torch.cat([classical_norm, quantum_norm], dim=1)
        
        return fused
    
    def forward(self, x):
        fused = self.extract_features(x)
        
        out = self.fusion_norm(fused)
        out = self.fusion_relu(out)
        out = self.fusion_fc(out)
        
        return out
    
    def forward_relu(self, x):
        fused = self.extract_features(x)

        out = self.fusion_norm(fused)
        out = self.fusion_relu(out)
        
        return out
    
    def get_parameter_groups(self, lr_classical=1e-4, lr_quantum=1e-3, lr_fusion=1e-3):
        param_groups = []

        classical_params = [p for p in self.classical_model.parameters() if p.requires_grad]
        if classical_params:
            param_groups.append({'params': classical_params, 'lr': lr_classical})
        
        quantum_params = [p for p in self.quantum_model.parameters() if p.requires_grad]
        if quantum_params:
            param_groups.append({'params': quantum_params, 'lr': lr_quantum})
        
        fusion_params = []
        if self.classical_projection is not None:
            fusion_params.extend([p for p in self.classical_projection.parameters() if p.requires_grad])
        if self.quantum_projection is not None:
            fusion_params.extend([p for p in self.quantum_projection.parameters() if p.requires_grad])
        fusion_params.extend([p for p in self.fusion_norm.parameters() if p.requires_grad])
        fusion_params.extend([p for p in self.fusion_fc.parameters() if p.requires_grad])
        
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': lr_fusion})
        
        return param_groups


__all__ = ["FusionMethods", "OnlineFusionModel"]

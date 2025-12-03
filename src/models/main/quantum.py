import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, patch_size, mode="2d", stride=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.mode = mode
        self.stride = stride if stride is not None else patch_size

        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.layers.RandomLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="adjoint")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def calculate_output_shape(self, h, w):
        p = self.patch_size
        s = self.stride
        
        if self.mode == "2d":
            pad_h = (p - (h % p)) % p if (h % p) != 0 else 0
            pad_w = (p - (w % p)) % p if (w % p) != 0 else 0
            
            h_padded = h + pad_h
            w_padded = w + pad_w
            

            out_h = (h_padded - p) // s + 1
            out_w = (w_padded - p) // s + 1
            
            total_features = self.n_qubits * out_h * out_w
            return (out_h, out_w), total_features
            
        else: 
           
            pad_w = (p - (w % p)) % p
            pad_h = (p - (h % p)) % p
            h_after_h_pad = h + pad_h
            w_after_h_pad = w + pad_w
            
            out_h_horiz = h_after_h_pad
            out_w_horiz = w_after_h_pad // p
            
        
            h_for_vert = out_h_horiz
            pad_h_vert = (p - (h_for_vert % p)) % p
            h_after_v_pad = h_for_vert + pad_h_vert
            
            out_h_final = h_after_v_pad // p
            out_w_final = out_w_horiz
            
            total_features = self.n_qubits * out_h_final * out_w_final
            return (out_h_final, out_w_final), total_features

    def _prepare_patches(self, data, dim_size):
        B, seq_len = data.shape
        p = self.patch_size
 
        num_patches = seq_len // p
        patches = data[:, :num_patches * p].view(B, num_patches, p)
        
   
        patches = patches.reshape(-1, p)
        

        if p > self.n_qubits:
            patches = patches[:, :self.n_qubits]
        elif p < self.n_qubits:
            pad_size = self.n_qubits - p
            patches = F.pad(patches, (0, pad_size))
        
        return patches, num_patches

    def quanv1d_horizontal(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Pad once
        pad_w = (p - (W % p)) % p
        pad_h = (p - (H % p)) % p
        if pad_w > 0 or pad_h > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        out_w = W // p
        
        rows = x[:, 0, :, :].reshape(B * H, W)
        
        # Prepare all patches at once
        patches, num_patches = self._prepare_patches(rows, W)
        
        q_out = self.qlayer(patches) 
        
  
        q_out = q_out.view(B, H, num_patches, self.n_qubits)
        q_out = q_out.permute(0, 3, 1, 2)  
        
        return q_out

    def quanv1d_vertical(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Pad once
        pad_h = (p - (H % p)) % p
        if pad_h > 0:
            x = F.pad(x, (0, 0, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        out_h = H // p
        
        cols = x[:, 0, :, :].permute(0, 2, 1).reshape(B * W, H)
        
        patches, num_patches = self._prepare_patches(cols, H)
        
        q_out = self.qlayer(patches) 
        
        q_out = q_out.view(B, W, num_patches, self.n_qubits)
        q_out = q_out.permute(0, 3, 2, 1)  
        
        return q_out

    def quanv2d(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride
    
        if C > 1:
            x = x.mean(dim=1, keepdim=True)  
            C = 1

        pad_h = (p - (H % p)) % p if (H % p) != 0 else 0
        pad_w = (p - (W % p)) % p if (W % p) != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape
        
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1

        patches = x.unfold(2, p, s).unfold(3, p, s)  
        
        patches = patches.squeeze(1) 
        patches = patches.reshape(B, out_h, out_w, p * p) 
        
        patches_flat = patches.reshape(B * out_h * out_w, p * p)
        
        patch_dim = p * p
        if patch_dim > self.n_qubits:
            patches_flat = patches_flat[:, :self.n_qubits]
        elif patch_dim < self.n_qubits:
            pad_size = self.n_qubits - patch_dim
            patches_flat = F.pad(patches_flat, (0, pad_size), value=0.0)
        
 
        q_out = self.qlayer(patches_flat)  
        
       
        q_out = q_out.view(B, out_h, out_w, self.n_qubits)
        q_out = q_out.permute(0, 3, 1, 2)  
        return q_out

    def forward(self, x):
        if self.mode == "2d":
            return self.quanv2d(x)
        elif self.mode == "1d":
            x = self.quanv1d_horizontal(x)
            x = self.quanv1d_vertical(x)
            return x
        elif self.mode == "hybrid":
            x_h = self.quanv1d_horizontal(x)
            x_v = self.quanv1d_vertical(x)
            return x_v @ x_h
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use '2d', '1d-sequential', or 'hybrid'")

class QuantumModelWithEncoder(nn.Module):
    def __init__(self, n_qubits, n_layers, num_classes,
                 patch_size, encoder: nn.Module = None, encoder_output_size: int = 28,
                 mode: str = "2d", stride: int = None):

        super().__init__()
        self.encoder = encoder

        self.quanv = QuanvLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            patch_size=patch_size,
            mode=mode,
            stride=stride
        )

        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        self.mode = mode

        output_shape, self.expected_features = self.quanv.calculate_output_shape(
            encoder_output_size, encoder_output_size
        )
    
        self.fc1 = None  
        self.fc2 = None  
        self.fc3 = None  

    def _apply_encoder(self, x):
        if self.encoder is None:
            return x
        enc = None

        if hasattr(self.encoder, "encode"):
            with torch.no_grad():
                enc = self.encoder.encode(x)
        elif hasattr(self.encoder, "encoder"):
            try:
                with torch.no_grad():
                    enc = self.encoder.encoder(x)
            except Exception:
                with torch.no_grad():
                    enc = self.encoder(x)
        else:
            with torch.no_grad():
                out = self.encoder(x)
            if isinstance(out, (tuple, list)) and len(out) > 1:
                enc = out[1]
            else:
                enc = out

        if enc is None:
            raise RuntimeError("Encoder returned None")

        enc = enc.detach().float()

        if enc.ndim == 4:
            enc = enc
        elif enc.ndim == 3:
            enc = enc.unsqueeze(1)
        elif enc.ndim == 2:
            B, F = enc.shape
            s = int(math.sqrt(F))
            if s * s == F:
                enc = enc.view(B, 1, s, s)
            else:
                enc = enc.view(B, 1, 1, F)
        else:
            raise RuntimeError(f"Unsupported encoder output ndim={enc.ndim}")

        if self.encoder_output_size is not None:
            target = self.encoder_output_size
            _, _, h, w = enc.shape
            if (h, w) != (target, target):
                mode = 'bilinear' if h > 1 and w > 1 else 'bilinear'
                enc = F.interpolate(enc, size=(target, target), mode=mode, align_corners=False)
        return enc

    def forward(self, x):
        x = self._apply_encoder(x)
        x = self.quanv(x) 
        x = x.flatten(start_dim=1)  

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)
            self.bn1 = nn.BatchNorm1d(128).to(x.device)
            self.fc2 = nn.Linear(128, 64).to(x.device)
            self.bn2 = nn.BatchNorm1d(64).to(x.device)
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
        """Return 64-dim features (after fc2)."""
        x = self._apply_encoder(x)
        x = self.quanv(x)
        x = x.flatten(start_dim=1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)
            self.fc2 = nn.Linear(128, 64).to(x.device)
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x  # 64-dim features

    def classify_from_flat(self, x):
        """Classify from 64-dim features."""
        if self.fc3 is None:
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)
        
        out = self.fc3(x)
        return F.log_softmax(out, dim=1)


class QuantumModel(nn.Module):
    def __init__(self, n_qubits, n_layers, num_classes, patch_size, 
                 mode: str = "2d", stride: int = None):
      
        super().__init__()
        self.quanv = QuanvLayer(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            patch_size=patch_size,
            mode=mode,
            stride=stride
        )
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.mode = mode
        
        # Initialize layers upfront
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.bn1 = None
        self.bn2 = None

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.quanv(x)
        x = x.flatten(start_dim=1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)
            self.bn1 = nn.BatchNorm1d(128).to(x.device)
            self.fc2 = nn.Linear(128, 64).to(x.device)
            self.bn2 = nn.BatchNorm1d(64).to(x.device)
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        
        x = self.quanv(x)
        x = x.flatten(start_dim=1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 128).to(x.device)
            self.bn1 = nn.BatchNorm1d(128).to(x.device)
            self.fc2 = nn.Linear(128, 64).to(x.device)
            self.bn2 = nn.BatchNorm1d(64).to(x.device)
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x 

    def classify_from_flat(self, x):
        if self.fc3 is None:
            self.fc3 = nn.Linear(64, self.num_classes).to(x.device)
        
        out = self.fc3(x)
        return F.log_softmax(out, dim=1)


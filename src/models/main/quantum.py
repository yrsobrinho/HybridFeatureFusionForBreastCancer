import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class NonTrainableLayer:
    def __init__(self, n_qubits, n_layers, rand_params):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rand_params = rand_params
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(phi):
            for j in range(n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            qml.templates.layers.RandomLayers(rand_params, list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
        
        self.circuit = circuit
    
    def __call__(self, phi):
        return self.circuit(phi)

class QuantumModel(nn.Module):
    def __init__(self, n_qubits, n_layers, rand_params, output_size=(14, 14), 
                 patch_size=2, num_classes=2):
        super(QuantumModel, self).__init__()
        self.output_size = output_size
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        
        self.circuit = NonTrainableLayer(n_qubits, n_layers, rand_params)
        
        self.flatten = nn.Flatten()
        self.fc = None 

    def _quanvolution_single(self, image):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        height_patches = image.shape[0] // self.patch_size
        width_patches = image.shape[1] // self.patch_size
        
        out = np.zeros((height_patches, width_patches, self.n_qubits))
        
        for j in range(height_patches):
            for k in range(width_patches):
                patch = []
                for i in range(self.patch_size):
                    for l in range(self.patch_size):
                        if (j * self.patch_size + i < image.shape[0]) and (k * self.patch_size + l < image.shape[1]):
                            patch.append(image[j * self.patch_size + i, k * self.patch_size + l, 0])
                        else:
                            patch.append(0)
                
                q_results = self.circuit(patch)
                
                for c in range(self.n_qubits):
                    out[j, k, c] = q_results[c]
        
        return out

    def _quanvolution_batch(self, images):
        batch_size = images.shape[0]
        processed = [
            self._quanvolution_single(images[i].detach().cpu().numpy())
            for i in range(batch_size)
        ]
        processed = np.array(processed)
        return torch.tensor(processed, dtype=torch.float32).to(images.device)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self._quanvolution_batch(x)
        x = torch.relu(x)
        x = self.flatten(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x

class QuantumModelWithEncoder(nn.Module):
    def __init__(self, autoencoder, n_qubits, n_layers, rand_params, latent_channels=3, 
                 output_size=(14, 14), patch_size=2, num_classes=2):
        super(QuantumModelWithEncoder, self).__init__()
        self.latent_channels = latent_channels
        self.output_size = output_size
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        
        self.autoencoder = autoencoder
        self.autoencoder.eval()
        
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.circuit = NonTrainableLayer(n_qubits, n_layers, rand_params)
        
        self.flatten = nn.Flatten()
        self.fc = None 

    def _quanvolution_single(self, image):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
        
        height_patches = image.shape[0] // self.patch_size
        width_patches = image.shape[1] // self.patch_size
        
        out = np.zeros((height_patches, width_patches, self.n_qubits))
        
        for j in range(height_patches):
            for k in range(width_patches):
                patch = []
                for i in range(self.patch_size):
                    for l in range(self.patch_size):
                        if (j * self.patch_size + i < image.shape[0]) and (k * self.patch_size + l < image.shape[1]):
                            patch.append(image[j * self.patch_size + i, k * self.patch_size + l, 0])
                        else:
                            patch.append(0)
                
                q_results = self.circuit(patch)
                
                for c in range(self.n_qubits):
                    out[j, k, c] = q_results[c]
        
        return out

    def _quanvolution_batch(self, images):
        batch_size = images.shape[0]
        processed = [
            self._quanvolution_single(images[i].detach().cpu().numpy())
            for i in range(batch_size)
        ]
        processed = np.array(processed)
        return torch.tensor(processed, dtype=torch.float32).to(images.device)

    def forward(self, x):
        with torch.no_grad():
            x = self.autoencoder.encode(x)
        
        # Resize encoder output to match expected size (e.g., 28x28)
        # The encoder may output smaller images (e.g., 3x3), so we need to interpolate
        target_size = 28  # Same as QuantumModel without encoder
        _, _, h, w = x.shape
        if h != target_size or w != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Average the 3 latent channels into 1 channel to match QuantumModel behavior
        # This gives us (B, 1, 28, 28) instead of (B, 3, 28, 28)
        x = x.mean(dim=1, keepdim=True)
        
        # Now process the single averaged channel
        x = x.permute(0, 2, 3, 1)
        x = self._quanvolution_batch(x)
        
        x = torch.relu(x)
        x = self.flatten(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x

class TrainableLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, patch_size, mode="2d", stride=None, use_parallel=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.mode = mode
        self.stride = stride if stride is not None else patch_size
        self.use_parallel = use_parallel

        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.layers.RandomLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        print(f"Quantum layer has {sum(p.numel() for p in self.qlayer.parameters())} trainable parameters")
        print(f"Requires grad: {all(p.requires_grad for p in self.qlayer.parameters())}")

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

    def horizontal(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        pad_w = (p - (W % p)) % p
        pad_h = (p - (H % p)) % p
        if pad_w > 0 or pad_h > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        out_w = W // p
        
        rows = x[:, 0, :, :].reshape(B * H, W)
        
        patches, num_patches = self._prepare_patches(rows, W)
        
        q_out = self.qlayer(patches) 
        
  
        q_out = q_out.view(B, H, num_patches, self.n_qubits)
        q_out = q_out.permute(0, 3, 1, 2)  
        
        return q_out

    def vertical(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

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
    
    def parallel(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        pad_w = (p - (W % p)) % p
        pad_h = (p - (H % p)) % p
        if pad_w > 0 or pad_h > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        x_single = x[:, 0, :, :]  

        rows = x_single.reshape(B * H, W)
        cols = x_single.permute(0, 2, 1).reshape(B * W, H)

        num_patches_h = W // p
        num_patches_v = H // p

        patches_h = rows[:, :num_patches_h * p].reshape(-1, p)
        patches_v = cols[:, :num_patches_v * p].reshape(-1, p)

        if p != self.n_qubits:
            if p > self.n_qubits:
                patches_h = patches_h[:, :self.n_qubits]
                patches_v = patches_v[:, :self.n_qubits]
            else:
                pad_size = self.n_qubits - p
                patches_h = F.pad(patches_h, (0, pad_size))
                patches_v = F.pad(patches_v, (0, pad_size))

        all_patches = torch.cat([patches_h, patches_v], dim=0)
        all_q_out = self.qlayer(all_patches)

        total_h = patches_h.shape[0]
        q_out_h = all_q_out[:total_h].view(B, H, num_patches_h, self.n_qubits)
        q_out_v = all_q_out[total_h:].view(B, W, num_patches_v, self.n_qubits)

        q_out_h = q_out_h.permute(0, 3, 1, 2)
        q_out_v = q_out_v.permute(0, 3, 2, 1)
        
        return q_out_h, q_out_v

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
        elif self.mode == "sequential":
            x = self.horizontal(x)
            x = self.vertical(x)
            return x
        elif self.mode == "matmul":
            if self.use_parallel:
                x_h, x_v = self.parallel(x)
            else:
                x_h = self.horizontal(x)
                x_v = self.vertical(x)
            return x_v @ x_h
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use '2d', 'sequential', or 'matmul'")

class ExtentedQuantumModelWithEncoder(nn.Module):
    def __init__(self, n_qubits, n_layers, num_classes,
                 patch_size, encoder: nn.Module = None, encoder_output_size: int = 28,
                 mode: str = "2d", stride: int = None, use_parallel: bool = True):

        super().__init__()
        self.encoder = encoder

        self.quanv = TrainableLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            patch_size=patch_size,
            mode=mode,
            stride=stride,
            use_parallel=use_parallel
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
            B, feature_dim = enc.shape
            s = int(math.sqrt(feature_dim))
            if s * s == feature_dim:
                enc = enc.view(B, 1, s, s)
            else:
                enc = enc.view(B, 1, 1, feature_dim)
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
        return x

class SimpleQuantumModel(nn.Module):
    """Trainable quantum circuit without encoder - minimal classical layers (1 FC)"""
    def __init__(self, n_qubits, n_layers, num_classes, patch_size, 
                 mode: str = "2d", stride: int = None, use_parallel: bool = True):
        super().__init__()
        self.quanv = TrainableLayer(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            patch_size=patch_size,
            mode=mode,
            stride=stride,
            use_parallel=use_parallel
        )
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.mode = mode
        
        self.fc = None  # Single FC layer

    def forward(self, x):
        x = self.quanv(x)
        x = x.flatten(start_dim=1)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)

        x = self.fc(x)
        return x


class SimpleQuantumModelWithEncoder(nn.Module):
    """Trainable quantum circuit with encoder - minimal classical layers (1 FC)"""
    def __init__(self, n_qubits, n_layers, num_classes,
                 patch_size, encoder: nn.Module = None, encoder_output_size: int = 28,
                 mode: str = "2d", stride: int = None, use_parallel: bool = True):
        super().__init__()
        self.encoder = encoder

        self.quanv = TrainableLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            patch_size=patch_size,
            mode=mode,
            stride=stride,
            use_parallel=use_parallel
        )

        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        self.mode = mode

        output_shape, self.expected_features = self.quanv.calculate_output_shape(
            encoder_output_size, encoder_output_size
        )
    
        self.fc = None  # Single FC layer

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
            B, feature_dim = enc.shape
            s = int(math.sqrt(feature_dim))
            if s * s == feature_dim:
                enc = enc.view(B, 1, s, s)
            else:
                enc = enc.view(B, 1, 1, feature_dim)
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

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)

        x = self.fc(x)
        return x


class ExtentedQuantumModel(nn.Module):
    """Trainable quantum circuit without encoder - multiple classical layers"""
    def __init__(self, n_qubits, n_layers, num_classes, patch_size, 
                 mode: str = "2d", stride: int = None, use_parallel: bool = True):
      
        super().__init__()
        self.quanv = TrainableLayer(
            n_qubits=n_qubits, 
            n_layers=n_layers, 
            patch_size=patch_size,
            mode=mode,
            stride=stride,
            use_parallel=use_parallel
        )
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.mode = mode
        
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None
        self.bn1 = None
        self.bn2 = None

    def forward(self, x):
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
        return x


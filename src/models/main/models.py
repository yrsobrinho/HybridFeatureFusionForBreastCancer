import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import NonTrainableLayer, TrainableLayer


class EncoderMixin:
    """Mixin class providing encoder application functionality."""
    
    def _apply_encoder(self, x):
        """Apply encoder to input."""
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
            if isinstance(self.encoder_output_size, tuple):
                target = self.encoder_output_size
            else:
                target = (self.encoder_output_size, self.encoder_output_size)
            _, _, h, w = enc.shape
            if (h, w) != target:
                mode = 'bilinear' if h > 1 and w > 1 else 'bilinear'
                enc = F.interpolate(enc, size=target, mode=mode, align_corners=False)
        return enc


class QuanvolutionMixin:
    """Mixin class providing quanvolution functionality for non-trainable quantum circuits."""
    
    def _quanvolution_single(self, image):
        """Apply quanvolution to a single image."""
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
        """Apply quanvolution to a batch of images."""
        batch_size = images.shape[0]
        processed = [
            self._quanvolution_single(images[i].detach().cpu().numpy())
            for i in range(batch_size)
        ]
        processed = np.array(processed)
        return torch.tensor(processed, dtype=torch.float32).to(images.device)


class QuantumModel(QuanvolutionMixin, nn.Module):
    """Non-trainable quantum model without encoder."""
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

    def extract_features(self, x):
        """Extract features before final classification layer.
        
        Returns:
            Features tensor [batch, features_dim] before classification
        """
        x = x.permute(0, 2, 3, 1)
        x = self._quanvolution_batch(x)
        x = torch.relu(x)
        x = self.flatten(x)
        return x
    
    def forward(self, x):
        x = self.extract_features(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x
    
    def forward_relu(self, x):
        x = self.extract_features(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        return x


class QuantumModelWithEncoder(EncoderMixin, QuanvolutionMixin, nn.Module):
    """Non-trainable quantum model with encoder."""
    def __init__(self, autoencoder, n_qubits, n_layers, rand_params, latent_channels=3, 
                 output_size=(14, 14), patch_size=2, num_classes=2, encoder_output_size=28):
        super(QuantumModelWithEncoder, self).__init__()
        self.latent_channels = latent_channels
        self.output_size = output_size
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size
        
        self.encoder = autoencoder
        if self.encoder is not None:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.circuit = NonTrainableLayer(n_qubits, n_layers, rand_params)
        
        self.flatten = nn.Flatten()
        self.fc = None

    def forward(self, x):
        x = self._apply_encoder(x)
        
        
        
        x = x.permute(0, 2, 3, 1)
        x = self._quanvolution_batch(x)
        
        x = torch.relu(x)
        x = self.flatten(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x

class ExtentedQuantumModel(nn.Module):
    """Trainable quantum model without encoder - single FC layer."""
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
        
        self.flatten = nn.Flatten()
        self.fc = None

    def extract_features(self, x):
        """Extract features before final classification layer.
        
        Returns:
            Features tensor [batch, features_dim] before classification
        """
        x = self.quanv(x)
        x = torch.relu(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x
    
    def forward(self, x):
        x = self.extract_features(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x

class ExtentedQuantumModelWithEncoder(EncoderMixin, nn.Module):
    """Trainable quantum model with encoder - single FC layer."""
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
    
        self.flatten = nn.Flatten()
        self.fc = None

    def forward(self, x):
        x = self._apply_encoder(x)
        
        x = self.quanv(x)
        x = torch.relu(x)
        x = self.flatten(x)

        if self.fc is None:
            in_features = x.shape[1]
            self.fc = nn.Linear(in_features, self.num_classes).to(x.device)
        
        x = self.fc(x)
        return x


class DualStreamQuantumModel(EncoderMixin, nn.Module):
    """Dual-stream quantum model with separate horizontal and vertical processing pipelines.
    
    Architecture:
    - Stream 1 (Horizontal): quantum horizontal → FC layers → features_h
    - Stream 2 (Vertical): quantum vertical → FC layers → features_v
    - Fusion: features_h + features_v → final classification
    
    Args:
        fusion_mode: How to combine streams - 'concat', 'sum', 'avg', 'max', 'weighted'
        use_horizontal: Enable horizontal stream
        use_vertical: Enable vertical stream
    """
    def __init__(self, n_qubits, n_layers, num_classes, patch_size,
                 encoder: nn.Module = None, encoder_output_size: int = 28,
                 stride: int = None, use_parallel: bool = True,
                 fusion_mode: str = "concat",
                 use_horizontal: bool = True,
                 use_vertical: bool = True,
                 hidden_dims: list = [128, 64]):
        super().__init__()
        
        if not use_horizontal and not use_vertical:
            raise ValueError("At least one stream (horizontal or vertical) must be enabled")
        
        self.encoder = encoder
        self.encoder_output_size = encoder_output_size
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode
        self.use_horizontal = use_horizontal
        self.use_vertical = use_vertical
        self.hidden_dims = hidden_dims
        
        # Horizontal stream (quantum + classical)
        if use_horizontal:
            self.quanv_h = TrainableLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                patch_size=patch_size,
                mode="sequential",  # Will only use horizontal
                stride=stride,
                use_parallel=use_parallel
            )
            self.fc_h1 = None
            self.fc_h2 = None
            self.bn_h1 = None
            self.bn_h2 = None
        
        # Vertical stream (quantum + classical)
        if use_vertical:
            self.quanv_v = TrainableLayer(
                n_qubits=n_qubits,
                n_layers=n_layers,
                patch_size=patch_size,
                mode="sequential",  # Will only use vertical
                stride=stride,
                use_parallel=use_parallel
            )
            self.fc_v1 = None
            self.fc_v2 = None
            self.bn_v1 = None
            self.bn_v2 = None
        
        # Fusion layer
        self.fc_fusion = None
        
        # Learnable fusion weights if using weighted fusion
        if fusion_mode == "weighted":
            self.alpha = nn.Parameter(torch.tensor(0.5))

    def _process_horizontal(self, x):
        """Process through horizontal quantum + classical stream."""
        x_h = self.quanv_h.horizontal(x)
        x_h = x_h.flatten(start_dim=1)
        
        if self.fc_h1 is None:
            in_features = x_h.shape[1]
            self.fc_h1 = nn.Linear(in_features, self.hidden_dims[0]).to(x_h.device)
            self.bn_h1 = nn.BatchNorm1d(self.hidden_dims[0]).to(x_h.device)
            self.fc_h2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1]).to(x_h.device)
            self.bn_h2 = nn.BatchNorm1d(self.hidden_dims[1]).to(x_h.device)
        
        x_h = self.fc_h1(x_h)
        x_h = self.bn_h1(x_h)
        x_h = F.relu(x_h)
        x_h = F.dropout(x_h, p=0.3, training=self.training)
        
        x_h = self.fc_h2(x_h)
        x_h = self.bn_h2(x_h)
        x_h = F.relu(x_h)
        x_h = F.dropout(x_h, p=0.2, training=self.training)
        
        return x_h

    def _process_vertical(self, x):
        """Process through vertical quantum + classical stream."""
        x_v = self.quanv_v.vertical(x)
        x_v = x_v.flatten(start_dim=1)
        
        if self.fc_v1 is None:
            in_features = x_v.shape[1]
            self.fc_v1 = nn.Linear(in_features, self.hidden_dims[0]).to(x_v.device)
            self.bn_v1 = nn.BatchNorm1d(self.hidden_dims[0]).to(x_v.device)
            self.fc_v2 = nn.Linear(self.hidden_dims[0], self.hidden_dims[1]).to(x_v.device)
            self.bn_v2 = nn.BatchNorm1d(self.hidden_dims[1]).to(x_v.device)
        
        x_v = self.fc_v1(x_v)
        x_v = self.bn_v1(x_v)
        x_v = F.relu(x_v)
        x_v = F.dropout(x_v, p=0.3, training=self.training)
        
        x_v = self.fc_v2(x_v)
        x_v = self.bn_v2(x_v)
        x_v = F.relu(x_v)
        x_v = F.dropout(x_v, p=0.2, training=self.training)
        
        return x_v

    def _fuse_features(self, x_h, x_v):
        """Fuse horizontal and vertical features."""
        if x_h is None:
            return x_v
        if x_v is None:
            return x_h
        
        if self.fusion_mode == "concat":
            fused = torch.cat([x_h, x_v], dim=1)
        elif self.fusion_mode == "sum":
            fused = x_h + x_v
        elif self.fusion_mode == "avg":
            fused = (x_h + x_v) / 2
        elif self.fusion_mode == "max":
            fused = torch.max(x_h, x_v)
        elif self.fusion_mode == "weighted":
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * x_h + (1 - alpha) * x_v
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        return fused

    def forward(self, x):
        # Apply encoder if available
        if hasattr(self, '_apply_encoder'):
            x = self._apply_encoder(x)
        
        # Ensure single channel
        
        
        # Process through streams
        x_h = self._process_horizontal(x) if self.use_horizontal else None
        x_v = self._process_vertical(x) if self.use_vertical else None
        
        # Fuse features
        fused = self._fuse_features(x_h, x_v)
        
        # Final classification layer
        if self.fc_fusion is None:
            in_features = fused.shape[1]
            self.fc_fusion = nn.Linear(in_features, self.num_classes).to(fused.device)
        
        out = self.fc_fusion(fused)
        return out

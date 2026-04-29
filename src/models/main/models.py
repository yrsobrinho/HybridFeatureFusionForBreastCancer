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


    def _pad_image_to_patch_grid(self, image):
        """Pad image to multiples of patch_size so no border information is dropped."""
        h, w = image.shape[:2]
        pad_h = (self.patch_size - (h % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (w % self.patch_size)) % self.patch_size

        if pad_h == 0 and pad_w == 0:
            return image

        padded = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        return padded
    
    def _quanvolution_single(self, image):
        """Apply quanvolution to a single image."""
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)

        image = self._pad_image_to_patch_grid(image)
        
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

    def _resize_feature_map_if_needed(self, x):
        """Resize quanvolution map to a fixed output size for input-size flexibility."""
        if self.output_size is None:
            return x

        if isinstance(self.output_size, int):
            target_size = (self.output_size, self.output_size)
        else:
            target_size = tuple(self.output_size)

        if x.shape[-2:] == target_size:
            return x

        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def extract_features(self, x):
        """Extract features before final classification layer.
        
        Returns:
            Features tensor [batch, features_dim] before classification
        """
        x = x.permute(0, 2, 3, 1)
        x = self._quanvolution_batch(x)
        x = x.permute(0, 3, 1, 2)
        x = self._resize_feature_map_if_needed(x)
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
                 mode: str = "2d", stride: int = None, use_parallel: bool = True,
                 output_size=None):
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
        self.output_size = output_size
        
        self.flatten = nn.Flatten()
        self.fc = None

    def _resize_feature_map_if_needed(self, x):
        """Resize trainable quanvolution feature map to keep FC input shape stable."""
        if self.output_size is None:
            return x

        if isinstance(self.output_size, int):
            target_size = (self.output_size, self.output_size)
        else:
            target_size = tuple(self.output_size)

        if x.shape[-2:] == target_size:
            return x

        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def extract_features(self, x):
        """Extract features before final classification layer.
        
        Returns:
            Features tensor [batch, features_dim] before classification
        """
        x = self.quanv(x)
        x = self._resize_feature_map_if_needed(x)
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

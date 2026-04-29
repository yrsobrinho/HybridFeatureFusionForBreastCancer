"""
Feature Extractors for Classical and Quantum Models

Provides wrapper classes that extract intermediate features from trained models
before the final classification layer.
"""

import torch
import torch.nn as nn


class ClassicalFeatureExtractor(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def forward(self, x):
        """Extract features before final classification layer."""
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features


class ClassicalFeatureExtractorWithEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
        # Ensure encoder is frozen
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        """Extract features after encoder and before final classification layer."""
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features


class QuantumFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
    def forward(self, x):
        """Extract features after quantum layer and before final FC layer."""
        with torch.no_grad():
            # For QuantumModel (non-trainable)
            if hasattr(self.model, 'circuit'):
                x = x.permute(0, 2, 3, 1)
                x = self.model._quanvolution_batch(x)
                x = torch.relu(x)
                x = self.model.flatten(x)
            elif hasattr(self.model, 'quanv'):
                x = self.model.quanv(x)
                x = torch.relu(x)
                x = self.model.flatten(x)
            else:
                raise ValueError("Unknown quantum model architecture")
        
        return x


class QuantumFeatureExtractorWithEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        
        # Ensure encoder is frozen
        if hasattr(self.model, 'encoder') and self.model.encoder is not None:
            self.model.encoder.eval()
            for param in self.model.encoder.parameters():
                param.requires_grad = False
        
    def forward(self, x):
        """Extract features after encoder + quantum layer and before final FC layer."""
        with torch.no_grad():
            # Apply encoder
            if hasattr(self.model, '_apply_encoder'):
                x = self.model._apply_encoder(x)
            
            # For QuantumModelWithEncoder (non-trainable)
            if hasattr(self.model, 'circuit'):
                x = x.permute(0, 2, 3, 1)
                x = self.model._quanvolution_batch(x)
                x = torch.relu(x)
                x = self.model.flatten(x)
            
            # For ExtentedQuantumModelWithEncoder (trainable - single FC layer)
            elif hasattr(self.model, 'quanv') and not hasattr(self.model, 'quanv_h'):
                x = self.model.quanv(x)
                x = torch.relu(x)
                x = self.model.flatten(x)
                # Don't pass through fc - we want features before classification
            
            # For DualStreamQuantumModel
            elif hasattr(self.model, 'quanv_h') or hasattr(self.model, 'quanv_v'):
                x_h = self.model._process_horizontal(x) if self.model.use_horizontal else None
                x_v = self.model._process_vertical(x) if self.model.use_vertical else None
                x = self.model._fuse_features(x_h, x_v)
            
            else:
                raise ValueError("Unknown quantum model architecture")
        
        return x


class GenericFeatureExtractor(nn.Module):
    """Generic feature extractor that automatically detects model type.
    
    Automatically determines the correct extraction method based on model architecture.
    
    Args:
        model: Trained model (classical or quantum, with or without encoder)
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        self._setup_extractor()
        
    def _setup_extractor(self):
        """Automatically detect model type and configure extractor."""
        has_encoder = hasattr(self.model, 'encoder') and self.model.encoder is not None
        has_quantum = (hasattr(self.model, 'circuit') or 
                      hasattr(self.model, 'quanv') or 
                      hasattr(self.model, 'quanv_h') or 
                      hasattr(self.model, 'quanv_v'))
        
        if has_quantum and has_encoder:
            self.extractor = QuantumFeatureExtractorWithEncoder(self.model)
            self.model_type = "quantum_with_encoder"
        elif has_quantum:
            self.extractor = QuantumFeatureExtractor(self.model)
            self.model_type = "quantum"
        elif has_encoder:
            self.extractor = ClassicalFeatureExtractorWithEncoder(self.model)
            self.model_type = "classical_with_encoder"
        else:
            self.extractor = ClassicalFeatureExtractor(self.model)
            self.model_type = "classical"
    
    def forward(self, x):
        """Extract features using the appropriate extractor."""
        return self.extractor(x)
    
    def __repr__(self):
        return f"GenericFeatureExtractor(model_type='{self.model_type}')"

def create_extractor(model):
    return GenericFeatureExtractor(model)


def extract_features_from_dataloader(model, dataloader, device):
    import numpy as np
    from tqdm import tqdm
    
    extractor = create_extractor(model).to(device)
    extractor.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, labels = batch[0], batch[1]
            else:
                imgs = batch
                labels = None
            
            imgs = imgs.to(device)
            features = extractor(imgs)
            
            features_list.append(features.cpu().numpy())
            
            if labels is not None:
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                labels_list.append(labels)
    
    features = np.concatenate(features_list, axis=0)
    
    if len(labels_list) > 0:
        labels = np.concatenate(labels_list, axis=0)
        return features, labels
    
    return features, None

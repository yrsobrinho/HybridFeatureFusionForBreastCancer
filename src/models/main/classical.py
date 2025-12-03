import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from src.models.preprocessing.autoencoder import Encoder


class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2, input_channels=1):
        super(ShallowCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  
        )
        self.fc = nn.Linear(128, num_classes)
        
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        output = self.fc(features)
        return output


class ShallowCNNWithEncoder(nn.Module):
    def __init__(self, num_classes=2, encoder=None, encoder_output_size=28):
        super(ShallowCNNWithEncoder, self).__init__()

        self.encoder = encoder if encoder is not None else Encoder()
        self.encoder_output_size = encoder_output_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.conv_input_channels = None
        
        self.conv1 = None  
        self.conv2 = None
        self.conv3 = None
        self.fc = nn.Linear(64, num_classes)
        
    def _init_conv_layers(self, input_channels, device):
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) 
        ).to(device)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) 
        ).to(device)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  
        ).to(device)
        
        self.conv_input_channels = input_channels
    
    def _apply_encoder(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)

        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0] 
        
        if encoded.ndim == 2:
            B, F = encoded.shape
            H = W = int(np.sqrt(F))
            if H * W != F:
                H = W = self.encoder_output_size
                encoded = torch.nn.functional.interpolate(
                    encoded.view(B, 1, 1, F), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                encoded = encoded.view(B, 1, H, W)
        elif encoded.ndim == 3:
            encoded = encoded.unsqueeze(-1)
        
        if encoded.shape[-2:] != (self.encoder_output_size, self.encoder_output_size):
            encoded = torch.nn.functional.interpolate(
                encoded, 
                size=(self.encoder_output_size, self.encoder_output_size),
                mode='bilinear',
                align_corners=False
            )
        
        return encoded
    
    def extract_features(self, x):
        x = self._apply_encoder(x)
        
        if self.conv1 is None:
            self._init_conv_layers(x.shape[1], x.device)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        output = self.fc(features)
        return output


class ClassicalModel(nn.Module):
    """ResNet18-based model for breast cancer classification.
    
    Uses centralized training functions from src.utils.training.
    """
    def __init__(self, num_classes):
        super(ClassicalModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
       
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        """Extract features from ResNet backbone before final fc layer."""
        with torch.no_grad():
            feats = self.features(x)
        feats = torch.flatten(feats, 1)
        return feats


class ClassicalModelWithEncoder(nn.Module):
    """Classical model with autoencoder encoder as feature extractor.
    
    Uses centralized training functions from src.utils.training.
    Encoder is frozen during training.
    """
    def __init__(self, num_classes, encoder=None, encoder_output_size=64):
        super(ClassicalModelWithEncoder, self).__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_output_size = encoder_output_size
        self.fc = nn.Linear(self.encoder_output_size, num_classes)
        
        for param in self.encoder.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        with torch.no_grad():
            feat_maps = self.encoder(x)
        if isinstance(feat_maps, (tuple, list)):
            feat_maps = feat_maps[0]
        pooled = self.pool(feat_maps)          
        flattened = pooled.view(pooled.size(0), -1)  
        return flattened

    def forward(self, x):
        feat_maps = self.encoder(x)
        if isinstance(feat_maps, (tuple, list)):
            feat_maps = feat_maps[0]
        pooled = self.pool(feat_maps)
        flattened = pooled.view(pooled.size(0), -1)
        return self.fc(flattened)


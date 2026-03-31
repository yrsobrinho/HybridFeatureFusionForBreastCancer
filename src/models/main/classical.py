import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


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
    
    def forward_relu(self, x):
        features = self.extract_features(x)
        return features
    
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
    
    def forward_relu(self, x):
        """Extract features from ResNet backbone before final fc layer."""
        with torch.no_grad():
            feats = self.features(x)
        feats = torch.flatten(feats, 1)
        return feats

import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.preprocessing.autoencoder import Encoder


class ShallowCNN(nn.Module):
    """
    Shallow CNN with only 3 convolutional layers for faster training and comparison.
    Simple architecture: Conv -> Conv -> Conv -> FC
    """
    def __init__(self, num_classes=2, input_channels=1):
        super(ShallowCNN, self).__init__()
        
        # 3 Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),  # 224x224 -> 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112x112 -> 56x56
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 7x7 -> 1x1
        )
        
        # Fully connected classifier
        self.fc = nn.Linear(128, num_classes)
        
    def extract_features(self, x):
        """Extract features before final classification layer."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, 128)
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        output = self.fc(features)
        return output


class ShallowCNNWithEncoder(nn.Module):
    """
    Shallow CNN with Encoder preprocessing.
    Uses autoencoder to compress 224x224 -> 28x28, then applies 3 conv layers.
    """
    def __init__(self, num_classes=2, encoder=None, encoder_output_size=28):
        super(ShallowCNNWithEncoder, self).__init__()
        
        # Use provided encoder or create default
        self.encoder = encoder if encoder is not None else Encoder()
        self.encoder_output_size = encoder_output_size
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Detect number of channels from encoder output
        # Most encoders output shape (B, C, H, W) where C varies
        # We'll determine this dynamically in forward pass
        self.conv_input_channels = None
        
        # 3 Convolutional layers (input will be 28x28 from encoder)
        self.conv1 = None  # Will be initialized dynamically
        self.conv2 = None
        self.conv3 = None
        
        # Fully connected classifier
        self.fc = nn.Linear(64, num_classes)
        
    def _init_conv_layers(self, input_channels, device):
        """Initialize conv layers based on encoder output channels."""
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28x28 -> 14x14
        ).to(device)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # 14x14 -> 14x14
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        ).to(device)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # 7x7 -> 1x1
        ).to(device)
        
        self.conv_input_channels = input_channels
    
    def _apply_encoder(self, x):
        """Apply encoder and handle different output formats."""
        with torch.no_grad():
            encoded = self.encoder(x)
        
        # Handle different encoder output formats
        if isinstance(encoded, (tuple, list)):
            encoded = encoded[0]  # Take first element if tuple/list
        
        # Ensure 4D tensor (B, C, H, W)
        if encoded.ndim == 2:
            # (B, features) -> reshape to spatial
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
            # (B, C, L) -> reshape
            encoded = encoded.unsqueeze(-1)
        
        # Resize to expected encoder output size if needed
        if encoded.shape[-2:] != (self.encoder_output_size, self.encoder_output_size):
            encoded = torch.nn.functional.interpolate(
                encoded, 
                size=(self.encoder_output_size, self.encoder_output_size),
                mode='bilinear',
                align_corners=False
            )
        
        return encoded
    
    def extract_features(self, x):
        """Extract features before final classification layer."""
        # Apply encoder
        x = self._apply_encoder(x)
        
        # Initialize conv layers if needed
        if self.conv1 is None:
            self._init_conv_layers(x.shape[1], x.device)
        
        # Apply conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten to (B, 64)
        return x
    
    def forward(self, x):
        features = self.extract_features(x)
        output = self.fc(features)
        return output


class ClassicalModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassicalModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
       
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        
    def forward(self, x):
        return self.resnet(x)

    def extract_features(self, x):
        """Return the flattened features from the ResNet backbone (before final fc)."""
        with torch.no_grad():
            feats = self.features(x)
        feats = torch.flatten(feats, 1)
        return feats

    def train_model(self, dataloader, criterion, optimizer, device):
        # optional save_features_dir mirrors the encoder-version API
        def _train(save_features_dir=None):
            self.train()
            total_loss = 0.0
            total_accuracy = 0.0
            print("\n[Classical Model Training]")
            save_feats = save_features_dir is not None
            if save_feats:
                os.makedirs(save_features_dir, exist_ok=True)
                features_list = []
                labels_list = []
            for inputs, labels in tqdm(dataloader, desc="Training Batches", leave=False):
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                optimizer.zero_grad()
                # extract features and run final fc
                with torch.no_grad():
                    feats = self.extract_features(inputs)
                outputs = self.resnet.fc(feats)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_accuracy += (predicted == labels).sum().item()
                if save_feats:
                    features_list.append(feats.cpu().numpy())
                    labels_list.append(labels.cpu().numpy())
            if save_feats and len(features_list) > 0:
                feats_arr = np.concatenate(features_list, axis=0)
                labs_arr = np.concatenate(labels_list, axis=0)
                np.save(os.path.join(save_features_dir, "train_features.npy"), feats_arr)
                np.save(os.path.join(save_features_dir, "train_labels.npy"), labs_arr)
            print(f"Training Loss: {total_loss / len(dataloader):.4f}, Accuracy: {total_accuracy / len(dataloader.dataset):.4f}")
            return total_loss / len(dataloader), total_accuracy / len(dataloader.dataset)

        # backward-compatible call without features
        return _train()

    def validate_model(self, dataloader, criterion, device):
        # optional save_features_dir mirrors the encoder-version API
        def _validate(save_features_dir=None):
            self.eval()
            total_loss = 0.0
            all_labels = []
            all_predictions = []
            all_probs = []

            print("\n[Classical Model Validation]")
            save_feats = save_features_dir is not None
            if save_feats:
                os.makedirs(save_features_dir, exist_ok=True)
                features_list = []
                labels_list = []

            with torch.no_grad():
                for inputs, labels in tqdm(dataloader, desc="Validation Batches", leave=False):
                    inputs, labels = inputs.to(device), labels.squeeze().to(device)
                    feats = self.extract_features(inputs)
                    outputs = self.resnet.fc(feats)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                    if save_feats:
                        features_list.append(feats.cpu().numpy())
                        labels_list.append(labels.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float("nan")

            print(f"Validation Loss: {total_loss / len(dataloader):.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

            if save_feats and len(features_list) > 0:
                feats_arr = np.concatenate(features_list, axis=0)
                labs_arr = np.concatenate(labels_list, axis=0)
                np.save(os.path.join(save_features_dir, "val_features.npy"), feats_arr)
                np.save(os.path.join(save_features_dir, "val_labels.npy"), labs_arr)

            return total_loss / len(dataloader), accuracy, precision, recall, f1, auc

        # backward-compatible call without features
        return _validate()
    
    def test_model(self, dataloader, criterion, device, hyperparams, output_file):
        # optional save_features_dir mirrors the encoder-version API
        def _test(save_features_dir=None):
            self.eval()
            total_loss = 0.0
            all_labels = []
            all_predictions = []
            all_probs = []

            print("\n[Classical Model Testing]")
            save_feats = save_features_dir is not None
            if save_feats:
                os.makedirs(save_features_dir, exist_ok=True)
                features_list = []
                labels_list = []

            with torch.no_grad():
                for inputs, labels in tqdm(dataloader, desc="Testing Batches", leave=False):
                    inputs, labels = inputs.to(device), labels.squeeze().to(device)
                    feats = self.extract_features(inputs)
                    outputs = self.resnet.fc(feats)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()

                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                    all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Probabilities for class 1
                    if save_feats:
                        features_list.append(feats.cpu().numpy())
                        labels_list.append(labels.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, zero_division=0)
            recall = recall_score(all_labels, all_predictions, zero_division=0)
            f1 = f1_score(all_labels, all_predictions, zero_division=0)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float("nan")

            # Write metrics to a file
            with open(output_file, "a") as f:
                f.write(f"\n[Classical Model Testing - {datetime.datetime.now()}]\n")
                f.write(f"Hyperparameters: {hyperparams}\n")
                f.write(f"Test Loss: {total_loss / len(dataloader):.4f}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"AUC: {auc:.4f}\n")
                f.write("-" * 50 + "\n")

            # save features if requested
            if save_feats and len(features_list) > 0:
                feats_arr = np.concatenate(features_list, axis=0)
                labs_arr = np.concatenate(labels_list, axis=0)
                np.save(os.path.join(save_features_dir, "test_features.npy"), feats_arr)
                np.save(os.path.join(save_features_dir, "test_labels.npy"), labs_arr)

            print(f"Test Loss: {total_loss / len(dataloader):.4f}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

            return

        # backward-compatible call without features
        return _test()


class ClassicalModelWithEncoder(nn.Module):
    """Classical model variant that uses the Autoencoder's Encoder as a feature extractor.

    This mirrors the training/validation/test methods of ClassicalModel but replaces the
    ResNet backbone with the encoder defined in src.models.preprocessing.autoencoder.
    The encoder's output is global-pooled and passed to a linear classifier.

    Now accepts:
      - encoder: optional nn.Module encoder instance to use (if None, uses default Encoder()).
      - encoder_output_size: number of features after global pooling (used to size the linear fc).
    """
    def __init__(self, num_classes, encoder=None, encoder_output_size=64):
        super(ClassicalModelWithEncoder, self).__init__()
        # use provided encoder if given, otherwise instantiate the default Encoder
        self.encoder = encoder if encoder is not None else Encoder()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # encoder_output_size should match the number of channels after pooling (default 64)
        self.encoder_output_size = encoder_output_size
        self.fc = nn.Linear(self.encoder_output_size, num_classes)

    def extract_features(self, x):
        """Return the flattened feature vector from the encoder (before the final fc).
        Keeps channel dimension so output shape is (B, C) after global pooling.
        """
        feat_maps = self.encoder(x)
        if isinstance(feat_maps, (tuple, list)):
            feat_maps = feat_maps[0]
        # feat_maps expected shape: (B, C, H, W)
        pooled = self.pool(feat_maps)          # -> (B, C, 1, 1)
        flattened = pooled.view(pooled.size(0), -1)  # -> (B, C)
        return flattened

    def forward(self, x):
        feat_maps = self.encoder(x)
        if isinstance(feat_maps, (tuple, list)):
            feat_maps = feat_maps[0]
        pooled = self.pool(feat_maps)
        flattened = pooled.view(pooled.size(0), -1)
        return self.fc(flattened)


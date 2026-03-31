import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.training import train_epoch, validate_model as validate_fn, test_model as test_fn

class ClassificationModelHandler:
    def __init__(self, device, num_classes, sequential_hyperparams):
        self.device = device
        self.num_classes = num_classes
        self.sequential_hyperparams = sequential_hyperparams
        self.model = nn.Sequential(
            nn.Linear(sequential_hyperparams['input_dim'], 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        ).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=sequential_hyperparams['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self, train_loader):
        """Train the model for one epoch using training.py utilities"""
        epoch_loss, epoch_acc = train_epoch(
            model=self.model,
            dataloader=train_loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
            extract_features=False
        )
        return epoch_loss

    def validate_model(self, val_loader):
        """Validate the model using training.py utilities"""
        val_loss, accuracy, precision, recall, f1, auc = validate_fn(
            model=self.model,
            dataloader=val_loader,
            criterion=self.criterion,
            device=self.device,
            extract_features=False
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        return val_loss, metrics

    def test_model(self, test_loader, output_file, method_name):
        """Test the model using training.py utilities"""
        hyperparams = {
            'method': method_name,
            'input_dim': self.sequential_hyperparams['input_dim'],
            'learning_rate': self.sequential_hyperparams['learning_rate'],
            'num_classes': self.num_classes
        }
        
        test_fn(
            model=self.model,
            dataloader=test_loader,
            criterion=self.criterion,
            device=self.device,
            hyperparams=hyperparams,
            output_file=output_file,
            extract_features=False,
            compute_detailed_metrics=True
        )

    def classify_from_npy(self, features_file, labels_file, output_file=None, method_name="NPY Classification"):
        # Load features and labels from .npy files
        features = np.load(features_file)
        labels = np.load(labels_file)
        
        # Convert to PyTorch tensors
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        # Create dataset and dataloader
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Perform classification
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for batch_features, batch_labels in dataloader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions = outputs.argmax(dim=1)
                
                all_labels.extend(batch_labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        }
        
        # Save results if output file is provided
        if output_file:
            with open(output_file, "a") as f:
                f.write(f"\n[{method_name}]\n")
                f.write(f"Features file: {features_file}\n")
                f.write(f"Labels file: {labels_file}\n")
                for metric, value in metrics.items():
                    f.write(f"{metric.capitalize()}: {value:.4f}\n")
                f.write("-" * 50 + "\n")
            print(f"Metrics for {method_name} saved to {output_file}.")
        
        return metrics

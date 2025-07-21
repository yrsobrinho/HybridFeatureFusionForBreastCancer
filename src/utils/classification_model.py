import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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
        self.model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate_model(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        }
        return val_loss / len(val_loader), metrics

    def test_model(self, test_loader, output_file, method_name):
        self.model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                predictions = outputs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions, zero_division=0),
            'recall': recall_score(all_labels, all_predictions, zero_division=0),
            'f1': f1_score(all_labels, all_predictions, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else float('nan')
        }

        with open(output_file, "a") as f:
            f.write(f"\n[Fusion Method: {method_name}]\n")
            for metric, value in metrics.items():
                f.write(f"{metric.capitalize()}: {value:.4f}\n")
            f.write("-" * 50 + "\n")

        print(f"Metrics for {method_name} saved to {output_file}.")

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from tqdm import tqdm
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ClassicalModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassicalModel, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

    def train_model(self, dataloader, criterion, optimizer, device):
        self.train()
        total_loss = 0.0
        total_accuracy = 0.0
        print("\n[Classical Model Training]")
        for inputs, labels in tqdm(dataloader, desc="Training Batches", leave=False):
            inputs, labels = inputs.to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_accuracy += (predicted == labels).sum().item()
        print(f"Training Loss: {total_loss / len(dataloader):.4f}, Accuracy: {total_accuracy / len(dataloader.dataset):.4f}")
        return total_loss / len(dataloader), total_accuracy / len(dataloader.dataset)

    def validate_model(self, dataloader, criterion, device):
        self.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []

        print("\n[Classical Model Validation]")
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation Batches", leave=False):
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Probabilities for class 1

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

        return total_loss / len(dataloader), accuracy, precision, recall, f1, auc
    
    def test_model(self, dataloader, criterion, device, hyperparams, output_file):
        self.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []

        print("\n[Classical Model Testing]")
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Testing Batches", leave=False):
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())  # Probabilities for class 1

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

        print(f"Test Loss: {total_loss / len(dataloader):.4f}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")

    def extract_features(self, dataloader, device):
        self.eval()
        features = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                output = self.resnet[:-1](inputs)  
                features.append(output.cpu().numpy())
        return np.concatenate(features, axis=0)
    
class ClassicalFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.resnet = original_model.resnet
        # Remove the final classification layer and get features from avgpool
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x
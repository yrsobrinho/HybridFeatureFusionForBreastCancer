import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
import datetime
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class QuantumModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=1, num_classes=2, output_size=(25, 25), patch_size=4):
        super(QuantumModel, self).__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.output_size = output_size
        self.patch_size = patch_size

        self.input_layer = nn.Linear(self.n_qubits, self.n_qubits)

        self.rand_params = nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True))

        self.circuit = self.define_circuit()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(output_size[0] * output_size[1] * n_qubits, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) 
        print(f"Before quanvolution: {x.shape}")
        x = self.quanvolution_batch(x)
        print(f"After quanvolution: {x.shape}")
        x = torch.relu(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def define_circuit(self):
        dev = qml.device("lightning.qubit", wires=self.n_qubits)

        @qml.qnode(dev, interface='torch')
        def circuit(phi, weights):
            for j in range(self.n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            qml.templates.layers.RandomLayers(weights, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        return circuit

    def quanvolution(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1) 

        padded_height = (image.shape[0] + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_width = (image.shape[1] + self.patch_size - 1) // self.patch_size * self.patch_size
        padded_image = np.zeros((padded_height, padded_width, image.shape[2]))
        padded_image[:image.shape[0], :image.shape[1], :] = image

        height_patches = padded_image.shape[0] // self.patch_size
        width_patches = padded_image.shape[1] // self.patch_size

        patches = padded_image.reshape(height_patches, self.patch_size, width_patches, self.patch_size, image.shape[2])
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(height_patches * width_patches, self.patch_size * self.patch_size * image.shape[2])

        patches_tensor = torch.tensor(patches, dtype=torch.float32).to(self.rand_params.device)

        q_results = torch.stack([torch.tensor(self.circuit(patch, self.rand_params)) for patch in patches_tensor])

        q_results = q_results.view(height_patches, width_patches, self.n_qubits)

        return q_results

    def quanvolution_batch(self, images):
        """Apply quantum convolution to a batch of images."""
        batch_size = images.shape[0]
        processed = [
            self.quanvolution(images[i].detach().cpu().numpy())
            for i in range(batch_size)
        ]
        processed = np.array(processed)
        return torch.tensor(processed, dtype=torch.float32).to(images.device)

    def train_model(self, train_loader, criterion, optimizer, device):
        self.train()
        total_loss = 0.0
        total_correct = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = self(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        accuracy = total_correct / len(train_loader.dataset)
        return total_loss / len(train_loader), accuracy
    
    def validate_model(self, dataloader, criterion, device):
        self.eval()
        total_loss = 0.0
        all_labels = []
        all_predictions = []
        all_probs = []

        print("\n[Quantum Model Validation]")
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation Batches", leave=False):
                images, labels = images.squeeze(1).to(device), labels.squeeze().to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()) 

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

        print("\n[Quantum Model Testing]")
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Testing Batches", leave=False):
                images, labels = images.squeeze(1).to(device), labels.squeeze().to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")

        with open(output_file, "a") as f:
            f.write(f"\n[Quantum Model Testing - {datetime.datetime.now()}]\n")
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


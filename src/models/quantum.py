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
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class QuantumModel(nn.Module):
    def __init__(self, n_qubits=4, n_layers=10, num_classes=2, output_size=(14, 14), patch_size=2):
        super(QuantumModel, self).__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.n_layers = n_layers
        self.output_size = output_size
        self.patch_size = patch_size

        # Linear layer to resize input features to match the number of qubits
        self.input_layer = nn.Linear(self.n_qubits, self.n_qubits)

        # Define trainable parameters for the quantum circuit
        self.rand_params = nn.Parameter(torch.randn(n_layers, n_qubits, requires_grad=True))

        # Define the quantum circuit
        self.circuit = self.define_circuit()

        # Fully connected layer for classification
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(output_size[0] * output_size[1] * n_qubits, num_classes)

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
    
        height_patches = image.shape[0] // self.patch_size
        width_patches = image.shape[1] // self.patch_size

        out = np.zeros((height_patches, width_patches, self.n_qubits))

        for j in range(height_patches):
            for k in range(width_patches):
                patch = []
                for i in range(self.patch_size):
                    for l in range(self.patch_size):
                        if (j * self.patch_size + i < image.shape[0]) and (k * self.patch_size + l < image.shape[1]):
                            patch.append(image[j * self.patch_size + i, k * self.patch_size + l])
                        else:
                            patch.append(0)

                q_results = self.circuit(patch, self.rand_params)

                for c in range(self.n_qubits):
                    out[j, k, c] = q_results[c].mean().item()

        return out

    def quanvolution_batch(self, images):
        """Apply quantum convolution to a batch of images."""
        batch_size = images.shape[0]
        processed = [
            self.quanvolution(images[i].detach().cpu().numpy())
            for i in range(batch_size)
        ]
        processed = np.array(processed)
        return torch.tensor(processed, dtype=torch.float32).to(images.device)

    def forward(self, x):
        # Apply quantum convolution
        x = x.permute(0, 2, 3, 1)  # Change to (batch_size, height, width, channels)
        x = self.quanvolution_batch(x)
        x = torch.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

    def train_model(self, dataloader, optimizer, criterion, epochs, device):
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            all_labels = []
            all_predictions = []

            print(f"\nEpoch {epoch + 1}/{epochs} [Quantum Model Training]")
            for images, labels in tqdm(dataloader, desc="Training Batches", leave=False):
                images, labels = images.squeeze(1).to(device), labels.squeeze().to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

            accuracy = accuracy_score(all_labels, all_predictions)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

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


    def extract_features_quantum(model, dataloader, dataset_name, output_dir, device):
        os.makedirs(output_dir, exist_ok=True)

        features_list, labels_list = [], []

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting Quantum Features ({dataset_name})"):
                images = images.to(device)

                # Extract features using the quantum model
                features = model(images)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        # Concatenate all features and labels
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)

        # Save features and labels as .npy files
        np.save(os.path.join(output_dir, f"{dataset_name}_features.npy"), features_array)
        np.save(os.path.join(output_dir, f"{dataset_name}_labels.npy"), labels_array)

        print(f"{dataset_name} quantum features saved in {output_dir}.")
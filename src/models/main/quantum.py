import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class QuanvLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, patch_size, mode="both"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.mode = mode

        self.dev = qml.device("lightning.gpu", wires=n_qubits)

        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.QNode(circuit, self.dev, interface="torch", diff_method="adjoint")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def quanv1d_horizontal(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        pad_w = (p - (W % p)) % p
        pad_h = (p - (H % p)) % p
        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        out_w = W // p
        outputs = []

        for i in range(H):
            row = x[:, 0, i, :]             
            patches = []
            for j in range(0, W, p):
                patch = row[:, j:j+p]        
                if patch.shape[1] != p:
                    continue
                patch_in = patch
                if patch_in.shape[1] != self.n_qubits:
                    if patch_in.shape[1] >= self.n_qubits:
                        patch_in = patch_in[:, :self.n_qubits]
                    else:
                        pad_size = self.n_qubits - patch_in.shape[1]
                        pad_tensor = torch.zeros(B, pad_size, device=patch_in.device, dtype=patch_in.dtype)
                        patch_in = torch.cat([patch_in, pad_tensor], dim=1)
                q_out = self.qlayer(patch_in)  
                patches.append(q_out)
            if len(patches) == 0:
                # fallback zeros
                patches = [torch.zeros(B, self.n_qubits, device=x.device, dtype=x.dtype) for _ in range(out_w)]
            patches = torch.stack(patches, dim=2)  
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=2)      
        return outputs

    def quanv1d_vertical(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        pad_w = (p - (W % p)) % p
        pad_h = (p - (H % p)) % p
        if pad_w != 0 or pad_h != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape

        out_h = H // p
        outputs = []

        for j in range(W):
            col = x[:, 0, :, j]            
            patches = []
            for i in range(0, H, p):
                patch = col[:, i:i+p]       
                if patch.shape[1] != p:
                    continue
                patch_in = patch
                if patch_in.shape[1] != self.n_qubits:
                    if patch_in.shape[1] >= self.n_qubits:
                        patch_in = patch_in[:, :self.n_qubits]
                    else:
                        pad_size = self.n_qubits - patch_in.shape[1]
                        pad_tensor = torch.zeros(B, pad_size, device=patch_in.device, dtype=patch_in.dtype)
                        patch_in = torch.cat([patch_in, pad_tensor], dim=1)
                q_out = self.qlayer(patch_in) 
                patches.append(q_out)
            if len(patches) == 0:
                patches = [torch.zeros(B, self.n_qubits, device=x.device, dtype=x.dtype) for _ in range(out_h)]
            patches = torch.stack(patches, dim=2)    
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=3)       
        return outputs

    def forward(self, x):
        x = self.quanv1d_horizontal(x)
        print("After horizontal quanv:", x.shape)
        x = self.quanv1d_vertical(x)
        print("After vertical quanv:", x.shape)
        return x

class QuantumModelWithEncoder(nn.Module):
    def __init__(self, n_qubits, n_layers, num_classes,
                 patch_size, encoder: nn.Module = None, encoder_output_size: int = 28):
        """
        encoder_output_size: if provided, resizes encoder output spatial dims to (encoder_output_size, encoder_output_size)
        """
        super().__init__()
        self.encoder = encoder

        self.quanv = QuanvLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            patch_size=patch_size,
        )

        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.encoder_output_size = encoder_output_size

        self.fc1 = None
        self.fc2 = None

    def _apply_encoder(self, x):
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
            B, F = enc.shape
            s = int(math.sqrt(F))
            if s * s == F:
                enc = enc.view(B, 1, s, s)
            else:
                enc = enc.view(B, 1, 1, F)
        else:
            raise RuntimeError(f"Unsupported encoder output ndim={enc.ndim}")

        if self.encoder_output_size is not None:
            target = self.encoder_output_size
            _, _, h, w = enc.shape
            if (h, w) != (target, target):
                mode = 'bilinear' if h > 1 and w > 1 else 'bilinear'
                enc = F.interpolate(enc, size=(target, target), mode=mode, align_corners=False)
        return enc

    def forward(self, x):
        print("Before encoder:", x.shape)
        x = self._apply_encoder(x)
        print("After encoder:", x.shape)
        if x.shape[1] != 1:
           x = x.mean(dim=1, keepdim=True)
        print("Starts Quanvolution, shape:", x.shape)
        x = self.quanv(x) 
        print("Finishes Quanvolution, shape:", x.shape)
        x = x.flatten(start_dim=1)  

        # lazy FC creation
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
        """Run encoder + quanv and return the flattened feature vector (before FCs)."""
        print("Before encoder (extract_features):", getattr(x, "shape", None))
        x = self._apply_encoder(x)
        print("After encoder (extract_features):", getattr(x, "shape", None))
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        print("Starts Quanvolution (extract_features), shape:", x.shape)
        x = self.quanv(x)
        print("Finishes Quanvolution (extract_features), shape:", x.shape)
        x = x.flatten(start_dim=1)

        # ensure fc layers exist and return the 64-dim feature vector (ReLU of fc1)
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        feat64 = F.relu(self.fc1(x))
        return feat64

    def classify_from_flat(self, x):
        """Given flattened features x, ensure FCs exist and return log-probabilities."""
        print("Classify from flat, input shape:", getattr(x, "shape", None))
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        if x.shape[1] == self.fc1.out_features:
            out = self.fc2(x)
        else:
            out = F.relu(self.fc1(x))
            out = self.fc2(out)
        return F.log_softmax(out, dim=1)


class QuantumModel(nn.Module):
    """Quantum model variant that does not use an encoder.

    Accepts raw input tensors (B, C, H, W). If C != 1 the model will average across
    channels similar to the encoder-backed model. Provides extract_features and
    classify_from_flat for efficient feature saving and training.
    """
    def __init__(self, n_qubits, n_layers, num_classes, patch_size):
        super().__init__()
        self.quanv = QuanvLayer(n_qubits=n_qubits, n_layers=n_layers, patch_size=patch_size)
        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = self.quanv(x)
        x = x.flatten(start_dim=1)

        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def extract_features(self, x):
            """Run quanv (and channel averaging) and return flattened features before FCs."""
            print("Starts Quanvolution (extract_features), input shape:", getattr(x, "shape", None))
            if x.shape[1] != 1:
                x = x.mean(dim=1, keepdim=True)
            print("After channel-average (extract_features), shape:", x.shape)
            x = self.quanv(x)
            print("Finishes Quanvolution (extract_features), shape:", x.shape)
            x = x.flatten(start_dim=1)

            if self.fc1 is None:
                in_features = x.shape[1]
                self.fc1 = nn.Linear(in_features, 64).to(x.device)
                self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

            feat64 = F.relu(self.fc1(x))
            return feat64

    def classify_from_flat(self, x):
        """Given flattened features x, ensure FCs exist and return log-probabilities."""
        print("Classify from flat (QuantumModel), input shape:", getattr(x, "shape", None))
        if self.fc1 is None:
            in_features = x.shape[1]
            self.fc1 = nn.Linear(in_features, 64).to(x.device)
            self.fc2 = nn.Linear(64, self.num_classes).to(x.device)

        if x.shape[1] == self.fc1.out_features:
            out = self.fc2(x)
        else:
            out = F.relu(self.fc1(x))
            out = self.fc2(out)
        return F.log_softmax(out, dim=1)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs, save_features_dir=None):
    """Train model; if save_features_dir is provided, save train features and labels as .npy"""
    model.to(device)
    history = {"train_loss": [], "train_acc": []}

    save_feats = save_features_dir is not None
    if save_feats:
        os.makedirs(save_features_dir, exist_ok=True)
        train_feats_list = []
        train_labels_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in pbar:
            images = images.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()

            feats = model.extract_features(images)
            outputs = model.classify_from_flat(feats)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if save_feats:
                train_feats_list.append(feats.cpu().numpy())
                train_labels_list.append(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}")

    if save_feats and len(train_feats_list) > 0:
        feats_arr = np.concatenate(train_feats_list, axis=0)
        labs_arr = np.concatenate(train_labels_list, axis=0)
        np.save(os.path.join(save_features_dir, "train_features.npy"), feats_arr)
        np.save(os.path.join(save_features_dir, "train_labels.npy"), labs_arr)

    return history

def validate_model(model, dataloader, criterion, device, save_features_dir=None):
    """Validate model; if save_features_dir is provided, save validation features and labels as .npy"""
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    save_feats = save_features_dir is not None
    if save_feats:
        os.makedirs(save_features_dir, exist_ok=True)
        val_feats_list = []
        val_labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            feats = model.extract_features(images)
            outputs = model.classify_from_flat(feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if save_feats:
                val_feats_list.append(feats.cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    if save_feats and len(val_feats_list) > 0:
        feats_arr = np.concatenate(val_feats_list, axis=0)
        labs_arr = np.concatenate(val_labels_list, axis=0)
        np.save(os.path.join(save_features_dir, "val_features.npy"), feats_arr)
        np.save(os.path.join(save_features_dir, "val_labels.npy"), labs_arr)

    return avg_loss, accuracy, precision, recall, f1, auc

def test_model(model, dataloader, criterion, device, hyperparams, output_file="test_results.txt", save_features_dir=None):
    """Test model; if save_features_dir is provided, save test features and labels as .npy"""
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    save_feats = save_features_dir is not None
    if save_feats:
        os.makedirs(save_features_dir, exist_ok=True)
        test_feats_list = []
        test_labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            feats = model.extract_features(images)
            outputs = model.classify_from_flat(feats)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if save_feats:
                test_feats_list.append(feats.cpu().numpy())
                test_labels_list.append(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    avg_loss = total_loss / len(dataloader)

    with open(output_file, "a") as f:
        f.write(f"\n[Quantum Model Testing - {datetime.datetime.now()}]\n")
        f.write(f"Hyperparameters: {hyperparams}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("-" * 50 + "\n")

    if save_feats and len(test_feats_list) > 0:
        feats_arr = np.concatenate(test_feats_list, axis=0)
        labs_arr = np.concatenate(test_labels_list, axis=0)
        np.save(os.path.join(save_features_dir, "test_features.npy"), feats_arr)
        np.save(os.path.join(save_features_dir, "test_labels.npy"), labs_arr)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return avg_loss, accuracy, precision, recall, f1, auc


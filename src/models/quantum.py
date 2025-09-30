import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
import pennylane as qml


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
        out_w = W // self.patch_size
        outputs = []

        for i in range(H): 
            row = x[:, 0, i, :]              # [B, W]
            patches = []
            for j in range(0, W, self.patch_size):
                patch = row[:, j:j+self.patch_size]  # [B, patch_size]
                if patch.shape[1] != self.n_qubits:
                    continue
                q_out = self.qlayer(patch)           # [B, n_qubits]
                patches.append(q_out)
            patches = torch.stack(patches, dim=2)    # [B, n_qubits, out_w]
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=2)        # [B, n_qubits, H, out_w]
        return outputs

    def quanv1d_vertical(self, x):
        B, C, H, W = x.shape
        out_h = H // self.patch_size
        outputs = []

        for j in range(W): 
            col = x[:, 0, :, j]              # [B, H]
            patches = []
            for i in range(0, H, self.patch_size):
                patch = col[:, i:i+self.patch_size]  # [B, patch_size]
                if patch.shape[1] != self.n_qubits:
                    continue
                q_out = self.qlayer(patch)           # [B, n_qubits]
                patches.append(q_out)
            patches = torch.stack(patches, dim=2)    # [B, n_qubits, out_h]
            outputs.append(patches)

        outputs = torch.stack(outputs, dim=3)        # [B, n_qubits, out_h, W]
        return outputs
    
    def quanv1d_interlaced(self, h_out, v_out):
        interlaced_tensor = torch.stack((h_out, v_out), dim = 3)
        return interlaced_tensor
    
    def aggregate_sobel(self, h_out, v_out, method):
        if method == "euclidean":
            v_out_transposed = v_out.transpose(2, 3)
            return torch.sqrt(h_out**2 + v_out_transposed**2)
        elif method == "sum":
            return h_out + v_out
        elif method == "abs_sum":
            return torch.abs(h_out) + torch.abs(v_out)
        else:
            raise ValueError(f"Invalid aggregation method: {method}")
    

    def forward(self, x):
        if self.mode == "horizontal":
            return self.quanv1d_horizontal(x)
        elif self.mode == "vertical":
            return self.quanv1d_vertical(x)
        elif self.mode == "both":
            print("Beginning horizontal processing...")
            h_out = self.quanv1d_horizontal(x)
            print("Beginning vertical processing...")
            v_out = self.quanv1d_vertical(x)
            print("Aggregation...")
            return self.aggregate_sobel(h_out, v_out, method="euclidean") 
        elif self.mode == "interlaced":
            print("Beginning horizontal processing...")
            h_out = self.quanv1d_horizontal(x)
            print("Beginning vertical processing...")
            v_out = self.quanv1d_vertical(x)
            print("Interlacing...")
            return self.quanv1d_interlaced(h_out, v_out)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class QuantumModel(nn.Module):
    def __init__(self, n_qubits, n_layers, num_classes,
                 patch_size, input_size, mode="horizontal"):
        super().__init__()
        self.quanv = QuanvLayer(
            n_qubits=n_qubits,
            n_layers=n_layers,
            patch_size=patch_size,
            mode=mode
        )

        self.patch_size = patch_size
        self.n_qubits = n_qubits
        self.mode = mode
      
        dummy_input = torch.zeros(1, 1, input_size, input_size)  
        dummy_out = self.quanv(dummy_input)
        feat_size = dummy_out.numel() // dummy_out.shape[0]      

        self.fc1 = nn.Linear(feat_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        print("Passing through quanvolution layer...")
        x = self.quanv(x)       
        print("Quanvolution output shape:", x.shape)    
        x = x.flatten(start_dim=1)   
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, log_interval=50):
    model.to(device)
    history = {"train_loss": [], "train_acc": []}

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
            outputs = model(images) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                pbar.set_postfix({
                    "loss": f"{running_loss / (batch_idx+1):.4f}",
                    "acc": f"{correct / total:.4f}"
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}")

    return history

def validate_model(model, dataloader, criterion, device):
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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

    return avg_loss, accuracy, precision, recall, f1, auc


def test_model(model, dataloader, criterion, device, hyperparams, output_file="test_results.txt"):
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return avg_loss, accuracy, precision, recall, f1, auc


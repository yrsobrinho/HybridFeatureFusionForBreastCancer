import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import BreastMNISTLoader
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from models.quantum import QuantumModel
from models.classical import ClassicalModel
from utils.extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.methods import AttentionFusion, SelfAttentionBlock

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("CUDA is not available. Using CPU.")

# Directories for saving models and features
quantum_model_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/best_models/pipeline/quantum"
classical_model_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/best_models/pipeline/classical"
quantum_features_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/pipeline/quantum"
classical_features_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/pipeline/classical"
fused_features_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/pipeline/fused"

os.makedirs(quantum_model_dir, exist_ok=True)
os.makedirs(classical_model_dir, exist_ok=True)
os.makedirs(fused_features_dir, exist_ok=True)
os.makedirs(quantum_features_dir, exist_ok=True)
os.makedirs(classical_features_dir, exist_ok=True)


# Common  Hyperparameters
common_hyperparams = {
    "epochs": 20,
    "batch_size": 128,
    "num_classes": 2,
    "attention_dim": 128
}

# Hyperparameters for Quantum Model
quantum_hyperparams = {
    "n_qubits": 4,
    "n_layers": 1,
    "learning_rate": 1e-2
}

# Hyperparameters for Classical Model
classical_hyperparams = {
    "learning_rate": 1e-5
}

batch_size = common_hyperparams["batch_size"]

# Load BreastMNIST data using the custom dataloader
loader = BreastMNISTLoader(batch_size, image_size=28)
train_loader, test_loader, val_loader = loader.get_dataloaders()

# Common Hyperparameters 
num_classes = common_hyperparams["num_classes"]
epochs = common_hyperparams["epochs"]
criterion = nn.CrossEntropyLoss()

# Train Quantum Model
n_qubits = quantum_hyperparams["n_qubits"]
n_layers = quantum_hyperparams["n_layers"]
quantum_learning_rate = quantum_hyperparams["learning_rate"]

rand_params = np.random.uniform(low=0, high=2 * np.pi, size=(n_qubits, n_layers)) 
quantum_model = QuantumModel(n_qubits, n_layers, num_classes).to(device)
quantum_optimizer = optim.Adam(quantum_model.parameters(), lr=quantum_learning_rate)

best_val_acc = 0.0
for epoch in range(epochs):
    quantum_model.train_model(train_loader, quantum_optimizer, criterion, 1, device)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = quantum_model.validate_model(val_loader, criterion, device)
    print(f"Quantum Model - Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(quantum_model.state_dict(), os.path.join(quantum_model_dir, "best_quantum_model.pth"))

# Test Quantum Model
quantum_model.load_state_dict(torch.load(os.path.join(quantum_model_dir, "best_quantum_model.pth", weights_only=True)))
quantum_model.test_model(test_loader, criterion, device, quantum_hyperparams, "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/results/quantum_test_results.txt")

# Train Classical Model
classical_learning_rate = classical_hyperparams["learning_rate"]

classical_model = ClassicalModel(num_classes=num_classes).to(device)
classical_optimizer = optim.Adam(classical_model.parameters(), lr=classical_learning_rate)

best_val_acc = 0.0
for epoch in range(epochs):
    train_loss, train_acc = classical_model.train_model(train_loader, criterion, classical_optimizer, device)
    val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = classical_model.validate_model(val_loader, criterion, device)
    print(f"Classical Model - Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(classical_model.state_dict(), os.path.join(classical_model_dir, "best_classical_model.pth"))

# Test Classical Model
classical_model.load_state_dict(torch.load(os.path.join(classical_model_dir, "best_classical_model.pth"), weights_only=True))
classical_model.test_model(val_loader, criterion, device, classical_hyperparams, "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/results/classical_test_results.txt")

# Feature Extraction

# Initialize the QuantumModel
quantum_model = QuantumModel(n_qubits=4, n_layers=10, num_classes=2).to(device)
quantum_state_dict = torch.load(quantum_model_dir, map_location=device)
quantum_model.load_state_dict(quantum_state_dict)

# Initialize the ClassicalModel
classical_model = ClassicalModel(num_classes=2).to(device)
classical_state_dict = torch.load(classical_model_dir, map_location=device)
classical_model.load_state_dict(classical_state_dict)

quantum_extractor = FeatureExtractor(quantum_model, device=device)
classical_extractor = FeatureExtractor(classical_model, device=device)

quantum_output_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/pipeline/quantum"
classical_output_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/pipeline/classical"

quantum_extractor.extract_features_quantum(
    dataloader=train_loader,
    dataset_name="BreastMNIST",
    output_dir=quantum_output_dir
)

classical_extractor.extract_features_classical(
    dataloader=train_loader,
    dataset_name="BreastMNIST",
    output_dir=classical_output_dir
)

# Aggregation Methods
fusion_methods = ['concat', 'add', 'attention', 'self_attention', 'dot_product']
attention_module = AttentionFusion(common_hyperparams["attention_dim"]).to(device)
self_attention_module = SelfAttentionBlock(common_hyperparams["attention_dim"]).to(device)

quantum_features = np.load(os.path.join(quantum_features_dir, "BreastMNIST_features.npy"))
classical_features = np.load(os.path.join(classical_features_dir, "BreastMNIST_features.npy"))

for method in fusion_methods:
    if method == 'attention':
        fused_features = attention_module(torch.tensor(quantum_features), torch.tensor(classical_features))
    elif method == 'self_attention':
        fused_features = self_attention_module(torch.cat([torch.tensor(quantum_features), torch.tensor(classical_features)], dim=1))
    elif method == 'concat':
        fused_features = np.concatenate([quantum_features, classical_features], axis=1)
    elif method == 'add':
        fused_features = quantum_features + classical_features
    else:
        raise ValueError(f"Unsupported fusion method: {method}")
    np.save(os.path.join(fused_features_dir, f"fused_features_{method}.npy"), fused_features)

# Sequential Layer for Classification
input_dim = fused_features.shape[1]
classification_model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes)
).to(device)

sequential_hyperparams = {
    "learning_rate": 1e-3,
    "batch_size": 128
}

classical_labels = np.load(os.path.join(classical_features_dir, "BreastMNIST_labels.npy"))
classification_optimizer = optim.Adam(classification_model.parameters(), lr=sequential_hyperparams["learning_rate"])
criterion = nn.CrossEntropyLoss()

# Train the classification model
classification_features = torch.tensor(fused_features, dtype=torch.float32).to(device)
classification_labels = torch.tensor(classical_labels, dtype=torch.long).to(device)
dataset = torch.utils.data.TensorDataset(classification_features, classification_labels)
loader = DataLoader(dataset, batch_size=sequential_hyperparams["batch_size"], shuffle=True)

best_val_acc = 0.0
for epoch in range(epochs):
    # Training Phase
    classification_model.train()
    total_loss = 0.0
    for features, labels in loader:
        classification_optimizer.zero_grad()
        outputs = classification_model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        classification_optimizer.step()
        total_loss += loss.item()
    print(f"Classification Model - Epoch {epoch + 1}, Training Loss: {total_loss / len(loader):.4f}")

    # Validation Phase
    classification_model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader: 
            features, labels = features.to(device), labels.to(device)
            outputs = classification_model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    # Compute metrics
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions, zero_division=0)
    val_recall = recall_score(all_labels, all_predictions, zero_division=0)
    val_f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = float("nan")

    # Print metrics
    print(f"Classification Model - Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader):.4f}")
    print(f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
    print(f"AUC: {val_auc:.4f}")

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(classification_model.state_dict(), "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/best_models/pipeline/fused/best_classification_model.pth")
        print(f"Best model saved with Validation Accuracy: {best_val_acc:.4f}")

# Evaluate the classification model on the test dataset
def evaluate_classification_model(classification_model, test_loader, device, output_file, method_name):
    classification_model.eval()
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classification_model(features)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    # Save metrics to a file
    with open(output_file, "a") as f:
        f.write(f"\n[Fusion Method: {method_name}]\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("-" * 50 + "\n")

    print(f"Metrics for {method_name} saved to {output_file}.")

# Save metrics for each fusion method
output_file = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/results/fusion_metrics.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

for method in fusion_methods:
    fused_features = np.load(os.path.join(fused_features_dir, f"fused_features_{method}.npy"))
    classification_features = torch.tensor(fused_features, dtype=torch.float32).to(device)
    classification_labels = torch.tensor(classical_labels, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(classification_features, classification_labels)
    test_loader = DataLoader(dataset, batch_size=sequential_hyperparams["batch_size"], shuffle=False)

    evaluate_classification_model(classification_model, test_loader, device, output_file, method)
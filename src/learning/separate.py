import sys
sys.path.insert(0, '/home/eflammere/HybridEmbeddingsForBreastCancer')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.dataloader import BreastMNISTLoader
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from src.models.quantum import QuantumModel
from src.models.classical import ClassicalModel 
from src.utils.extraction import QuantumFeatureExtractor, ClassicalFeatureExtractor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.methods import AttentionFusion, SelfAttentionBlock
from src.utils.fusion_methods import FusionMethods
from src.utils.classification_model import ClassificationModelHandler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("CUDA is not available. Using CPU.")

def setup_directories():
    directories = {
        "quantum_model_dir": "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/quantum",
        "classical_model_dir": "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/classical",
        "quantum_features_dir": "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/quantum",
        "classical_features_dir": "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/classical",
        "fused_features_dir": "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/fused",
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    return directories

# Common Hyperparameters
common_hyperparams = {
    "epochs": 10,
    "batch_size": 32,
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

loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=28)
train_loader, test_loader, val_loader = loader.get_dataloaders()
dataloader = {"train": train_loader, "test": test_loader, "val": val_loader}

# Common Hyperparameters
num_classes = common_hyperparams["num_classes"]
epochs = common_hyperparams["epochs"]
criterion = nn.CrossEntropyLoss()

def train_and_validate_model(model, dataloader, criterion, optimizer, device, epochs, model_dir, model_name):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = model.train_model(dataloader["train"], criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = model.validate_model(dataloader["val"], criterion, device)
        print(f"{model_name} - Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, f"best_{model_name.lower()}_model.pth"))
    return best_val_acc

# Train Classical Model
classical_model = ClassicalModel(num_classes=num_classes).to(device)
classical_optimizer = optim.Adam(classical_model.parameters(), lr=classical_hyperparams["learning_rate"])
train_and_validate_model(classical_model, dataloader, criterion, classical_optimizer, device, epochs, "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/classical", "Classical")
print(count_parameters(classical_model))

# # Test Classical Model
classical_model.load_state_dict(torch.load(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/classical", "best_classical_model.pth"), weights_only=True))
classical_model.test_model(test_loader, criterion, device, classical_hyperparams, "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/results/classical_test_results.txt")

# # Train Quantum Model
n_qubits = quantum_hyperparams["n_qubits"]
n_layers = quantum_hyperparams["n_layers"]
quantum_learning_rate = quantum_hyperparams["learning_rate"]

rand_params = np.random.uniform(low=0, high=2 * np.pi, size=(n_qubits, n_layers)) 
quantum_model = QuantumModel(n_qubits, n_layers, num_classes).to(device)
quantum_optimizer = optim.Adam(quantum_model.parameters(), lr=quantum_hyperparams["learning_rate"])
print(count_parameters(quantum_model))
train_and_validate_model(quantum_model, dataloader, criterion, quantum_optimizer, device, epochs, "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/quantum", "Quantum")

# Test Quantum Model
quantum_model.load_state_dict(torch.load(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/quantum", "best_quantum_model.pth"), weights_only=True))
quantum_model.test_model(test_loader, criterion, device, quantum_hyperparams, "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/results/quantum_test_results.txt")

# Feature Extraction

quantum_model_path = os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/quantum", "best_quantum_model.pth")
classical_model_path = os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/classical", "best_classical_model.pth")

quantum_output_dir = "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/quantum"
classical_output_dir = "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/classical"

# 2. Carregue os pesos salvos
quantum_state_dict = torch.load(quantum_model_path, map_location=device)
classical_state_dict = torch.load(classical_model_path, map_location=device)

quantum_model.load_state_dict(quantum_state_dict)
classical_model.load_state_dict(classical_state_dict)

extractor_classical = ClassicalFeatureExtractor(classical_model).to(device)
extractor_quantum = QuantumFeatureExtractor(quantum_model).to(device)

extractor_classical.extract_features(dataloader, device, classical_output_dir, "BreastCancer")
extractor_quantum.extract_features(dataloader, device, quantum_output_dir, "BreastCancer")

# Ensure attention modules are initialized
attention_module = AttentionFusion(common_hyperparams["attention_dim"]).to(device)
self_attention_module = SelfAttentionBlock(common_hyperparams["attention_dim"]).to(device)

# Ensure quantum and classical features are loaded
quantum_features = np.load(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/quantum", "BreastCancer_features.npy"))
classical_features = np.load(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/classical", "BreastCancer_features.npy"))

# Ensure fused_features_dir is defined
fused_features_dir = "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/fused"

# Initialize FusionMethods class
fusion_methods_instance = FusionMethods(device, common_hyperparams["attention_dim"])

fusion_methods = {
    'concat': fusion_methods_instance.fuse_concat,
    'add': fusion_methods_instance.fuse_add,
    'dot': fusion_methods_instance.fuse_dot,
    'attention': fusion_methods_instance.fuse_attention,
    'self_attention': fusion_methods_instance.fuse_self_attention
}

for method_name, fusion_function in fusion_methods.items():
    if method_name in ['attention', 'self_attention']:
        # Adaptation epochs for attention-based methods
        for epoch in range(5):  # Example adaptation epochs
            attention_module.train() if method_name == 'attention' else self_attention_module.train()
    
    fused_features = fusion_function(quantum_features, classical_features)
    np.save(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/fused",
                         f"fused_features_{method_name}.npy"), fused_features)

# Sequential Layer for Classification
input_dim = fused_features.shape[1]
classification_model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, num_classes=2)
).to(device)

sequential_hyperparams = {
    "epochs": 10,
    "learning_rate": 1e-3,
    "batch_size": 128
}

classical_labels = np.load(os.path.join("/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/features/classical", "BreastCancer_labels.npy"))
classification_optimizer = optim.Adam(classification_model.parameters(), lr=sequential_hyperparams["learning_rate"])
criterion = nn.CrossEntropyLoss()

classification_features = torch.tensor(fused_features, dtype=torch.float32).to(device)
classification_labels = torch.tensor(classical_labels, dtype=torch.long).to(device)
dataset = torch.utils.data.TensorDataset(classification_features, classification_labels)
loader = DataLoader(dataset, batch_size=sequential_hyperparams["batch_size"], shuffle=True)

# Initialize ClassificationModelHandler
classification_handler = ClassificationModelHandler(device, num_classes, {
    'input_dim': fused_features.shape[1],
    'epochs': sequential_hyperparams['epochs'],
    'learning_rate': sequential_hyperparams['learning_rate'],
    'batch_size': sequential_hyperparams['batch_size']
})

# Train the classification model
classification_features = torch.tensor(fused_features, dtype=torch.float32).to(device)
classification_labels = torch.tensor(classical_labels, dtype=torch.long).to(device)
dataset = torch.utils.data.TensorDataset(classification_features, classification_labels)
loader = DataLoader(dataset, batch_size=classification_handler.sequential_hyperparams['batch_size'], shuffle=True)

best_val_acc = 0.0
for epoch in range(classification_handler.sequential_hyperparams['epochs']):
    train_loss = classification_handler.train_model(loader)
    print(f"Classification Model - Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

    val_loss, metrics = classification_handler.validate_model(val_loader)
    print(f"Classification Model - Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")

    if metrics['accuracy'] > best_val_acc:
        best_val_acc = metrics['accuracy']
        torch.save(classification_handler.model.state_dict(), "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/best_models/fused/best_classification_model.pth")
        print(f"Best model saved with Validation Accuracy: {best_val_acc:.4f}")

# Evaluate the classification model on the test dataset
output_file = "/home/eflammere/HybridEmbeddingsForBreastCancer/experiments/separate/results/fusion_metrics.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

for method in fusion_methods:
    fused_features = np.load(os.path.join(fused_features_dir, f"fused_features_{method}.npy"))
    classification_features = torch.tensor(fused_features, dtype=torch.float32).to(device)
    classification_labels = torch.tensor(classical_labels, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(classification_features, classification_labels)
    test_loader = DataLoader(dataset, batch_size=classification_handler.sequential_hyperparams['batch_size'], shuffle=False)

    classification_handler.test_model(test_loader, output_file, method)
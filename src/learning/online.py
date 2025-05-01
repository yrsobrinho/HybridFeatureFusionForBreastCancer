import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel
from src.models.classical import ClassicalModel
from src.utils.methods import AttentionFusion, SelfAttentionBlock
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Using: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("CUDA is not available. Using CPU.")

# Directories
output_dir = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/online"
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
common_hyperparams = {
    "epochs": 20,
    "batch_size": 128,
    "num_classes": 2,
    "attention_dim": 128,
    "temperature": 3.0, 
    "alpha": 0.5        
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

sequential_hyperparams = {
    "learning_rate": 1e-3,
}

# Load data
loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=28)
train_loader, test_loader, val_loader = loader.get_dataloaders()

# Initialize models
quantum_model = QuantumModel(n_qubits=quantum_hyperparams["n_qubits"], n_layers=quantum_hyperparams["n_layers"], num_classes=common_hyperparams["num_classes"]).to(device)
classical_model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)

# Fusion module
fusion_methods = ['concat', 'add', 'attention', 'self_attention']
attention_module = AttentionFusion(common_hyperparams["attention_dim"]).to(device)
self_attention_module = SelfAttentionBlock(common_hyperparams["attention_dim"]).to(device)

# Classification layer
fusion_output_dim = common_hyperparams["attention_dim"] if 'attention' in fusion_methods else quantum_model.output_size + classical_model.output_size
classification_model = nn.Sequential(
    nn.Linear(fusion_output_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, common_hyperparams["num_classes"])
).to(device)

# Optimizers
quantum_optimizer = optim.Adam(quantum_model.parameters(), lr=quantum_hyperparams["learning_rate"])
classical_optimizer = optim.Adam(classical_model.parameters(), lr=classical_hyperparams["learning_rate"])
classification_optimizer = optim.Adam(classification_model.parameters(), lr=sequential_hyperparams["learning_rate"])

# Loss functions
criterion = nn.CrossEntropyLoss()
distillation_loss = nn.KLDivLoss(reduction="batchmean")

print("Starting training...")
for epoch in range(common_hyperparams["epochs"]):
    print(f"Epoch {epoch + 1}/{common_hyperparams['epochs']} - Training Phase")
    quantum_model.train()
    classical_model.train()
    classification_model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"  Processing batch {batch_idx + 1}/{len(train_loader)}")
        images, labels = images.to(device), labels.to(device)

        # Forward pass for both models
        quantum_features = quantum_model(images)
        print("  Quantum features extracted.")
        classical_features = classical_model(images)
        print("  Classical features extracted.")

        # Teacher's soft predictions
        teacher_logits = classical_features / common_hyperparams["temperature"]
        teacher_probs = torch.softmax(teacher_logits, dim=1)
        print("  Teacher's soft predictions computed.")

        # Student's predictions
        student_logits = quantum_features / common_hyperparams["temperature"]
        student_probs = torch.log_softmax(student_logits, dim=1)
        print("  Student's predictions computed.")

        # Fusion logic
        fused_features = None
        for method in fusion_methods:
            if method == 'concat':
                fused_features = torch.cat([quantum_features, classical_features], dim=1)
            elif method == 'add':
                fused_features = quantum_features + classical_features
            elif method == 'attention':
                fused_features = attention_module(quantum_features, classical_features)
            elif method == 'self_attention':
                fused_features = self_attention_module(torch.cat([quantum_features, classical_features], dim=1))
            print(f"  Fusion method '{method}' applied.")

        # Classification
        outputs = classification_model(fused_features)
        print("  Classification outputs computed.")

        # Compute losses
        ce_loss = criterion(outputs, labels) 
        kd_loss = distillation_loss(student_probs, teacher_probs) * (common_hyperparams["temperature"] ** 2)  
        total_loss = common_hyperparams["alpha"] * kd_loss + (1 - common_hyperparams["alpha"]) * ce_loss
        print(f"  Loss computed: CE Loss = {ce_loss.item():.4f}, KD Loss = {kd_loss.item():.4f}, Total Loss = {total_loss.item():.4f}")

        # Backward pass
        quantum_optimizer.zero_grad()
        classical_optimizer.zero_grad()
        classification_optimizer.zero_grad()
        total_loss.backward()
        quantum_optimizer.step()
        classical_optimizer.step()
        classification_optimizer.step()
        print("  Backward pass and optimizer step completed.")

    print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss.item():.4f}")

    # Validation Phase
    print(f"Epoch {epoch + 1}/{common_hyperparams['epochs']} - Validation Phase")
    quantum_model.eval()
    classical_model.eval()
    classification_model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            print(f"  Validating batch {batch_idx + 1}/{len(val_loader)}")
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            quantum_features = quantum_model(images)
            classical_features = classical_model(images)

            # Fusion
            fused_features = torch.cat([quantum_features, classical_features], dim=1)
            outputs = classification_model(fused_features)

            # Compute loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Collect predictions and probabilities
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

    print(f"Validation - Epoch {epoch + 1} completed.")
    print(f"  Loss: {val_loss / len(val_loader):.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

# Save models
print("Saving models...")
torch.save(quantum_model.state_dict(), os.path.join(output_dir, "quantum_model.pth"))
torch.save(classical_model.state_dict(), os.path.join(output_dir, "classical_model.pth"))
torch.save(classification_model.state_dict(), os.path.join(output_dir, "classification_model.pth"))
print("Models saved successfully.")
import sys
sys.path.insert(0, '/home/eflammere/HybridEmbeddingsForBreastCancerDetection')
import os
import torch
import torch.nn as nn
import torch.optim as optim
import jax.numpy as jnp
from src.utils.common import count_parameters, setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel 
from src.models.classical import ClassicalModel

def train_and_validate_model(model, dataloader, criterion, optimizer, device, epochs, model_dir, model_name):
    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = model.train_model(dataloader["train"], criterion, optimizer, device)
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = model.validate_model(dataloader["val"], criterion, device)
        print(f"{model_name} - Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(model_dir, f"best_{model_name.lower()}_model.pth"))
    return best_val_acc

def main():
    device = setup_device()
    directories = setup_directories()

    # Hyperparameters
    common_hyperparams = {
        "epochs": 10,
        "batch_size": 8,
        "num_classes": 2,
    }

    quantum_hyperparams = {
        "n_qubits": 10,
        "n_layers": 1,
        "learning_rate": 1e-2
    }

    classical_hyperparams = {
        "learning_rate": 1e-5
    }

    # Initialize models and train
    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=28)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "test": test_loader, "val": val_loader}

    criterion = nn.CrossEntropyLoss()

    # Train Classical Model
    classical_model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)
    classical_optimizer = optim.Adam(classical_model.parameters(), lr=classical_hyperparams["learning_rate"])
    train_and_validate_model(classical_model, dataloader, criterion, classical_optimizer, device, 
                           common_hyperparams["epochs"], directories["classical_model_dir"], "Classical")

    # Train Quantum Model
    quantum_model = QuantumModel(
        quantum_hyperparams["n_qubits"],
        quantum_hyperparams["n_layers"],
        common_hyperparams["num_classes"]
    ).to(device)

    print(f"Total trainable parameters: {count_parameters(quantum_model)}")

    # --- Optimizer ---
    optimizer = optim.Adam(
        quantum_model.parameters(),
        lr=quantum_hyperparams["learning_rate"]
    )

    # --- Train ---
    params_q, opt_state_q = quantum_model.train_model(
        dataloader["train"],
        criterion,
        optimizer,
        params_q,
        opt_state_q,
        device,
        common_hyperparams["epochs"]
    )

    # --- Validate ---
    quantum_model.validate_model(
        dataloader["val"], criterion, device, params_q
    )

    # --- Save model ---
    model_path = os.path.join(directories["quantum_model_dir"], "best_quantum_model.pth")
    torch.save(quantum_model.state_dict(), model_path)
    print(f"Best model saved at {model_path}")

if __name__ == "__main__":
    main()

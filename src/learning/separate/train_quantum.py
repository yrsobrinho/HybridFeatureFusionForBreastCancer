import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.common import count_parameters, setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel, train_model, validate_model  

def main():
    device = setup_device()
    directories = setup_directories()

    common_hyperparams = {
        "epochs": 1,
        "batch_size": 8,
        "num_classes": 2,
    }

    quantum_hyperparams = {
        "n_qubits": 7,
        "n_layers": 1,
        "learning_rate": 1e-2,
        "mode": "both",  
        "patch_size": 7     
    }

    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=28)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

    criterion = nn.CrossEntropyLoss()

    quantum_model = QuantumModel(
        n_qubits=quantum_hyperparams["n_qubits"],
        n_layers=quantum_hyperparams["n_layers"],
        num_classes=common_hyperparams["num_classes"],
        patch_size=quantum_hyperparams["patch_size"],
        mode=quantum_hyperparams["mode"],
        input_size=28
    ).to(device)

    print(f"Total trainable parameters: {count_parameters(quantum_model)}")

    optimizer = optim.Adam(
        quantum_model.parameters(),
        lr=quantum_hyperparams["learning_rate"]
    )

    best_val_acc = 0.0
    for epoch in range(common_hyperparams["epochs"]):
        train_loss, train_acc = train_model(
            model=quantum_model,
            train_loader=dataloader["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=5
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validate_model(
            model=quantum_model,
            dataloader=dataloader["val"],
            criterion=criterion,
            device=device
        )

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(directories["quantum_model_dir"], "best_quantum_model.pth")
            torch.save(quantum_model.state_dict(), model_path)
            print(f"Best model saved at {model_path}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = validate_model(
        model=quantum_model,
        dataloader=dataloader["test"],
        criterion=criterion,
        device=device
    )
    print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()

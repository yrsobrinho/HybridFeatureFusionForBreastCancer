import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.common import count_parameters, setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel, train_model, validate_model, test_model  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autoencoder_path",
        type=str,
        default="/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/separate/models/autoencoder/autoencoder.pth",
        help="Path to autoencoder .pth (state_dict) to use as encoder. If not present, pipeline uses raw images."
    )
    args = parser.parse_args()

    device = setup_device()
    directories = setup_directories()

    common_hyperparams = {
        "epochs": 1,
        "batch_size": 8,
        "num_classes": 2,
    }

    quantum_hyperparams = {
        "n_qubits": 4,
        "n_layers": 1,
        "learning_rate": 1e-2,
        "patch_size": 7
    }

    # load dataloaders (raw medmnist images)
    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

    # try to load Autoencoder and extract encoder module
    # import Autoencoder directly (no existence/import tests)
    from src.models.autoencoder import Autoencoder
    encoder_module = None
    if args.autoencoder_path:
        try:
            ae = Autoencoder()
        except TypeError:
            ae = Autoencoder(input_channels=1, img_size=28)
        state = torch.load(args.autoencoder_path, map_location="cpu", weights_only=True)
        state_dict = state.get("state_dict", state)
        try:
            ae.load_state_dict(state_dict)
        except Exception:
            ae.load_state_dict(state_dict, strict=False)
        ae.eval()
        encoder_module = ae
        for p in encoder_module.parameters():
            p.requires_grad = False
        encoder_module.to(device)
        print(f"[info] Autoencoder carregado de {args.autoencoder_path}")

    criterion = nn.CrossEntropyLoss()
    quantum_model = QuantumModel(
        n_qubits=quantum_hyperparams["n_qubits"],
        n_layers=quantum_hyperparams["n_layers"],
        num_classes=common_hyperparams["num_classes"],
        patch_size=quantum_hyperparams["patch_size"],
        encoder=encoder_module,
        encoder_output_size=28 
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, quantum_model.parameters()),
        lr=quantum_hyperparams["learning_rate"]
    )

    best_val_acc = 0.0
    for epoch in range(common_hyperparams["epochs"]):
        print(f"\nEpoch {epoch+1}/{common_hyperparams['epochs']}:")
        history = train_model(
            model=quantum_model,
            train_loader=dataloader["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=1
        )

        train_acc = history["train_acc"][-1]

        validation_metrics = validate_model(
            model=quantum_model,
            dataloader=dataloader["val"],
            criterion=criterion,
            device=device
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = validation_metrics

        print(f"Results - Train Accuracy: {train_acc:.4f}, Validation Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(directories["quantum_model_dir"], "best_quantum_model.pth")
            torch.save(quantum_model.state_dict(), model_path)
            print(f"  New best model saved (acc: {best_val_acc:.4f}) at {model_path}")

    hyperparams_for_logging = {**common_hyperparams, **quantum_hyperparams, "encoder_loaded": encoder_module is not None}

    test_results = test_model(
        model=quantum_model,
        dataloader=dataloader["test"],
        criterion=criterion,
        device=device,
        hyperparams=hyperparams_for_logging
    )

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_results

    print("\nFINAL TEST")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()

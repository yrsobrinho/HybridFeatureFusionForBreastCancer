import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.utils.training import (
    train_epoch,
    validate_model,
    test_model,
    extract_and_save_features,
    training_loop
)
from src.models.preprocessing.autoencoder import Autoencoder
from src.models.main.quantum import QuantumModel, QuantumModelWithEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract features and save to experiments dirs')
    parser.add_argument('--encoder', action='store_true', help='Use QuantumModelWithEncoder')
    parser.add_argument('--load-best', action='store_true', help='Load best model weights before extracting/training')
    parser.add_argument('--mode', type=str, default='2d', choices=['2d', '1d-sequential', 'hybrid'],
                        help='Quanvolution mode: 2d (default, efficient), 1d-sequential (legacy), or hybrid')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride for 2D quanvolution. If None, uses patch_size (non-overlapping)')
    args = parser.parse_args()

    autoencoder_path = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/models/autoencoder1/autoencoder1.pth"
    device = setup_device()
    directories = setup_directories()

    encoder_model_dir = os.path.join(directories.get("quantum_model_dir", "."), "encoder")
    no_encoder_model_dir = os.path.join(directories.get("quantum_model_dir", "."), "no-encoder")
    os.makedirs(encoder_model_dir, exist_ok=True)
    os.makedirs(no_encoder_model_dir, exist_ok=True)

    common_hyperparams = {
        "epochs": 10,
        "batch_size": 8,
        "num_classes": 2,
    }

    quantum_hyperparams = {
        "n_qubits": 4,
        "n_layers": 1,
        "learning_rate": 1e-2,
        "patch_size": 4
    }

    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

    encoder_module = None
    if args.encoder and autoencoder_path:
        try:
            ae = Autoencoder()
        except TypeError:
            ae = Autoencoder(input_channels=1, img_size=28)
        state = torch.load(autoencoder_path, map_location="cpu", weights_only=True)
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
        print(f"Autoencoder loaded from {autoencoder_path}")

    if args.encoder:
        quantum_model = QuantumModelWithEncoder(
            n_qubits=quantum_hyperparams["n_qubits"],
            n_layers=quantum_hyperparams["n_layers"],
            num_classes=common_hyperparams["num_classes"],
            patch_size=quantum_hyperparams["patch_size"],
            encoder=encoder_module,
            encoder_output_size=28,
            mode=args.mode,
            stride=args.stride
        ).to(device)
    else:
        quantum_model = QuantumModel(
            n_qubits=quantum_hyperparams["n_qubits"],
            n_layers=quantum_hyperparams["n_layers"],
            num_classes=common_hyperparams["num_classes"],
            patch_size=quantum_hyperparams["patch_size"],
            mode=args.mode,
            stride=args.stride
        ).to(device)
    
    print(f"\nQuantum Model Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Stride: {args.stride if args.stride else quantum_hyperparams['patch_size']} (default: patch_size)")
    print(f"  Patch size: {quantum_hyperparams['patch_size']}")
    print(f"  Qubits: {quantum_hyperparams['n_qubits']}")
    print(f"  Layers: {quantum_hyperparams['n_layers']}\n")

    if args.load_best:
        chosen_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
        model_path = os.path.join(chosen_dir, "best_quantum_model.pth")
        if os.path.isfile(model_path):
            state = torch.load(model_path, map_location=device)
            try:
                quantum_model.load_state_dict(state)
            except Exception:
                if isinstance(state, dict) and "state_dict" in state:
                    quantum_model.load_state_dict(state["state_dict"]) 
                else:
                    quantum_model.load_state_dict(state)
            quantum_model.eval()
            print(f"Loaded best quantum model from {model_path}")
        else:
            print(f"No best model found at {model_path}; continuing with initialized model")

    if args.extract:
        print("\nStarting Quantum feature extraction...")
        extract_and_save_features(
            quantum_model, 
            train_loader, 
            device, 
            directories["quantum_features_dir"], 
            "train",
            extract_features=True
        )
        extract_and_save_features(
            quantum_model, 
            val_loader, 
            device, 
            directories["quantum_features_dir"], 
            "val",
            extract_features=True
        )
        extract_and_save_features(
            quantum_model, 
            test_loader, 
            device, 
            directories["quantum_features_dir"], 
            "test",
            extract_features=True
        )
        return

    print("Starting Training")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, quantum_model.parameters()),
        lr=quantum_hyperparams["learning_rate"]
    )

    # Determine save directory based on encoder usage
    save_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
    
    # Run training loop
    history = training_loop(
        model=quantum_model,
        train_loader=dataloader["train"],
        val_loader=dataloader["val"],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=common_hyperparams["epochs"],
        save_dir=save_dir,
        model_name="quantum_model",
        extract_features=True, 
        save_best_only=True
    )

 
    print("Loading best model for final testing...")
    save_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
    model_path = os.path.join(save_dir, "best_quantum_model.pth")
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        quantum_model.load_state_dict(state)
        print(f"Loaded best model from {model_path}")
    else:
        print("Warning: No best model found, using final epoch weights")

    hyperparams_for_logging = {**common_hyperparams, **quantum_hyperparams, "encoder_loaded": encoder_module is not None}

    # Test the model
    test_results = test_model(
        model=quantum_model,
        dataloader=dataloader["test"],
        criterion=criterion,
        device=device,
        hyperparams=hyperparams_for_logging,
        output_file="test_results.txt",
        extract_features=True,
        compute_detailed_metrics=True
    )

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_results

    print("FINAL TEST RESULTS")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()

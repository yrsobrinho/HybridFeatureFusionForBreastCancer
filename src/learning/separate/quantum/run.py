import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from src.utils.common import count_parameters, setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.preprocessing.autoencoder import Autoencoder
from src.models.main.quantum import QuantumModel, QuantumModelWithEncoder, train_model, validate_model, test_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract features and save to experiments dirs')
    parser.add_argument('--encoder', action='store_true', help='Use QuantumModelWithEncoder')
    parser.add_argument('--load-best', action='store_true', help='Load best model weights before extracting/training')
    args = parser.parse_args()

    autoencoder_path = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/separate/models/autoencoder/autoencoder.pth"
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

    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

    encoder_module = None
    # load encoder only when --encoder is passed
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
            encoder_output_size=28
        ).to(device)
    else:
        quantum_model = QuantumModel(
            n_qubits=quantum_hyperparams["n_qubits"],
            n_layers=quantum_hyperparams["n_layers"],
            num_classes=common_hyperparams["num_classes"],
            patch_size=quantum_hyperparams["patch_size"],
        ).to(device)

    # optionally load best weights
    if args.load_best:
        model_path = os.path.join(directories.get("quantum_model_dir", "."), "best_quantum_model.pth")
        if os.path.isfile(model_path):
            state = torch.load(model_path, map_location=device)
            quantum_model.load_state_dict(state)
            quantum_model.eval()
            print(f"Loaded best quantum model from {model_path}")
        else:
            print(f"No best model found at {model_path}; continuing with initialized model")

    if args.extract:
        # extraction helper
        def _extract_and_save(model, loader, device, out_dir, split_name):
            os.makedirs(out_dir, exist_ok=True)
            model.eval()
            feats_list = []
            labels_list = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Extract {split_name}"):
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        imgs, labels = batch[0], batch[1]
                    else:
                        imgs = batch
                        labels = None
                    imgs = imgs.to(device)
                    feats = model.extract_features(imgs)
                    feats_list.append(feats.cpu().numpy())
                    if labels is not None:
                        if isinstance(labels, torch.Tensor):
                            labels = labels.cpu().numpy()
                        labels_list.append(labels)
            feats = np.concatenate(feats_list, axis=0)
            np.save(os.path.join(out_dir, f"{split_name}_features.npy"), feats)
            if len(labels_list) > 0:
                labs = np.concatenate(labels_list, axis=0)
                np.save(os.path.join(out_dir, f"{split_name}_labels.npy"), labs)
            print(f"Saved {split_name} features -> {out_dir}")

        print("\nStarting Quantum feature extraction...")
        _extract_and_save(quantum_model, train_loader, device, directories["quantum_features_dir"], "train")
        _extract_and_save(quantum_model, test_loader, device, directories["quantum_features_dir"], "test")
        _extract_and_save(quantum_model, val_loader, device, directories["quantum_features_dir"], "val")
        return

    # If not extracting, run the original training flow
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, quantum_model.parameters()),
        lr=quantum_hyperparams["learning_rate"]
    )

    best_val_acc = 0.0
    for epoch in range(common_hyperparams["epochs"]):
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

        print(f"Results - Train Accuracy: {train_acc:.4f}")
        print(
            f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )

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

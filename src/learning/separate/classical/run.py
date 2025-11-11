import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.main.classical import ClassicalModel, ClassicalModelWithEncoder
from src.models.preprocessing.autoencoder import Autoencoder
import numpy as np
import torch
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract features and save to experiments dirs')
    parser.add_argument('--encoder', action='store_true', help='Use ClassicalModelWithEncoder')
    parser.add_argument('--load-best', action='store_true', help='Load best model weights before extracting')
    args = parser.parse_args()

    device = setup_device()
    dirs = setup_directories()

    autoencoder_path = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/separate/models/autoencoder/autoencoder.pth"
    encoder_module = None
    if args.encoder and autoencoder_path:
        try:
            ae = Autoencoder()
        except TypeError:
            ae = Autoencoder(input_channels=1, img_size=28)
        # load state dict (handle both raw state_dict or {"state_dict": ...})
        state = torch.load(autoencoder_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        try:
            ae.load_state_dict(state_dict)
        except Exception:
            ae.load_state_dict(state_dict, strict=False)
        ae.eval()
        # prefer the encoder submodule if available
        encoder_module = ae.encoder if hasattr(ae, "encoder") else ae
        for p in encoder_module.parameters():
            p.requires_grad = False
        encoder_module.to(device)
        print(f"Autoencoder loaded from {autoencoder_path}")

    common_hyperparams = {
        "epochs": 1,
        "batch_size": 8,
        "num_classes": 2,
        "learning_rate": 1e-3
    }

    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()

    # choose model variant depending on --encoder
    if args.encoder:
        # encoder_output_size should be number of channels from encoder after pooling (default 64)
        model = ClassicalModelWithEncoder(num_classes=common_hyperparams["num_classes"], encoder=encoder_module, encoder_output_size=64).to(device)
    else:
        model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)

    # optionally load best weights
    if args.load_best:
        model_path = os.path.join(dirs.get("classical_model_dir", "."), "best_classical_model.pth")
        if os.path.isfile(model_path):
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model.eval()
            print(f"Loaded best classical model from {model_path}")
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

        print("\nStarting Classical feature extraction...")
        _extract_and_save(model, train_loader, device, dirs["classical_features_dir"], "train")
        _extract_and_save(model, test_loader, device, dirs["classical_features_dir"], "test")
        _extract_and_save(model, val_loader, device, dirs["classical_features_dir"], "val")
        return

    # If not extracting, run the original training flow
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=common_hyperparams["learning_rate"])

    best_val_acc = 0.0
    best_model_path = os.path.join(dirs.get("classical_model_dir", "."), "best_classical_model.pth")

    for epoch in range(common_hyperparams["epochs"]):
        print(f"\nEpoch {epoch+1}/{common_hyperparams['epochs']}:")
        train_loss, train_acc = model.train_model(train_loader, criterion, optimizer, device)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")

        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = model.validate_model(val_loader, criterion, device)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved (acc: {best_val_acc:.4f}) at {best_model_path}")

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = model.validate_model(test_loader, criterion, device)
    print("\nFINAL TEST")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    out_file = "classical_test_results.txt"
    hyperparams_for_logging = common_hyperparams
    try:
        model.test_model(test_loader, criterion, device, hyperparams_for_logging, out_file)
    except TypeError:
        with open(out_file, "a") as f:
            f.write(f"\n[Classical Model Testing]\nHyperparams: {hyperparams_for_logging}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {test_acc:.4f}\n")
            f.write(f"Precision: {test_prec:.4f}\n")
            f.write(f"Recall: {test_rec:.4f}\n")
            f.write(f"F1 Score: {test_f1:.4f}\n")
            f.write(f"AUC: {test_auc:.4f}\n")
            f.write("-" * 50 + "\n")

if __name__ == "__main__":
    main()

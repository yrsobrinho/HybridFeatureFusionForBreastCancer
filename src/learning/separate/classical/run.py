import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.classical import ClassicalModel

def main():

    device = setup_device()
    dirs = setup_directories()

    common_hyperparams = {
        "epochs": 1,
        "batch_size": 8,
        "num_classes": 2,
        "learning_rate": 1e-3
    }

    loader = BreastMNISTLoader(batch_size=common_hyperparams["batch_size"], image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()

    model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)
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

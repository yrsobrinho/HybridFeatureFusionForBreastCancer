import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.utils.training import (
    training_loop,
    test_model,
    extract_and_save_features
)
from src.models.main.classical import ClassicalModel, ClassicalModelWithEncoder
from src.models.preprocessing.autoencoder import Autoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract features and save to experiments dirs')
    parser.add_argument('--encoder', action='store_true', help='Use ClassicalModelWithEncoder')
    parser.add_argument('--load-best', action='store_true', help='Load best model weights before extracting')
    args = parser.parse_args()

    device = setup_device()
    dirs = setup_directories()

    encoder_model_dir = os.path.join(dirs.get("classical_model_dir", "."), "encoder")
    no_encoder_model_dir = os.path.join(dirs.get("classical_model_dir", "."), "no-encoder")
    os.makedirs(encoder_model_dir, exist_ok=True)
    os.makedirs(no_encoder_model_dir, exist_ok=True)

    autoencoder_path = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/models/autoencoder/autoencoder.pth"
    encoder_module = None
    if args.encoder and autoencoder_path:
        try:
            ae = Autoencoder()
        except TypeError:
            ae = Autoencoder(input_channels=1, img_size=28)
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

    if args.encoder:
        model = ClassicalModelWithEncoder(num_classes=common_hyperparams["num_classes"], encoder=encoder_module, encoder_output_size=64).to(device)
    else:
        model = ClassicalModel(num_classes=common_hyperparams["num_classes"]).to(device)

    if args.load_best:
        chosen_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
        model_path = os.path.join(chosen_dir, "best_classical_model.pth")
        if os.path.isfile(model_path):
            state = torch.load(model_path, map_location=device)
            try:
                model.load_state_dict(state)
            except Exception:
                if isinstance(state, dict) and "state_dict" in state:
                    model.load_state_dict(state["state_dict"]) 
                else:
                    model.load_state_dict(state)
            model.eval()
            print(f"Loaded best classical model from {model_path}")
        else:
            print(f"No best model found at {model_path}; continuing with initialized model")

    if args.extract:
        print("\nStarting Classical feature extraction...")
        extract_and_save_features(
            model, 
            train_loader, 
            device, 
            dirs["classical_features_dir"], 
            "train",
            extract_features=True 
        )
        extract_and_save_features(
            model, 
            val_loader, 
            device, 
            dirs["classical_features_dir"], 
            "val",
            extract_features=True
        )
        extract_and_save_features(
            model, 
            test_loader, 
            device, 
            dirs["classical_features_dir"], 
            "test",
            extract_features=True
        )
        return
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=common_hyperparams["learning_rate"])

    save_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
    
    # Run training loop
    history = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=common_hyperparams["epochs"],
        save_dir=save_dir,
        model_name="classical_model",
        extract_features=False, 
        save_best_only=True
    )
    
    print("Loading best model for final testing...")
    model_path = os.path.join(save_dir, "best_classical_model.pth")
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best model from {model_path}")
    else:
        print("Warning: No best model found, using final epoch weights")

    hyperparams_for_logging = {**common_hyperparams, "encoder_loaded": encoder_module is not None}

    test_results = test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        hyperparams=hyperparams_for_logging,
        output_file="classical_test_results.txt",
        extract_features=False,
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

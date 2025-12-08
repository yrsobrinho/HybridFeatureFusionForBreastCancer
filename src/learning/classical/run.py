import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader, BCDRLoader
from src.utils.training import (
    training_loop,
    test_model,
    extract_and_save_features
)
from src.models.main.classical import (
    ClassicalModel, 
    ClassicalModelWithEncoder,
    ShallowCNN,
    ShallowCNNWithEncoder
)
from src.models.preprocessing.autoencoder import Autoencoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract', action='store_true', help='Extract features and save to experiments dirs')
    parser.add_argument('--encoder', action='store_true', help='Use encoder-based model variant')
    parser.add_argument('--shallow', action='store_true', help='Use Shallow CNN (3 layers) instead of ResNet18')
    parser.add_argument('--load-best', action='store_true', help='Load best model weights before extracting')
    parser.add_argument('--dataset', type=str, default='breastmnist', choices=['breastmnist', 'bcdr'],
                        help='Dataset to use: breastmnist (default) or bcdr')
    parser.add_argument('--bcdr_root', type=str, default='src/data',
                        help='Root directory for BCDR dataset (used with --dataset bcdr)')
    parser.add_argument('--save-all-epochs', action='store_true',
                        help='Save checkpoint at every epoch')
    parser.add_argument('--save-frequency', type=int, default=1,
                        help='Save checkpoint every N epochs (with --save-all-epochs)')
    args = parser.parse_args()

    device = setup_device()
    dirs = setup_directories()

    # Determine model type
    model_type = "shallow" if args.shallow else "resnet"
    
    # Create model directories based on dataset and encoder usage
    # /experiments/models/{BreastMNIST|BCDR}/classical/encoder or /experiments/models/{BreastMNIST|BCDR}/classical
    base_models_dir = os.path.join(dirs.get("project_dir", "."), "experiments", "models")
    dataset_name = "BreastMNIST" if args.dataset == "breastmnist" else "BCDR"
    dataset_model_dir = os.path.join(base_models_dir, dataset_name, "classical")
    
    if args.encoder:
        encoder_model_dir = os.path.join(dataset_model_dir, "encoder", model_type)
        no_encoder_model_dir = None  # Not used when encoder=True
    else:
        encoder_model_dir = None  # Not used when encoder=False
        no_encoder_model_dir = os.path.join(dataset_model_dir, model_type)
    
    # Create the appropriate directory
    save_dir = encoder_model_dir if args.encoder else no_encoder_model_dir
    os.makedirs(save_dir, exist_ok=True)

    encoder_module = None
    use_rgb = False
    
    if args.encoder:
        if args.shallow:
            print(f"[Loading Autoencoder from {args.encoder_path}]")
            encoder_output_size = 28 
            print(f"Dataloader mode: {'RGB (3 channels)' if use_rgb else 'Grayscale (1 channel)'}")
        else:
            autoencoder_path = f"/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/models/autoencoder/autoencoder_{loader.dataset_name}.pth"
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
            encoder_output_size = 64  # ResNet expects 64
            print(f"Autoencoder loaded from {autoencoder_path}")

    common_hyperparams = {
        "epochs": 20 if args.shallow else 1,  # Shallow trains faster, can do more epochs
        "batch_size": 16 if args.shallow else 8,  # Shallow can handle larger batches
        "num_classes": 2,
        "learning_rate": 1e-3,
        "model_type": f"{model_type}_{'encoder' if args.encoder else 'no_encoder'}",
        "dataset": args.dataset
    }

    # Create dataloader based on dataset choice
    print(f"\n[Loading Dataset: {args.dataset.upper()}]")
    if args.dataset == 'bcdr':
        loader = BCDRLoader(
            batch_size=common_hyperparams["batch_size"], 
            image_size=224,
            use_rgb=use_rgb,
            root_dir=args.bcdr_root
        )
        print(f"BCDR dataset from: {args.bcdr_root}")
    else:
        loader = BreastMNISTLoader(
            batch_size=common_hyperparams["batch_size"], 
            image_size=224,
            use_rgb=use_rgb
        )
        print(f"BreastMNIST dataset")
    
    print(f"Dataloader mode: {'RGB (3 channels)' if use_rgb else 'Grayscale (1 channel)'}")
    train_loader, test_loader, val_loader = loader.get_dataloaders()

    # Create model based on flags
    print(f"\n[Creating Model: {model_type.upper()} {'with' if args.encoder else 'without'} encoder]")
    
    if args.shallow:
        if args.encoder:
            model = ShallowCNNWithEncoder(
                num_classes=common_hyperparams["num_classes"],
                encoder=encoder_module,
                encoder_output_size=encoder_output_size
            ).to(device)
        else:
            model = ShallowCNN(
                num_classes=common_hyperparams["num_classes"],
                input_channels=1
            ).to(device)
    else:
        if args.encoder:
            model = ClassicalModelWithEncoder(
                num_classes=common_hyperparams["num_classes"],
                encoder=encoder_module,
                encoder_output_size=encoder_output_size
            ).to(device)
        else:
            model = ClassicalModel(
                num_classes=common_hyperparams["num_classes"]
            ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.load_best:
        model_filename = f"best_{'shallow' if args.shallow else 'classical'}_model.pth"
        model_path = os.path.join(save_dir, model_filename)
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

    model_name = f"{'shallow' if args.shallow else 'classical'}_model"
    
    # Run training loop
    print(f"\n[Training {common_hyperparams['epochs']} epochs]")
    history = training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=common_hyperparams["epochs"],
        save_dir=save_dir,
        model_name=model_name,
        extract_features=False, 
        save_best_only=not args.save_all_epochs,
        save_all_epochs=args.save_all_epochs,
        save_frequency=args.save_frequency
    )
    
    print("\n[Loading best model for final testing...]")
    model_path = os.path.join(save_dir, f"best_{model_name}.pth")
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded best model from {model_path}")
    else:
        print("Warning: No best model found, using final epoch weights")

    hyperparams_for_logging = {
        **common_hyperparams, 
        "encoder_loaded": encoder_module is not None,
        "shallow": args.shallow
    }

    output_file = f"{'shallow' if args.shallow else 'classical'}_test_results.txt"
    
    test_results = test_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        hyperparams=hyperparams_for_logging,
        output_file=output_file,
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

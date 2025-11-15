import sys
sys.path.insert(0, '/home/eflammere/HybridEmbeddingsForBreastCancerDetection')

import os
import numpy as np
import torch
from tqdm import tqdm
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from models.main.quantum import QuantumModel
from models.main.classical import ClassicalModel
from models.main.extraction import QuantumFeatureExtractor, ClassicalFeatureExtractor


def load_models(device, directories, model_params):
    """Create models, load weights to device and set eval mode."""
    quantum_model = QuantumModel(model_params["n_qubits"], model_params["n_layers"], model_params["num_classes"]).to(device)
    classical_model = ClassicalModel(num_classes=model_params["num_classes"]).to(device)

    q_path = os.path.join(directories["quantum_model_dir"], "best_quantum_model.pth")
    c_path = os.path.join(directories["classical_model_dir"], "best_classical_model.pth")

    # Load with map_location so CPU users are supported
    quantum_state = torch.load(q_path, map_location=device)
    classical_state = torch.load(c_path, map_location=device)

    quantum_model.load_state_dict(quantum_state)
    classical_model.load_state_dict(classical_state)

    quantum_model.eval()
    classical_model.eval()

    return quantum_model, classical_model


def _extract_and_save(extractor, loader, device, out_dir, split_name):
    """Run extractor over loader, collect features and labels, and save to out_dir.

    Returns the shape of the saved features array.
    """
    os.makedirs(out_dir, exist_ok=True)

    features_list = []
    labels_list = []

    extractor.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split_name}"):
            # support dataloaders that return (imgs, labels) or (imgs, labels, ...)
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                imgs, labels = batch[0], batch[1]
            else:
                # if only images provided
                imgs = batch
                labels = None

            imgs = imgs.to(device)
            feats = extractor(imgs)

            feats = feats.cpu().numpy()
            features_list.append(feats)

            if labels is not None:
                # ensure labels are numpy array
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                labels_list.append(labels)

    if len(features_list) == 0:
        raise RuntimeError(f"No features extracted for split {split_name}")

    features = np.concatenate(features_list, axis=0)
    np.save(os.path.join(out_dir, f"{split_name}_features.npy"), features)

    if len(labels_list) > 0:
        labels = np.concatenate(labels_list, axis=0)
        np.save(os.path.join(out_dir, f"{split_name}_labels.npy"), labels)

    return features.shape


def main():
    device = setup_device()
    directories = setup_directories()

    # Setup data
    loader = BreastMNISTLoader(batch_size=8, image_size=224)
    train_loader, test_loader, val_loader = loader.get_dataloaders()
    dataloader = {"train": train_loader, "val": val_loader, "test": test_loader}

    # Load models
    model_params = {"n_qubits": 4, "n_layers": 1, "num_classes": 2}
    quantum_model, classical_model = load_models(device, directories, model_params)

    print("\nStarting Feature Extraction")
    extractor_classical = ClassicalFeatureExtractor(classical_model).to(device)
    extractor_quantum = QuantumFeatureExtractor(quantum_model).to(device)

    dataloaders = [("train", train_loader), ("test", test_loader), ("val", val_loader)]

    for split_name, loader in dataloaders:
        print(f"\nProcessing {split_name} split...")
        try:
            n_batches = len(loader)
        except Exception:
            n_batches = "unknown"
        print(f"Number of batches: {n_batches}")

        print(f"\nExtracting Classical Features for {split_name}...")
        c_shape = _extract_and_save(extractor_classical, loader, device, directories["classical_features_dir"], split_name)
        print(f"Classical features shape: {c_shape}")

        print(f"\nExtracting Quantum Features for {split_name}...")
        q_shape = _extract_and_save(extractor_quantum, loader, device, directories["quantum_features_dir"], split_name)
        print(f"Quantum features shape: {q_shape}")

        print(f"\nFeatures saved for {split_name} split:")
        print(f"- Classical: {directories['classical_features_dir']}/{split_name}_features.npy")
        print(f"- Classical labels: {directories['classical_features_dir']}/{split_name}_labels.npy")
        print(f"- Quantum: {directories['quantum_features_dir']}/{split_name}_features.npy")
        print(f"- Quantum labels: {directories['quantum_features_dir']}/{split_name}_labels.npy")

    print("\n=== Feature Extraction Completed ===")


if __name__ == "__main__":
    main()


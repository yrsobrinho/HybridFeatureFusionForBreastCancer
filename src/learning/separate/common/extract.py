import sys
sys.path.insert(0, '/home/eflammere/HybridEmbeddingsForBreastCancerDetection')

import torch
from tqdm import tqdm
from src.utils.common import setup_device, setup_directories
from src.utils.dataloader import BreastMNISTLoader
from src.models.quantum import QuantumModel
from src.models.classical import ClassicalModel
from src.utils.extraction import QuantumFeatureExtractor, ClassicalFeatureExtractor

def load_models(device, directories, model_params):
    quantum_model = QuantumModel(model_params["n_qubits"], model_params["n_layers"], model_params["num_classes"]).to(device)
    classical_model = ClassicalModel(num_classes=model_params["num_classes"]).to(device)

    quantum_model.load_state_dict(torch.load(f"{directories['quantum_model_dir']}/best_quantum_model.pth"))
    classical_model.load_state_dict(torch.load(f"{directories['classical_model_dir']}/best_classical_model.pth"))

    return quantum_model, classical_model

def main():
    device = setup_device()
    directories = setup_directories()

    # Setup data
    loader = BreastMNISTLoader(batch_size=32, image_size=100)
    train_loader, test_loader, val_loader = loader.get_dataloaders(
        {
            "train": "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/data/sets/train/images",
            "test": "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/data/sets/test/images",
            "val": "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/data/sets/val/images"
        },
        "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/data/csv/labels.csv"
    )

    # Load models
    model_params = {
        "n_qubits": 4,
        "n_layers": 1,
        "num_classes": 2
    }
    quantum_model, classical_model = load_models(device, directories, model_params)

    print("\n=== Starting Feature Extraction ===")
    extractor_classical = ClassicalFeatureExtractor(classical_model).to(device)
    extractor_quantum = QuantumFeatureExtractor(quantum_model).to(device)

    dataloaders = [
        ("train", train_loader),
        ("test", test_loader),
        ("val", val_loader)
    ]

    for split_name, loader in dataloaders:
        print(f"\nProcessing {split_name} split...")
        print(f"Number of batches: {len(loader)}")
        
        print(f"\nExtracting Classical Features for {split_name}...")
        features_classical = extractor_classical.extract_features(loader, device, directories["classical_features_dir"], split_name)
        print(f"Classical features shape: {features_classical.shape}")
        
        print(f"\nExtracting Quantum Features for {split_name}...")
        features_quantum = extractor_quantum.extract_features(loader, device, directories["quantum_features_dir"], split_name)
        print(f"Quantum features shape: {features_quantum.shape}")
        
        print(f"\nFeatures saved for {split_name} split:")
        print(f"- Classical: {directories['classical_features_dir']}/{split_name}_features.npy")
        print(f"- Quantum: {directories['quantum_features_dir']}/{split_name}_features.npy")

    print("\n=== Feature Extraction Completed ===")

if __name__ == "__main__":
    main()


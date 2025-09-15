import torch
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.")
    return device

def setup_directories():
    base_path = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/separate"
    directories = {
        "quantum_model_dir": f"{base_path}/best_models/quantum",
        "classical_model_dir": f"{base_path}/best_models/classical",
        "quantum_features_dir": f"{base_path}/features/quantum",
        "classical_features_dir": f"{base_path}/features/classical",
        "fused_features_dir": f"{base_path}/features/fused",
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    return directories

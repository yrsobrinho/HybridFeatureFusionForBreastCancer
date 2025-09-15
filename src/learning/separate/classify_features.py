import sys
sys.path.insert(0, '/home/eflammere/HybridEmbeddingsForBreastCancerDetection')

import torch
import numpy as np
from torch.utils.data import DataLoader
from src.utils.common import setup_device, setup_directories
from src.utils.methods import AttentionFusion, SelfAttentionBlock
from src.utils.fusion_methods import FusionMethods
from src.utils.classification_model import ClassificationModelHandler

def load_features(directories):
    quantum_features = np.load(f"{directories['quantum_features_dir']}/BreastMNIST_features.npy")
    classical_features = np.load(f"{directories['classical_features_dir']}/BreastMNIST_features.npy")
    labels = np.load(f"{directories['classical_features_dir']}/BreastMNIST_labels.npy")
    return quantum_features, classical_features, labels

def main():
    device = setup_device()
    directories = setup_directories()

    # Load features
    quantum_features, classical_features, labels = load_features(directories)

    # Setup fusion methods
    fusion_methods_instance = FusionMethods(device, attention_dim=128)
    fusion_methods = {
        'concat': fusion_methods_instance.fuse_concat,
        'add': fusion_methods_instance.fuse_add,
        'dot': fusion_methods_instance.fuse_dot,
        'attention': fusion_methods_instance.fuse_attention,
        'self_attention': fusion_methods_instance.fuse_self_attention
    }

    # Perform fusion and classification for each method
    for method_name, fusion_function in fusion_methods.items():
        fused_features = fusion_function(quantum_features, classical_features)
        np.save(f"{directories['fused_features_dir']}/fused_features_{method_name}.npy", fused_features)

        # Setup classification
        classification_handler = ClassificationModelHandler(device, num_classes=2, hyperparams={
            'input_dim': fused_features.shape[1],
            'epochs': 10,
            'learning_rate': 1e-3,
            'batch_size': 128
        })

        classification_handler.train_and_evaluate(
            fused_features, 
            labels,
            f"{directories['fused_features_dir']}/best_classification_model_{method_name}.pth",
            f"/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/separate/results/fusion_metrics_{method_name}.txt"
        )

if __name__ == "__main__":
    main()

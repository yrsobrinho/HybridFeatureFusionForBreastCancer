import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap

# Add project root to path
project_root = Path.cwd().parent
sys.path.insert(0, str(project_root))

from src.models.main.classical import ShallowCNN, ClassicalModel
from src.models.main.models import QuantumModel, ExtentedQuantumModel
from src.models.fusion import OnlineFusionModel
from src.utils.dataloader import BreastMNISTLoader, INbreastLoader, BUSUCLMLoader

EXPERIMENT_CONFIG = {
    "dataset": "BreastMNIST",              # BreastMNIST, INbreast, BUSUCLM
    "classical_model_type": "shallow",      # shallow, resnet
    "quantum_model_type": "quantum",  # quantum, extended_quantum
    "batch_size": 32,
    "image_size": 28,
    "num_classes": 2,
    "feature_dim": 128,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
    "output_dir": "../../experiments/models/fusion/online/BreastMNIST/shallow_extended_quantum/umap",
}

# Checkpoints (opcionais)
FUSION_CHECKPOINT_PATH = os.path.join(
    EXPERIMENT_CONFIG["output_dir"],
    f"best_fusion_{EXPERIMENT_CONFIG['classical_model_type']}_{EXPERIMENT_CONFIG['quantum_model_type']}.pth"
)
CLASSICAL_PRETRAINED_PATH = "../../experiments/models/classical/BreastMNIST/shallow/best_shallow.pth"
QUANTUM_PRETRAINED_PATH = "../../experiments/models/quantum/trainable/BreastMNIST/checkpoints/model_epoch15.pth"

DATASET = EXPERIMENT_CONFIG["dataset"]
CLASSICAL_MODEL_TYPE = EXPERIMENT_CONFIG["classical_model_type"]
QUANTUM_MODEL_TYPE = EXPERIMENT_CONFIG["quantum_model_type"]
BATCH_SIZE = EXPERIMENT_CONFIG["batch_size"]
IMAGE_SIZE = EXPERIMENT_CONFIG["image_size"]
NUM_CLASSES = EXPERIMENT_CONFIG["num_classes"]
FEATURE_DIM = EXPERIMENT_CONFIG["feature_dim"]
N_QUBITS = EXPERIMENT_CONFIG["n_qubits"]
N_LAYERS = EXPERIMENT_CONFIG["n_layers"]
PATCH_SIZE = EXPERIMENT_CONFIG["patch_size"]
OUTPUT_DIR = EXPERIMENT_CONFIG["output_dir"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Configuracao ativa:")
for k, v in EXPERIMENT_CONFIG.items():
    print(f"  - {k}: {v}")
print("Fusion checkpoint:", FUSION_CHECKPOINT_PATH)
print("Output dir:", OUTPUT_DIR)
print("Device:", device)
MAX_SAMPLES = None 

def _flatten(feats):
    if feats.ndim > 2:
        return feats.flatten(start_dim=1)
    return feats

if DATASET == "BreastMNIST":
    loader = BreastMNISTLoader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
elif DATASET == "INbreast":
    loader = INbreastLoader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
elif DATASET == "BUSUCLM":
    loader = BUSUCLMLoader(batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
else:
    raise ValueError(f"Dataset nao suportado: {DATASET}")

train_loader, test_loader, val_loader = loader.get_dataloaders()
print(f"✓ Dataset {DATASET} carregado (test batches: {len(test_loader)})")

if CLASSICAL_MODEL_TYPE == "shallow":
    classical_model = ShallowCNN(num_classes=NUM_CLASSES, input_channels=1)
elif CLASSICAL_MODEL_TYPE == "resnet":
    classical_model = ClassicalModel(num_classes=NUM_CLASSES)
else:
    raise ValueError(f"Tipo de modelo classico nao suportado: {CLASSICAL_MODEL_TYPE}")

if QUANTUM_MODEL_TYPE == "quantum":
    rand_params = np.random.uniform(high=2*np.pi, size=(N_LAYERS, N_QUBITS))
    quantum_model = QuantumModel(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        rand_params=rand_params,
        output_size=(14, 14),
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES
    )
elif QUANTUM_MODEL_TYPE == "extended_quantum":
    quantum_model = ExtentedQuantumModel(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        num_classes=NUM_CLASSES,
        patch_size=PATCH_SIZE,
        mode="2d",
        stride=PATCH_SIZE,
        use_parallel=True
    )
else:
    raise ValueError(f"Tipo de modelo quantico nao suportado: {QUANTUM_MODEL_TYPE}")

fusion_model = OnlineFusionModel(
    classical_model=classical_model,
    quantum_model=quantum_model,
    feature_dim=FEATURE_DIM,
    num_classes=NUM_CLASSES,
    freeze_classical=False,
    freeze_quantum=False
).to(device)

fusion_model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    _ = fusion_model(dummy_input)
fusion_model.train()

if os.path.exists(FUSION_CHECKPOINT_PATH):
    fusion_model.load_state_dict(torch.load(FUSION_CHECKPOINT_PATH, map_location=device))
    print(f"✓ Fusion checkpoint carregado: {FUSION_CHECKPOINT_PATH}")
elif CLASSICAL_PRETRAINED_PATH or QUANTUM_PRETRAINED_PATH:
    fusion_model.load_pretrained_weights(
        classical_path=CLASSICAL_PRETRAINED_PATH,
        quantum_path=QUANTUM_PRETRAINED_PATH,
        device=device
    )
    print("✓ Pesos base carregados (classico/quantico)")
else:
    print("⚠ Nenhum checkpoint encontrado. Usando pesos aleatorios.")

fusion_model.eval()

classical_l2_list = []
quantum_l2_list = []
fusion_l2_list = []
sample_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        classical_raw = fusion_model.classical_model.forward_relu(images)
        quantum_raw = fusion_model.quantum_model.forward_relu(images)
        fusion_raw = fusion_model.forward_relu(images)

        classical_l2_list.append(_flatten(classical_raw).cpu().numpy())
        quantum_l2_list.append(_flatten(quantum_raw).cpu().numpy())
        fusion_l2_list.append(_flatten(fusion_raw).cpu().numpy())
        sample_labels.append(labels.numpy())

classical_l2 = np.concatenate(classical_l2_list, axis=0)
quantum_l2 = np.concatenate(quantum_l2_list, axis=0)
fusion_l2 = np.concatenate(fusion_l2_list, axis=0)
sample_labels = np.concatenate(sample_labels, axis=0).squeeze()

if MAX_SAMPLES is not None and sample_labels.shape[0] > MAX_SAMPLES:
    idx = np.random.default_rng(42).choice(sample_labels.shape[0], MAX_SAMPLES, replace=False)
    classical_l2 = classical_l2[idx]
    quantum_l2 = quantum_l2[idx]
    fusion_l2 = fusion_l2[idx]
    sample_labels = sample_labels[idx]
    print("Subsampled to:", sample_labels.shape[0])

print("Classical:", classical_l2.shape)
print("Quantum:", quantum_l2.shape)
print("Fusion:", fusion_l2.shape)
print("Labels:", sample_labels.shape)

def print_feature_stats(name, features):
    print(f"\n{name}")
    print(f"  media: {features.mean():.6f}")
    print(f"  min:   {features.min():.6f}")
    print(f"  max:   {features.max():.6f}")

feature_sets = {
    "Classical L2": classical_l2,
    "Quantum L2": quantum_l2,
    "Fusion L2": fusion_l2,
}

normalized_feature_sets = {}
for name, feats in feature_sets.items():
    print_feature_stats(f"[ANTES] {name}", feats)
    normalized_feature_sets[name] = feats
    print_feature_stats(f"[DEPOIS] {name}", feats)

def plot_umap(features, labels, title, filename, output_dir):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=50
    )
    plt.colorbar(scatter, label="Class")
    plt.xlabel("UMAP Component 1")
    plt.ylabel("UMAP Component 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print("Saved:", out_path)

def plot_umap_3d(features, labels, title, filename, output_dir):
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        c=labels,
        cmap="coolwarm",
        alpha=0.6,
        s=30
    )
    fig.colorbar(scatter, ax=ax, label="Class")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")
    ax.set_zlabel("UMAP Component 3")
    ax.set_title(title)
    plt.tight_layout()

    out_path = os.path.join(output_dir, filename)
    plt.show()
    print("Saved:", out_path)

plot_umap(normalized_feature_sets["Classical L2"], sample_labels, "UMAP - Classical", "umap_classical.png", OUTPUT_DIR)
plot_umap(normalized_feature_sets["Quantum L2"], sample_labels, "UMAP - Quantum", "umap_quantum.png", OUTPUT_DIR)
plot_umap(normalized_feature_sets["Fusion L2"], sample_labels, "UMAP - Fusion", "umap_fusion.png", OUTPUT_DIR)

plot_umap_3d(normalized_feature_sets["Classical L2"], sample_labels, "UMAP 3D - Classical", "umap_3d_classical.png", OUTPUT_DIR)
plot_umap_3d(normalized_feature_sets["Quantum L2"], sample_labels, "UMAP 3D - Quantum", "umap_3d_quantum.png", OUTPUT_DIR)
plot_umap_3d(normalized_feature_sets["Fusion L2"], sample_labels, "UMAP 3D - Fusion", "umap_3d_fusion.png", OUTPUT_DIR)
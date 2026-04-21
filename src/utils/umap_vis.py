import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import re

# Use Computer Modern through mathtext so it works without a LaTeX install.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion import OnlineFusionModel
from src.models.main.classical import ClassicalModel, ShallowCNN
from src.models.main.models import ExtentedQuantumModel, QuantumModel
from src.models.classification import ClassificationModelHandler
from src.utils.dataloader import BUSUCLMLoader, INbreastLoader, BreastMNISTLoader


BASE_CONFIG = {
    "batch_size": 32,
    "num_classes": 2,
    "feature_dim": 128,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
    "max_samples": None,
    "output_root": PROJECT_ROOT / "graphs" / "umap",
}

QUANTUM_BASIC_RAND_PARAMS_PATH = PROJECT_ROOT / "notebooks" / "extra" / "best_model_rand_params.npy"

DATASETS = {
    "BreastMNIST": {"loader": BreastMNISTLoader, "image_size": 28},
    "INbreast": {"loader": INbreastLoader, "image_size": 224},
    "BUSUCLM": {"loader": BUSUCLMLoader, "image_size": 224},
}

CLASSICAL_VARIANTS = [
    {
        "slug": "classical_shallow",
        "display": "Classical - SCNN",
        "classical_type": "shallow",
        "checkpoint": "experiments/models/classical/{dataset}/shallow/best_shallow.pth",
    },
    {
        "slug": "classical_resnet",
        "display": "Classical - ResNet",
        "classical_type": "resnet",
        "checkpoint": "experiments/models/classical/{dataset}/resnet/best_resnet.pth",
    },
]

QUANTUM_VARIANTS = [
    {
        "slug": "quantum_basic",
        "display": "Quantum - Non-trainable",
        "quantum_type": "quantum",
        "checkpoint": "experiments/models/quantum/basic/{dataset}/best_model.pth",
    },
    {
        "slug": "quantum_trainable",
        "display": "Quantum - Trainable",
        "quantum_type": "extended_quantum",
        "checkpoint": "experiments/models/quantum/trainable/{dataset}/best_model.pth",
    },
]

FUSION_ONLINE_VARIANTS = [
    {
        "slug": "fusion_dhf_scnn_nontrainable",
        "display": "DHF (SCNN + Non-trainable)",
        "classical_type": "shallow",
        "quantum_type": "quantum",
        "with_temperature": False,
        "checkpoint": "experiments/models/fusion/online/{dataset}/shallow_quantum/best_fusion_shallow_quantum.pth",
    },
    {
        "slug": "fusion_dhf_scnn_trainable",
        "display": "DHF (SCNN + Trainable)",
        "classical_type": "shallow",
        "quantum_type": "extended_quantum",
        "with_temperature": False,
        "checkpoint": "experiments/models/fusion/online/{dataset}/shallow_extended_quantum/best_fusion_shallow_extended_quantum.pth",
    },
    {
        "slug": "fusion_dhf_resnet_nontrainable",
        "display": "DHF (ResNet + Non-trainable)",
        "classical_type": "resnet",
        "quantum_type": "quantum",
        "with_temperature": False,
        "checkpoint": "experiments/models/fusion/online/{dataset}/resnet_quantum/best_fusion_resnet_quantum.pth",
    },
    {
        "slug": "fusion_dhf_resnet_trainable",
        "display": "DHF (ResNet + Trainable)",
        "classical_type": "resnet",
        "quantum_type": "extended_quantum",
        "with_temperature": False,
        "checkpoint": "experiments/models/fusion/online/{dataset}/resnet_extended_quantum/best_fusion_resnet_extended_quantum.pth",
    },
]

FUSION_TEMPERATURE_VARIANTS = [
    {
        "slug": "fusion_tshf_scnn_nontrainable",
        "display": "TSHF (SCNN + Non-trainable)",
        "classical_type": "shallow",
        "quantum_type": "quantum",
        "with_temperature": True,
        "checkpoint": "experiments/models/fusion/online/{dataset}/shallow_quantum/temperature/best_fusion_shallow_quantum_temperature.pth",
    },
    {
        "slug": "fusion_tshf_scnn_trainable",
        "display": "TSHF (SCNN + Trainable)",
        "classical_type": "shallow",
        "quantum_type": "extended_quantum",
        "with_temperature": True,
        "checkpoint": "experiments/models/fusion/online/{dataset}/shallow_extended_quantum/temperature/best_fusion_shallow_extended_quantum_temperature.pth",
    },
    {
        "slug": "fusion_tshf_resnet_nontrainable",
        "display": "TSHF (ResNet + Non-trainable)",
        "classical_type": "resnet",
        "quantum_type": "quantum",
        "with_temperature": True,
        "checkpoint": "experiments/models/fusion/online/{dataset}/resnet_quantum/temperature/best_fusion_resnet_quantum_temperature.pth",
    },
    {
        "slug": "fusion_tshf_resnet_trainable",
        "display": "TSHF (ResNet + Trainable)",
        "classical_type": "resnet",
        "quantum_type": "extended_quantum",
        "with_temperature": True,
        "checkpoint": "experiments/models/fusion/online/{dataset}/resnet_extended_quantum/temperature/best_fusion_resnet_extended_quantum_temperature.pth",
    },
]

FUSION_OFFLINE_VARIANTS = [
    {
        "slug": "fusion_shf_scnn_nontrainable",
        "display": "SHF (SCNN + Non-trainable)",
        "features": "experiments/models/fusion/offline/{dataset}/concatenated/basic_shallow/test_features_concat.npy",
        "classifier": "experiments/models/classification/offline/{dataset}/basic_shallow/best_classification_{dataset_lower}_basic_shallow.pth",
    },
    {
        "slug": "fusion_shf_scnn_trainable",
        "display": "SHF (SCNN + Trainable)",
        "features": "experiments/models/fusion/offline/{dataset}/concatenated/trainable_shallow/test_features_concat.npy",
        "classifier": "experiments/models/classification/offline/{dataset}/trainable_shallow/best_classification_{dataset_lower}_trainable_shallow.pth",
    },
    {
        "slug": "fusion_shf_resnet_nontrainable",
        "display": "SHF (ResNet + Non-trainable)",
        "features": "experiments/models/fusion/offline/{dataset}/concatenated/basic_resnet/test_features_concat.npy",
        "classifier": "experiments/models/classification/offline/{dataset}/basic_resnet/best_classification_{dataset_lower}_basic_resnet.pth",
    },
    {
        "slug": "fusion_shf_resnet_trainable",
        "display": "SHF (ResNet + Trainable)",
        "features": "experiments/models/fusion/offline/{dataset}/concatenated/trainable_resnet/test_features_concat.npy",
        "classifier": "experiments/models/classification/offline/{dataset}/trainable_resnet/best_classification_{dataset_lower}_trainable_resnet.pth",
    },
]

CLASS_COLOR_MAP = {
    0: "#1D4ED8",  # Negative
    1: "#DC2626",  # Positive
}
CLASS_MARKER_MAP = {
    0: "o",
    1: "^",
}


class TemperatureFusionModel(OnlineFusionModel):
    """Temperature-based online fusion used by TSHF checkpoints."""

    def __init__(
        self,
        classical_model,
        quantum_model,
        feature_dim=128,
        num_classes=2,
        freeze_classical=False,
        freeze_quantum=False,
        init_gamma=1.0,
    ):
        super().__init__(
            classical_model=classical_model,
            quantum_model=quantum_model,
            feature_dim=feature_dim,
            num_classes=num_classes,
            freeze_classical=freeze_classical,
            freeze_quantum=freeze_quantum,
        )
        init_gamma = float(max(init_gamma, 1e-6))
        self.log_gamma = nn.Parameter(torch.tensor(np.log(init_gamma), dtype=torch.float32))

    @property
    def gamma(self):
        return torch.exp(self.log_gamma)

    def extract_features(self, x):
        if hasattr(self.classical_model, "extract_features"):
            classical_feats = self.classical_model.extract_features(x)
        else:
            classical_feats = self.classical_model(x)

        if hasattr(self.quantum_model, "extract_features"):
            quantum_feats = self.quantum_model.extract_features(x)
        else:
            quantum_feats = self.quantum_model(x)

        if classical_feats.ndim > 2:
            classical_feats = classical_feats.flatten(start_dim=1)
        if quantum_feats.ndim > 2:
            quantum_feats = quantum_feats.flatten(start_dim=1)

        if self.classical_projection is None or self.quantum_projection is None:
            self._init_projections(classical_feats.shape[1], quantum_feats.shape[1], x.device)

        classical_proj = self.classical_projection(classical_feats)
        quantum_proj = self.quantum_projection(quantum_feats)

        classical_norm = F.normalize(classical_proj, p=2, dim=1)
        quantum_norm = F.normalize(quantum_proj, p=2, dim=1)
        quantum_scaled = self.gamma * quantum_norm

        return torch.cat([classical_norm, quantum_scaled], dim=1)


def _flatten(feats):
    if feats.ndim > 2:
        return feats.flatten(start_dim=1)
    return feats


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
    return checkpoint_obj


def _load_weights(model, checkpoint_path, ignored_keys=None, strict=False):
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint_obj)

    if not isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
        return

    ignored_keys = set(ignored_keys or [])
    if ignored_keys:
        state_dict = {k: v for k, v in state_dict.items() if k not in ignored_keys}
        load_result = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in load_result.unexpected_keys if k not in ignored_keys]
        if load_result.missing_keys:
            raise RuntimeError(f"Missing keys after load: {load_result.missing_keys}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys after load: {unexpected}")
        return

    model.load_state_dict(state_dict, strict=strict)


def _build_loader(dataset_name):
    dataset_cfg = DATASETS[dataset_name]
    loader = dataset_cfg["loader"](
        batch_size=BASE_CONFIG["batch_size"],
        image_size=dataset_cfg["image_size"],
    )
    return loader.get_dataloaders()[1], dataset_cfg["image_size"]


def _create_classical_model(classical_type):
    if classical_type == "shallow":
        return ShallowCNN(num_classes=BASE_CONFIG["num_classes"], input_channels=1)
    if classical_type == "resnet":
        return ClassicalModel(num_classes=BASE_CONFIG["num_classes"])
    raise ValueError(f"Unsupported classical model type: {classical_type}")


def _create_quantum_model(quantum_type, image_size):
    if quantum_type == "quantum":
        if not QUANTUM_BASIC_RAND_PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"Saved rand params not found: {QUANTUM_BASIC_RAND_PARAMS_PATH}"
            )
        rand_params = np.load(QUANTUM_BASIC_RAND_PARAMS_PATH)
        output_size = (image_size // BASE_CONFIG["patch_size"], image_size // BASE_CONFIG["patch_size"])
        return QuantumModel(
            n_qubits=BASE_CONFIG["n_qubits"],
            n_layers=BASE_CONFIG["n_layers"],
            rand_params=rand_params,
            output_size=output_size,
            patch_size=BASE_CONFIG["patch_size"],
            num_classes=BASE_CONFIG["num_classes"],
        )

    if quantum_type == "extended_quantum":
        return ExtentedQuantumModel(
            n_qubits=BASE_CONFIG["n_qubits"],
            n_layers=BASE_CONFIG["n_layers"],
            num_classes=BASE_CONFIG["num_classes"],
            patch_size=BASE_CONFIG["patch_size"],
            mode="2d",
            stride=BASE_CONFIG["patch_size"],
            use_parallel=True,
        )

    raise ValueError(f"Unsupported quantum model type: {quantum_type}")


def _warmup_model(model, test_loader, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            break


def _extract_features_from_model(model, test_loader, device, mode):
    feature_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            if mode == "classical":
                feats = model.forward_relu(images)
            elif mode == "quantum":
                feats = model.forward_relu(images)
            elif mode == "fusion":
                feats = model.forward_relu(images)
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            feature_list.append(_flatten(feats).detach().cpu().numpy())
            label_list.append(labels.detach().cpu().numpy().reshape(-1))

    features = np.concatenate(feature_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    return features, labels


def _subsample(features, labels):
    max_samples = BASE_CONFIG["max_samples"]
    if max_samples is None or labels.shape[0] <= max_samples:
        return features, labels
    rng = np.random.default_rng(42)
    idx = rng.choice(labels.shape[0], max_samples, replace=False)
    return features[idx], labels[idx]


def _plot_umap(features, labels, title, output_path):
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    for class_label in unique_labels:
        class_mask = labels == class_label
        class_label_int = int(class_label)
        if class_label_int == 1:
            legend_label = "Positive (Class 1)"
        elif class_label_int == 0:
            legend_label = "Negative (Class 0)"
        else:
            legend_label = f"Class {class_label_int}"

        plt.scatter(
            embedding[class_mask, 0],
            embedding[class_mask, 1],
            c=CLASS_COLOR_MAP.get(class_label_int, "#374151"),
            marker=CLASS_MARKER_MAP.get(class_label_int, "o"),
            alpha=0.75,
            s=70,
            edgecolors="black",
            linewidths=0.5,
            label=legend_label,
        )

    # Remove labels/ticks de componente para um visual mais limpo.
    plt.xlabel("")
    plt.ylabel("")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="best", frameon=True, fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"    ✓ Saved: {output_path}")


def _resolve_path(path_template, dataset_name):
    return PROJECT_ROOT / path_template.format(dataset=dataset_name)


def _safe_filename(text):
    normalized = text.lower().strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "model"


def _process_classical_models(dataset_name, test_loader, device):
    for cfg in CLASSICAL_VARIANTS:
        ckpt_path = _resolve_path(cfg["checkpoint"], dataset_name)
        if not ckpt_path.exists():
            print(f"    ✗ Missing checkpoint: {ckpt_path}")
            continue

        model = _create_classical_model(cfg["classical_type"]).to(device)
        _load_weights(model, ckpt_path, strict=False)
        features, labels = _extract_features_from_model(model, test_loader, device, mode="classical")
        features, labels = _subsample(features, labels)

        file_name = f"umap_2d_{_safe_filename(cfg['display'])}.png"
        out_path = BASE_CONFIG["output_root"] / dataset_name / file_name
        _plot_umap(features, labels, f"UMAP - {dataset_name} - {cfg['display']}", out_path)


def _process_quantum_models(dataset_name, test_loader, image_size, device):
    for cfg in QUANTUM_VARIANTS:
        ckpt_path = _resolve_path(cfg["checkpoint"], dataset_name)
        if not ckpt_path.exists():
            print(f"    ✗ Missing checkpoint: {ckpt_path}")
            continue

        model = _create_quantum_model(cfg["quantum_type"], image_size).to(device)
        _warmup_model(model, test_loader, device)
        _load_weights(model, ckpt_path, strict=True)
        features, labels = _extract_features_from_model(model, test_loader, device, mode="quantum")
        features, labels = _subsample(features, labels)

        file_name = f"umap_2d_{_safe_filename(cfg['display'])}.png"
        out_path = BASE_CONFIG["output_root"] / dataset_name / file_name
        _plot_umap(features, labels, f"UMAP - {dataset_name} - {cfg['display']}", out_path)


def _build_fusion_model(classical_type, quantum_type, image_size, with_temperature=False):
    classical_model = _create_classical_model(classical_type)
    quantum_model = _create_quantum_model(quantum_type, image_size)
    fusion_cls = TemperatureFusionModel if with_temperature else OnlineFusionModel
    return fusion_cls(
        classical_model=classical_model,
        quantum_model=quantum_model,
        feature_dim=BASE_CONFIG["feature_dim"],
        num_classes=BASE_CONFIG["num_classes"],
        freeze_classical=False,
        freeze_quantum=False,
    )


def _process_online_fusion_models(dataset_name, test_loader, image_size, device, variants):
    for cfg in variants:
        ckpt_path = _resolve_path(cfg["checkpoint"], dataset_name)
        if not ckpt_path.exists():
            print(f"    ✗ Missing checkpoint: {ckpt_path}")
            continue

        model = _build_fusion_model(
            cfg["classical_type"],
            cfg["quantum_type"],
            image_size,
            with_temperature=cfg.get("with_temperature", False),
        ).to(device)
        _warmup_model(model, test_loader, device)
        _load_weights(model, ckpt_path, strict=True)

        features, labels = _extract_features_from_model(model, test_loader, device, mode="fusion")
        features, labels = _subsample(features, labels)

        file_name = f"umap_2d_{_safe_filename(cfg['display'])}.png"
        out_path = BASE_CONFIG["output_root"] / dataset_name / file_name
        _plot_umap(features, labels, f"UMAP - {dataset_name} - {cfg['display']}", out_path)


def _process_offline_fusion_models(dataset_name, device):
    labels_path = PROJECT_ROOT / f"experiments/labels/{dataset_name}/test_labels.npy"
    if not labels_path.exists():
        print(f"    ✗ Missing labels for offline fusion: {labels_path}")
        return

    labels = np.load(labels_path).reshape(-1)

    for cfg in FUSION_OFFLINE_VARIANTS:
        features_path = _resolve_path(cfg["features"], dataset_name)
        classifier_path = PROJECT_ROOT / cfg["classifier"].format(
            dataset=dataset_name,
            dataset_lower=dataset_name.lower(),
        )
        if not features_path.exists():
            print(f"    ✗ Missing offline features: {features_path}")
            continue
        if not classifier_path.exists():
            print(f"    ✗ Missing offline classifier: {classifier_path}")
            continue

        features = np.load(features_path)
        if features.shape[0] != labels.shape[0]:
            print(
                "    ✗ Shape mismatch for offline fusion "
                f"{cfg['slug']}: features={features.shape[0]}, labels={labels.shape[0]}"
            )
            continue

        classifier = ClassificationModelHandler(
            device=device,
            num_classes=BASE_CONFIG["num_classes"],
            sequential_hyperparams={"input_dim": features.shape[1], "learning_rate": 1e-3},
        )
        classifier.model.load_state_dict(torch.load(classifier_path, map_location=device), strict=True)
        classifier.model.eval()

        with torch.no_grad():
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            # Penultimate representation from offline classifier.
            offline_embeddings = classifier.model[:-1](feature_tensor).detach().cpu().numpy()

        offline_embeddings, local_labels = _subsample(offline_embeddings, labels)
        file_name = f"umap_2d_{_safe_filename(cfg['display'])}.png"
        out_path = BASE_CONFIG["output_root"] / dataset_name / file_name
        _plot_umap(offline_embeddings, local_labels, f"UMAP - {dataset_name} - {cfg['display']}", out_path)


def main():
    BASE_CONFIG["output_root"].mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for dataset_name in DATASETS:
        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        test_loader, image_size = _build_loader(dataset_name)

        print("  [1/5] Classical models")
        _process_classical_models(dataset_name, test_loader, device)

        print("  [2/5] Quantum models")
        _process_quantum_models(dataset_name, test_loader, image_size, device)

        print("  [3/5] Fusion DHF models")
        _process_online_fusion_models(dataset_name, test_loader, image_size, device, FUSION_ONLINE_VARIANTS)

        print("  [4/5] Fusion TSHF models")
        _process_online_fusion_models(dataset_name, test_loader, image_size, device, FUSION_TEMPERATURE_VARIANTS)

        print("  [5/5] Fusion SHF models")
        _process_offline_fusion_models(dataset_name, device)

    print("\nFinished batch UMAP generation for all datasets and model types.")


if __name__ == "__main__":
    main()
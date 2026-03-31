import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.classification import ClassificationModelHandler
from src.models.main.classical import ClassicalModel, ShallowCNN
from src.utils.setup import count_parameters


@dataclass
class ModelParamStats:
    name: str
    total_params: int
    trainable_params: int


class FallbackResNetModel(nn.Module):
    """ResNet18 without pretrained weights, used when pretrained download fails."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = tv_models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.features(x)
        return torch.flatten(feats, 1)


def total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def init_lazy_layers(model: nn.Module, dummy_input: torch.Tensor) -> None:
    """Runs one dry forward pass to initialize lazy layers such as late-created FCs."""
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)


def build_models(
    num_classes: int,
    image_size: int,
    n_qubits: int,
    n_layers: int,
    patch_size: int,
    device: torch.device,
) -> Dict[str, nn.Module]:
    shallow_model = ShallowCNN(num_classes=num_classes, input_channels=1).to(device)

    try:
        resnet_model = ClassicalModel(num_classes=num_classes).to(device)
    except Exception as exc:
        print(f"[warn] Could not initialize ClassicalModel with pretrained weights: {exc}")
        print("[warn] Falling back to ResNet18(weights=None) for parameter counting.")
        resnet_model = FallbackResNetModel(num_classes=num_classes).to(device)

    models = {
        "ShallowModel": shallow_model,
        "ResNetModel": resnet_model,
    }

    try:
        from src.models.main.models import ExtentedQuantumModel, QuantumModel

        rand_params = np.random.uniform(high=2 * math.pi, size=(n_qubits, n_layers))
        quantum_model = QuantumModel(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rand_params=rand_params,
            output_size=(image_size // patch_size, image_size // patch_size),
            patch_size=patch_size,
            num_classes=num_classes,
        ).to(device)

        extended_quantum_model = ExtentedQuantumModel(
            n_qubits=n_qubits,
            n_layers=n_layers,
            num_classes=num_classes,
            patch_size=patch_size,
            mode="2d",
            stride=patch_size,
            use_parallel=True,
        ).to(device)

        dummy = torch.randn(1, 1, image_size, image_size, device=device)
        init_lazy_layers(quantum_model, dummy)
        init_lazy_layers(extended_quantum_model, dummy)

        models["QuantumModel"] = quantum_model
        models["ExtendedQuantumModel"] = extended_quantum_model
    except Exception as exc:
        print(f"[warn] Quantum models could not be initialized: {exc}")
        print("[warn] Install compatible PennyLane/autograd/numpy versions to include quantum counts.")

    return models


def get_feature_dim(model: nn.Module, image_size: int, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 1, image_size, image_size, device=device)
        if hasattr(model, "extract_features"):
            feats = model.extract_features(x)
        else:
            feats = model(x)
    if feats.ndim > 2:
        feats = feats.flatten(start_dim=1)
    return int(feats.shape[1])


def build_classification_handlers(
    models: Dict[str, nn.Module],
    image_size: int,
    num_classes: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, ClassificationModelHandler]:
    handlers: Dict[str, ClassificationModelHandler] = {}

    for model_name, model in models.items():
        input_dim = get_feature_dim(model, image_size=image_size, device=device)
        hyperparams = {
            "input_dim": input_dim,
            "learning_rate": learning_rate,
        }
        handlers[f"ClassificationHandler[{model_name}]"] = ClassificationModelHandler(
            device=device,
            num_classes=num_classes,
            sequential_hyperparams=hyperparams,
        )

    return handlers


def collect_parameter_stats(models: Dict[str, nn.Module]) -> Tuple[ModelParamStats, ...]:
    stats = []
    for name, model in models.items():
        trainable = count_parameters(model)
        total = total_parameters(model)
        stats.append(ModelParamStats(name=name, total_params=total, trainable_params=trainable))
    return tuple(stats)


def print_report(title: str, stats: Tuple[ModelParamStats, ...]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"{'Model':40} {'Total Params':>15} {'Trainable Params':>20}")
    print("-" * 80)
    for item in stats:
        print(f"{item.name:40} {item.total_params:15,} {item.trainable_params:20,}")


def save_csv(csv_path: str, base_stats: Tuple[ModelParamStats, ...], handler_stats: Tuple[ModelParamStats, ...]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "model", "total_params", "trainable_params"])

        for item in base_stats:
            writer.writerow(["base_model", item.name, item.total_params, item.trainable_params])

        for item in handler_stats:
            writer.writerow(["classification_handler", item.name, item.total_params, item.trainable_params])

    print(f"\nCSV saved to: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate total/trainable parameters for base models and classification handlers."
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--handler-lr", type=float, default=1e-3)
    parser.add_argument(
        "--csv-out",
        type=str,
        default="",
        help="Optional output CSV path. Example: experiments/models/parameter_report.csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for model instantiation and dry-run initialization.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    models = build_models(
        num_classes=args.num_classes,
        image_size=args.image_size,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        patch_size=args.patch_size,
        device=device,
    )
    model_stats = collect_parameter_stats(models)
    print_report("Base Models", model_stats)

    handlers = build_classification_handlers(
        models=models,
        image_size=args.image_size,
        num_classes=args.num_classes,
        learning_rate=args.handler_lr,
        device=device,
    )
    handler_stats = collect_parameter_stats({name: h.model for name, h in handlers.items()})
    print_report("Classification Handlers", handler_stats)

    if args.csv_out:
        save_csv(args.csv_out, model_stats, handler_stats)


if __name__ == "__main__":
    main()

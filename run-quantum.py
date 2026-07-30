"""Quantum baselines — multi-run evaluation, all configs in one sequential pass.

Each entry in CONFIGS is a distinct model trained/evaluated independently, in the
order listed: the fixed (non-trainable) quanvolution first, then the trainable one.

    python run-quantum.py                       # BUS-UCLM, quantum then extended, 5 runs each
    python run-quantum.py --only quantum        # a single config
    python run-quantum.py --dataset breastmnist
    python run-quantum.py --test-only           # no training: re-tests best_*.pth on disk
    python run-quantum.py --runs 3 --epochs 10

Outputs go to reports/<Dataset>/quantum/<variant>/ (see reports/README.md).

Cost warning: the fixed quanvolution evaluates one PennyLane circuit per patch in
a Python loop on CPU (~3.7 s per batch of 16 at 28x28 / patch 2), so a GPU does
not help. Budget ~1 min per epoch on BUS-UCLM and scale --runs/--epochs to taste.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.main.models import ExtentedQuantumModel, QuantumModel
from src.utils.dataloader import (
    BCDRLoader,
    BUSUCLMLoader,
    BreastMNISTLoader,
    INbreastLoader,
)
from src.utils.multirun import evaluate_checkpoints, run_multi_seed
from src.utils.training import SELECTION_METRICS

FAMILY = "quantum"

# Folder name under reports/ for each --dataset value.
DATASETS = {
    "breastmnist": "BreastMNIST",
    "bcdr": "BCDR",
    "inbreast": "INbreast",
    "busuclm": "BUS-UCLM",
}

# Validation metric each run's best_*.pth is selected on, per dataset.
# BUS-UCLM's case-wise split is imbalanced in opposite directions (val 32/7 benign-heavy,
# test 7/19 malignant-heavy), so val accuracy just rewards the epoch that predicts benign
# most and the checkpoint it keeps tests badly — balanced accuracy weights both classes.
SELECTION_METRIC_DEFAULTS = {"busuclm": "f1"}
DEFAULT_SELECTION_METRIC = "accuracy"

DEFAULTS = {
    "dataset": "busuclm",
    "batch_size": 16,
    "image_size": 28,
    "num_classes": 2,
    "input_channels": 1,
    "num_epochs": 10,
    "learning_rate": 1e-3,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
}

CONFIGS = [
    #{
    #   # QuantumModel: fixed random circuit, only the FC head learns.
     #  "name": "quantum",
     #   "model_type": "basic",
        # Checkpoint stem kept as `quantum_run<N>` — src/utils/fusion_experiment.py
        # reloads these as the quantum branch of the fusion models.
     #    "run_tag": "quantum",
     #    "reports_subdir": "non-trainable",
    #},
    {
        # ExtentedQuantumModel: the quanvolution rotations are trained too.
        "name": "extended",
        "model_type": "trainable",
        "run_tag": "extended_quantum",
        "reports_subdir": "trainable",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULTS["dataset"], choices=sorted(DATASETS),
                   help=f"dataset to train on (default: {DEFAULTS['dataset']})")
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="run only these configs (by 'name'); default: all, in order")
    p.add_argument("--runs", type=int, default=5, help="seeds per config (default: 5)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override num_epochs for every config")
    p.add_argument("--batch-size", type=int, default=None, help="override batch size")
    p.add_argument("--image-size", type=int, default=None, help="override image size")
    p.add_argument("--lr", type=float, default=None, help="override learning rate")
    p.add_argument("--n-qubits", type=int, default=None, help="override qubit count")
    p.add_argument("--n-layers", type=int, default=None, help="override circuit depth")
    p.add_argument("--patch-size", type=int, default=None, help="override quanvolution patch size")
    p.add_argument("--selection-metric", default=None, choices=list(SELECTION_METRICS),
                   help="validation metric the best checkpoint is picked on "
                        f"(default: {DEFAULT_SELECTION_METRIC}, "
                        f"{SELECTION_METRIC_DEFAULTS['busuclm']} on busuclm)")
    p.add_argument("--test-only", action="store_true",
                   help="skip training; reload best_*.pth and regenerate metrics/figures")
    return p.parse_args()


def resolve(config, args):
    """Merge a partial config over DEFAULTS plus the CLI overrides."""
    cfg = {**DEFAULTS, **config, "dataset": args.dataset}
    for key, value in [("num_epochs", args.epochs), ("batch_size", args.batch_size),
                       ("image_size", args.image_size), ("learning_rate", args.lr),
                       ("n_qubits", args.n_qubits), ("n_layers", args.n_layers),
                       ("patch_size", args.patch_size)]:
        if value is not None:
            cfg[key] = value
    cfg["dataset_name"] = DATASETS[cfg["dataset"]]
    cfg["selection_metric"] = args.selection_metric or SELECTION_METRIC_DEFAULTS.get(
        cfg["dataset"], DEFAULT_SELECTION_METRIC)
    if cfg["model_type"] == "basic" and cfg["n_qubits"] != cfg["patch_size"] ** 2:
        raise SystemExit(
            f"[{cfg['name']}] 'basic' feeds one patch pixel per qubit, so n_qubits must equal "
            f"patch_size ** 2 (got n_qubits={cfg['n_qubits']}, patch_size={cfg['patch_size']} -> "
            f"{cfg['patch_size'] ** 2} pixels per patch).")
    return cfg


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg):
    """Return (train_loader, test_loader, val_loader) for cfg['dataset']."""
    kwargs = {"batch_size": cfg["batch_size"], "image_size": cfg["image_size"]}
    if cfg["dataset"] == "bcdr":
        loader = BCDRLoader(root_dir=str(PROJECT_ROOT / "src/data/BCDR/png"), **kwargs)
    elif cfg["dataset"] == "inbreast":
        loader = INbreastLoader(root_dir=str(PROJECT_ROOT / "src/data/INbreast/png"), **kwargs)
    elif cfg["dataset"] == "busuclm":
        # png/ is the case-wise split written by src/data/BUS-UCLM/utils/preprocessing.py,
        # so train/val/test never share a patient.
        loader = BUSUCLMLoader(root_dir=str(PROJECT_ROOT / "src/data/BUS-UCLM/png"), **kwargs)
    else:
        loader = BreastMNISTLoader(**kwargs)
    return loader.get_dataloaders()


def probe_input_shape(train_loader, cfg):
    """Actual (H, W) coming out of the loader — the transforms, not cfg, decide it."""
    images, _ = next(iter(train_loader))
    return images.shape[-2], images.shape[-1]


def build_model(cfg, device):
    """Build the model and materialize its lazy FC head.

    Both variants create `fc` on the first forward pass, so it has to exist before
    the optimizer is built — otherwise Adam never sees the classifier's parameters.
    """
    input_h, input_w = cfg["input_h"], cfg["input_w"]

    if cfg["model_type"] == "trainable":
        model = ExtentedQuantumModel(
            n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"],
            num_classes=cfg["num_classes"], patch_size=cfg["patch_size"],
            mode="2d", stride=None, use_parallel=True,
        ).to(device)
    elif cfg["model_type"] == "basic":
        # Drawn inside build_fn, i.e. after run_multi_seed seeds the RNGs, so each
        # run gets its own reproducible fixed circuit.
        rand_params = np.random.uniform(high=2 * np.pi, size=(cfg["n_layers"], cfg["n_qubits"]))
        model = QuantumModel(
            n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"], rand_params=rand_params,
            output_size=cfg["output_size"], patch_size=cfg["patch_size"],
            num_classes=cfg["num_classes"],
        ).to(device)
    else:
        raise ValueError(f"Modelo quantico nao suportado: {cfg['model_type']}")

    with torch.no_grad():
        _ = model(torch.randn(1, cfg["input_channels"], input_h, input_w).to(device))
    return model


def make_build_fn(cfg, device):
    """Return `build_fn(seed) -> (model, optimizer, model_name)` for src.utils.multirun."""

    def build_fn(seed):
        model = build_model(cfg, device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        return model, optimizer, f"{cfg['run_tag']}_run{seed + 1}"

    return build_fn


def reports_dir(cfg):
    """`reports/<Dataset>/quantum/<variant>` under the project root."""
    return os.path.join(str(PROJECT_ROOT), "reports", cfg["dataset_name"],
                        FAMILY, cfg["reports_subdir"])


def hyperparams(cfg):
    keys = ["dataset", "model_type", "batch_size", "image_size", "learning_rate",
            "num_epochs", "num_classes", "n_qubits", "n_layers", "patch_size",
            "selection_metric"]
    return {k: cfg[k] for k in keys}


def report_meta(cfg):
    return {
        "Dataset": cfg["dataset_name"],
        "Model Variant": cfg["model_type"],
        "Model": "ExtentedQuantumModel" if cfg["model_type"] == "trainable" else "QuantumModel",
        "Epochs per run": cfg["num_epochs"],
        "Batch Size": cfg["batch_size"],
        "Image Size": cfg["image_size"],
        "Learning Rate": cfg["learning_rate"],
        "n_qubits": cfg["n_qubits"],
        "n_layers": cfg["n_layers"],
        "patch_size": cfg["patch_size"],
        "Quantum output size": cfg.get("output_size"),
        "Checkpoint selection": f"val_{cfg['selection_metric']}",
    }


def describe(cfg):
    return (f"{cfg['model_type']} | {cfg['dataset_name']} | epochs={cfg['num_epochs']} | "
            f"bs={cfg['batch_size']} | img={cfg['image_size']} | lr={cfg['learning_rate']} | "
            f"qubits={cfg['n_qubits']} layers={cfg['n_layers']} patch={cfg['patch_size']} | "
            f"select=val_{cfg['selection_metric']}")


def main():
    args = parse_args()

    configs = [resolve(c, args) for c in CONFIGS]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c["name"] for c in configs}
        if unknown:
            raise SystemExit(f"Config desconhecida: {sorted(unknown)}. "
                             f"Disponiveis: {[c['name'] for c in configs]}")
        configs = [c for c in configs if c["name"] in wanted]

    device = get_device()
    print(f"Device: {device}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Configs a rodar ({len(configs)}), sequencialmente:")
    for c in configs:
        print(f"  - {c['name']}: {describe(c)}")

    loader_cache = {}
    summaries = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{i}/{len(configs)}] {FAMILY.upper()} :: {cfg['name']} — "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# {describe(cfg)}")
        print(f"{'#' * 80}")

        key = (cfg["dataset"], cfg["batch_size"], cfg["image_size"])
        if key not in loader_cache:
            loader_cache[key] = build_dataloaders(cfg)
            train_loader, test_loader, val_loader = loader_cache[key]
            print(f"  Training batches: {len(train_loader)}")
            print(f"  Validation batches: {len(val_loader)}")
            print(f"  Test batches: {len(test_loader)}")
        train_loader, test_loader, val_loader = loader_cache[key]

        cfg["input_h"], cfg["input_w"] = probe_input_shape(train_loader, cfg)
        # The quanvolution map is interpolated to this size, so the FC input stays
        # the same whatever the loader hands over.
        cfg["output_size"] = (max(1, cfg["input_h"] // cfg["patch_size"]),
                              max(1, cfg["input_w"] // cfg["patch_size"]))
        print(f"  Input size: {cfg['input_h']}x{cfg['input_w']} | "
              f"Quantum output_size: {cfg['output_size']}")

        common = dict(
            reports_dir=reports_dir(cfg),
            run_tag=cfg["run_tag"],
            n_runs=args.runs,
            num_classes=cfg["num_classes"],
            hyperparams=hyperparams(cfg),
            report_meta=report_meta(cfg),
            title_prefix=f"{cfg['dataset_name']} {cfg['run_tag']}",
        )
        build_fn = make_build_fn(cfg, device)

        if args.test_only:
            result = evaluate_checkpoints(build_fn, test_loader, device, **common)
        else:
            result = run_multi_seed(build_fn, train_loader, val_loader, test_loader,
                                    device, epochs=cfg["num_epochs"],
                                    selection_metric=cfg["selection_metric"], **common)

        summaries.append((cfg["name"], result))

    print(f"\n{'=' * 80}")
    print(f"RESUMO — {FAMILY.upper()} ({len(summaries)} configs)")
    print(f"{'=' * 80}")
    for name, result in summaries:
        s = result["stats"]
        print(f"{name:<12} acc {s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}  "
              f"f1 {s['f1']['mean']:.4f}±{s['f1']['std']:.4f}  "
              f"auc {s['auc']['mean']:.4f}±{s['auc']['std']:.4f}  "
              f"ap {s['ap']['mean']:.4f}±{s['ap']['std']:.4f}")
        print(f"{'':<12} -> {result['stats_path']}")


if __name__ == "__main__":
    main()

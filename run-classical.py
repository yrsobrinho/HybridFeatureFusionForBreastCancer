"""Classical baselines — multi-run evaluation, all configs in one sequential pass.

Each entry in CONFIGS is a distinct model trained/evaluated independently, in the
order listed: ResNet18 first, then the ShallowCNN.

    python run-classical.py                     # BUS-UCLM, resnet then shallow, 5 runs each
    python run-classical.py --only shallow      # a single config
    python run-classical.py --dataset breastmnist
    python run-classical.py --test-only         # no training: re-tests best_*.pth on disk
    python run-classical.py --runs 3 --epochs 20
    python run-classical.py --selection-metric auc   # pick best_*.pth on val AUC

The best checkpoint of each run is picked on val balanced accuracy for BUS-UCLM and on
val accuracy elsewhere; --selection-metric overrides it (see SELECTION_METRIC_DEFAULTS).

Outputs go to reports/<Dataset>/classical/<model>/ (see reports/README.md).
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.main.classical import ClassicalModel, ShallowCNN
from src.utils.dataloader import (
    BCDRLoader,
    BUSUCLMLoader,
    BreastMNISTLoader,
    INbreastLoader,
)
from src.utils.multirun import evaluate_checkpoints, run_multi_seed
from src.utils.training import SELECTION_METRICS

FAMILY = "classical"

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
    "num_epochs": 100,
    "learning_rate": 1e-3,
}

CONFIGS = [
    {
        # ResNet18 pre-trained on ImageNet, conv1 adapted to 1 channel, fine-tuned end-to-end.
        "name": "resnet",
        "model_type": "resnet",
        # Checkpoint stem kept as `resnet_run<N>` — src/utils/fusion_experiment.py
        # reloads these as the classical branch of the fusion models.
        "run_tag": "resnet",
    },
    {
        # 3-block CNN trained from scratch, no pre-trained encoder.
        "name": "shallow",
        "model_type": "shallow",
        "run_tag": "shallow_no_encoder",
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
                       ("image_size", args.image_size), ("learning_rate", args.lr)]:
        if value is not None:
            cfg[key] = value
    cfg["dataset_name"] = DATASETS[cfg["dataset"]]
    cfg["selection_metric"] = args.selection_metric or SELECTION_METRIC_DEFAULTS.get(
        cfg["dataset"], DEFAULT_SELECTION_METRIC)
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


def build_model(cfg, device):
    if cfg["model_type"] == "shallow":
        model = ShallowCNN(num_classes=cfg["num_classes"],
                           input_channels=cfg["input_channels"])
    elif cfg["model_type"] == "resnet":
        model = ClassicalModel(num_classes=cfg["num_classes"])
    else:
        raise ValueError(f"Modelo classico nao suportado: {cfg['model_type']}")
    return model.to(device)


def make_build_fn(cfg, device):
    """Return `build_fn(seed) -> (model, optimizer, model_name)` for src.utils.multirun."""

    def build_fn(seed):
        model = build_model(cfg, device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
        return model, optimizer, f"{cfg['run_tag']}_run{seed + 1}"

    return build_fn


def reports_dir(cfg):
    """`reports/<Dataset>/classical/<model_type>` under the project root."""
    return os.path.join(str(PROJECT_ROOT), "reports", cfg["dataset_name"],
                        FAMILY, cfg["model_type"])


def hyperparams(cfg):
    keys = ["dataset", "model_type", "batch_size", "image_size",
            "learning_rate", "num_epochs", "num_classes", "selection_metric"]
    return {k: cfg[k] for k in keys}


def report_meta(cfg):
    return {
        "Dataset": cfg["dataset_name"],
        "Model Variant": cfg["model_type"],
        "Epochs per run": cfg["num_epochs"],
        "Batch Size": cfg["batch_size"],
        "Image Size": cfg["image_size"],
        "Learning Rate": cfg["learning_rate"],
        "Checkpoint selection": f"val_{cfg['selection_metric']}",
    }


def describe(cfg):
    return (f"{cfg['model_type']} | {cfg['dataset_name']} | epochs={cfg['num_epochs']} | "
            f"bs={cfg['batch_size']} | img={cfg['image_size']} | lr={cfg['learning_rate']} | "
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

        common = dict(
            reports_dir=reports_dir(cfg),
            run_tag=cfg["run_tag"],
            n_runs=args.runs,
            num_classes=cfg["num_classes"],
            hyperparams=hyperparams(cfg),
            report_meta=report_meta(cfg),
            title_prefix=f"{cfg['dataset_name']} {cfg['model_type']}",
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

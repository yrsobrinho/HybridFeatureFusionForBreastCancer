"""SHF (Shallow Hybrid Fusion) — multi-run evaluation, all configs in one sequential pass.

Each entry in CONFIGS is a distinct model and is trained/evaluated independently:
resnet+quantum and shallow+quantum keep their own freeze policy.

    python run-shf.py                    # both configs, 5 runs each
    python run-shf.py --only shallow     # a single config
    python run-shf.py --test-only        # no training: re-tests best_*.pth on disk
    python run-shf.py --runs 3 --epochs 5

Outputs go to reports/shf/<classical>+<quantum>/ (see reports/README.md).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import fusion_experiment as fx
from src.utils.multirun import evaluate_checkpoints, run_multi_seed

FAMILY = "shf"

CONFIGS = [
    {
        "name": "resnet",
        "classical_model": "resnet",
        "quantum_model": "extended_quantum",
        # ResNet is pre-trained and large: only the projections + fusion head are learned.
        "freeze_classical": True,
        "freeze_quantum": True,
        "num_epochs": 10,
    },
    {
        "name": "shallow",
        "classical_model": "shallow",
        "quantum_model": "extended_quantum",
        # ShallowCNN is small enough to fine-tune end-to-end alongside the quantum branch.
        "freeze_classical": False,
        "freeze_quantum": False,
        "num_epochs": 10,
    },
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="run only these configs (by 'name'); default: all, in order")
    p.add_argument("--runs", type=int, default=5, help="seeds per config (default: 5)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override num_epochs for every config")
    p.add_argument("--test-only", action="store_true",
                   help="skip training; reload best_*.pth and regenerate metrics/figures")
    return p.parse_args()


def main():
    args = parse_args()

    configs = [fx.resolve(c) for c in CONFIGS]
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c["name"] for c in configs}
        if unknown:
            raise SystemExit(f"Config desconhecida: {sorted(unknown)}. "
                             f"Disponiveis: {[c['name'] for c in configs]}")
        configs = [c for c in configs if c["name"] in wanted]
    if args.epochs is not None:
        for c in configs:
            c["num_epochs"] = args.epochs

    device = fx.get_device()
    print(f"Device: {device}")
    print(f"Project root: {fx.PROJECT_ROOT}")
    print(f"Configs a rodar ({len(configs)}), sequencialmente:")
    for c in configs:
        print(f"  - {c['name']}: {fx.describe(c)}")

    loader_cache = {}
    summaries = []

    for i, cfg in enumerate(configs, 1):
        print(f"\n{'#' * 80}")
        print(f"# [{i}/{len(configs)}] {FAMILY.upper()} :: {cfg['name']} — "
              f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# {fx.describe(cfg)}")
        print(f"{'#' * 80}")

        key = (cfg["dataset"], cfg["batch_size"], cfg["image_size"])
        if key not in loader_cache:
            loader_cache[key] = fx.build_dataloaders(cfg)
        train_loader, test_loader, val_loader = loader_cache[key]

        common = dict(
            reports_dir=fx.reports_dir(cfg, FAMILY),
            run_tag=fx.run_tag(cfg),
            n_runs=args.runs,
            num_classes=cfg["num_classes"],
            hyperparams=fx.hyperparams(cfg),
            report_meta=fx.report_meta(cfg),
            title_prefix=f"SHF {cfg['classical_model']}+{cfg['quantum_model']}",
        )
        build_fn = fx.make_build_fn(cfg, device)

        if args.test_only:
            result = evaluate_checkpoints(build_fn, test_loader, device, **common)
        else:
            result = run_multi_seed(build_fn, train_loader, val_loader, test_loader,
                                    device, epochs=cfg["num_epochs"], **common)

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

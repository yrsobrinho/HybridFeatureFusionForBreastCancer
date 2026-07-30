"""TSHF (Temperature Shallow Hybrid Fusion) — gamma ablation, all configs in one sequential pass.

The three gamma conditions live in the same graph (src/models/fusion.py):

    'none'      gamma = 1.0 frozen  -> exact SHF baseline
    'fixed'     gamma frozen at init_gamma -> value sweep
    'trainable' gamma learned from init_gamma -> TSHF

Each (classical backbone x gamma condition) pair is a distinct model, trained and
evaluated independently. The default pass is the ablation itself — 'none' plus the
fixed-value sweep. The trainable arm is TSHF proper, so it is skipped unless named
explicitly with --only / --only-gamma:

    python run-tshf.py                          # ablation grid (no trainable), 5 runs each
    python run-tshf.py --only shallow-trainable # a single config
    python run-tshf.py --only-model shallow     # ablation grid, shallow only
    python run-tshf.py --only-gamma trainable   # both backbones, trainable gamma only
    python run-tshf.py --test-only              # no training: re-tests best_*.pth on disk

`--quantum` picks the quantum branch, orthogonally to the gamma condition:

    python run-tshf.py --quantum extended_quantum --only-gamma trainable

Outputs go to reports/tshf_ablation/<gamma_tag>/<classical>+<quantum>/, so the two
quantum branches land in separate folders (see reports/README.md).
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

FAMILY = "tshf_ablation_busuclm"

MODELS = [
    {"name": "resnet", "classical_model": "resnet", "freeze_classical": False, "freeze_quantum": False},
    {"name": "shallow", "classical_model": "shallow", "freeze_classical": False, "freeze_quantum": False},
]

# Gamma conditions. 'fixed' at 1.0 is omitted on purpose — it is identical to 'none'.
GAMMA_FIXED_VALUES = [0.001, 0.01, 0.1, 0.5, 2.0]

# Ablation grid: baseline + fixed-value sweep. Trainable gamma is TSHF itself, not an
# ablation arm, so it stays out of the default pass and only runs when asked for by name.
GAMMA_CONDITIONS = (
    [{"name": "none", "gamma_mode": "none"}]
    + [{"name": f"fixed{v:g}".replace(".", "p"), "gamma_mode": "fixed", "init_gamma": v}
       for v in GAMMA_FIXED_VALUES]
)
TRAINABLE_CONDITIONS = [
    {"name": "trainable", "gamma_mode": "trainable", "init_gamma": 1.0, "lr_gamma": 1e-2},
]
ALL_GAMMA_CONDITIONS = GAMMA_CONDITIONS + TRAINABLE_CONDITIONS

SHARED = {
    "num_epochs": 10,
    "lr_fusion": 1e-1,
}

# Quantum branch. Each one is initialized from its own pre-trained checkpoint
# (see fusion_experiment.PRETRAINED) and writes to its own reports subtree, so the
# two branches never share results:
#   'quantum'           fixed random quanvolution (QuantumModel)
#   'extended_quantum'  trainable quanvolution (ExtentedQuantumModel)
QUANTUM_MODELS = ["quantum", "extended_quantum"]


def build_configs(quantum_model):
    configs = []
    for model in MODELS:
        for gamma in ALL_GAMMA_CONDITIONS:
            cfg = {**SHARED,
                   "quantum_model": quantum_model,
                   **{k: v for k, v in model.items() if k != "name"},
                   **{k: v for k, v in gamma.items() if k != "name"},
                   "name": f"{model['name']}-{gamma['name']}",
                   "model_name": model["name"],
                   "gamma_name": gamma["name"]}
            configs.append(fx.resolve(cfg))
    return configs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="run only these configs (e.g. shallow-trainable)")
    p.add_argument("--only-model", nargs="+", metavar="NAME",
                   choices=[m["name"] for m in MODELS],
                   help="restrict to these backbones")
    p.add_argument("--only-gamma", nargs="+", metavar="NAME",
                   choices=[g["name"] for g in ALL_GAMMA_CONDITIONS],
                   help="restrict to these gamma conditions (trainable only runs if named here)")
    p.add_argument("--quantum", default="extended_quantum", choices=QUANTUM_MODELS,
                   help="quantum branch: 'quantum' = fixed quanvolution, "
                        "'extended_quantum' = trainable quanvolution (default: quantum)")
    p.add_argument("--runs", type=int, default=5, help="seeds per config (default: 5)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override num_epochs for every config")
    p.add_argument("--test-only", action="store_true",
                   help="skip training; reload best_*.pth and regenerate metrics/figures")
    return p.parse_args()


def main():
    args = parse_args()

    configs = build_configs(args.quantum)
    if args.only:
        wanted = set(args.only)
        unknown = wanted - {c["name"] for c in configs}
        if unknown:
            raise SystemExit(f"Config desconhecida: {sorted(unknown)}. "
                             f"Disponiveis: {[c['name'] for c in configs]}")
        configs = [c for c in configs if c["name"] in wanted]
    if args.only_model:
        configs = [c for c in configs if c["model_name"] in set(args.only_model)]
    if args.only_gamma:
        configs = [c for c in configs if c["gamma_name"] in set(args.only_gamma)]
    if not args.only and not args.only_gamma:
        # Default ablation pass excludes the trainable-gamma arm.
        ablation = {g["name"] for g in GAMMA_CONDITIONS}
        configs = [c for c in configs if c["gamma_name"] in ablation]
    if args.epochs is not None:
        for c in configs:
            c["num_epochs"] = args.epochs
    if not configs:
        raise SystemExit("Nenhuma config selecionada pelos filtros.")

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
            title_prefix=f"TSHF {cfg['classical_model']}+{cfg['quantum_model']} ({cfg['gamma_name']})",
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
        line = (f"{name:<22} acc {s['accuracy']['mean']:.4f}±{s['accuracy']['std']:.4f}  "
                f"f1 {s['f1']['mean']:.4f}±{s['f1']['std']:.4f}  "
                f"auc {s['auc']['mean']:.4f}±{s['auc']['std']:.4f}  "
                f"ap {s['ap']['mean']:.4f}±{s['ap']['std']:.4f}")
        if "gamma" in s:
            line += f"  γ {s['gamma']['mean']:.4f}±{s['gamma']['std']:.4f}"
        print(line)
        print(f"{'':<22} -> {result['stats_path']}")


if __name__ == "__main__":
    main()

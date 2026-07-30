"""Runtime and computational-cost benchmark — classical vs quantum vs fusion.

Answers the reviewer question "what does the quantum simulation actually cost?"
with two independent kinds of evidence, both produced in one pass:

  (A) Analytical cost — hardware-independent and deterministic: trainable
      parameters (split classical/quantum), classical multiply-accumulates,
      number of circuit evaluations per image, and the expanded gate count and
      depth of one circuit (via qml.specs at level="device", so templates like
      RandomLayers are counted as the gates they decompose into, not as 1).

  (B) Measured wall-clock — warmup, then timed inference forwards and full
      training steps (forward + backward + optimizer step), reported as
      mean +- std over repeats on the same fixed batch.

Nothing here imports or mutates the training pipeline, so it cannot disturb
results already in reports/.

    python benchmark_cost.py                          # all models, CPU and GPU
    python benchmark_cost.py --only resnet extended   # a subset
    python benchmark_cost.py --device cpu             # single device
    python benchmark_cost.py --dataset breastmnist    # real batches/epoch for the estimate
    python benchmark_cost.py --repeats 50 --budget 60

Outputs land in reports/cost/: a readable .txt, a .csv for plotting, and a
.md table ready to paste into the paper.

Cost warning: the non-trainable `quantum` config evaluates one PennyLane
circuit per patch in a Python loop (~3.7 s per batch of 16 at 28x28, patch 2),
so it dominates the run. `--budget` caps the time spent on each measurement;
lower it, or use --only, for a quick pass.
"""

import argparse
import csv
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.fusion import OnlineFusionModel, TemperatureFusionModel
from src.models.main.classical import ClassicalModel, ShallowCNN
from src.models.main.models import ExtentedQuantumModel, QuantumModel

DEFAULTS = {
    "batch_size": 16,
    "image_size": 28,
    "num_classes": 2,
    "input_channels": 1,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
    "feature_dim": 128,
    "output_size": (14, 14),
}

DATASETS = {
    "breastmnist": "BreastMNIST",
    "bcdr": "BCDR",
    "inbreast": "INbreast",
    "busuclm": "BUS-UCLM",
}


# --------------------------------------------------------------------------
# Model builders
# --------------------------------------------------------------------------

def _shallow(cfg, device):
    return ShallowCNN(num_classes=cfg["num_classes"],
                      input_channels=cfg["input_channels"]).to(device)


def _resnet(cfg, device):
    return ClassicalModel(num_classes=cfg["num_classes"]).to(device)


def _quantum(cfg, device):
    # The fixed circuit feeds one patch pixel per qubit, so n_qubits == patch_size ** 2.
    n_qubits = cfg["patch_size"] ** 2
    rand_params = np.random.uniform(high=2 * np.pi, size=(cfg["n_layers"], n_qubits))
    return QuantumModel(
        n_qubits=n_qubits, n_layers=cfg["n_layers"], rand_params=rand_params,
        output_size=cfg["output_size"], patch_size=cfg["patch_size"],
        num_classes=cfg["num_classes"],
    ).to(device)


def _extended(cfg, device):
    return ExtentedQuantumModel(
        n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"],
        num_classes=cfg["num_classes"], patch_size=cfg["patch_size"],
        mode="2d", stride=None, use_parallel=True,
    ).to(device)


def _fusion(cls, classical_builder, cfg, device, **kwargs):
    """SHF/TSHF over a fresh classical + trainable-quantum pair.

    Weights are random: wall-clock and operation counts do not depend on the
    values, so no checkpoint from reports/ needs to be touched.
    """
    return cls(
        classical_model=classical_builder(cfg, device),
        quantum_model=_extended(cfg, device),
        feature_dim=cfg["feature_dim"], num_classes=cfg["num_classes"],
        freeze_classical=False, freeze_quantum=False, **kwargs,
    ).to(device)


# name -> (builder, family). `family` marks the classical baselines used as the
# denominator of the slowdown column.
MODELS = {
    "shallow": (_shallow, "classical"),
    "resnet": (_resnet, "classical"),
    "quantum": (_quantum, "quantum"),
    "extended": (_extended, "quantum"),
    "shf-shallow": (lambda c, d: _fusion(OnlineFusionModel, _shallow, c, d), "fusion"),
    "shf-resnet": (lambda c, d: _fusion(OnlineFusionModel, _resnet, c, d), "fusion"),
    "tshf-shallow": (lambda c, d: _fusion(TemperatureFusionModel, _shallow, c, d,
                                          init_gamma=1.0, trainable_gamma=True), "fusion"),
    "tshf-resnet": (lambda c, d: _fusion(TemperatureFusionModel, _resnet, c, d,
                                         init_gamma=1.0, trainable_gamma=True), "fusion"),
}


# --------------------------------------------------------------------------
# (A) Analytical cost
# --------------------------------------------------------------------------

def count_macs(model, sample):
    """Multiply-accumulates of every Conv2d/Linear reached by one forward pass.

    Hook-based rather than analytic so lazily-created heads and the branches a
    fusion model actually executes are the ones counted. Quantum layers
    contribute nothing here by construction — they are reported separately as
    circuit evaluations, which is the honest unit for a simulated circuit.
    """
    total = [0]
    handles = []

    def conv_hook(module, inputs, output):
        # out_elements * (kernel volume / groups); bias excluded, as is conventional.
        kernel_ops = int(np.prod(module.kernel_size)) * (module.in_channels // module.groups)
        total[0] += int(output.numel()) * kernel_ops

    def linear_hook(module, inputs, output):
        total[0] += int(output.numel()) * module.in_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(sample)
    finally:
        for h in handles:
            h.remove()
    return total[0]


def _quantum_submodule(model):
    """The quantum-bearing module inside `model`, or None if it is purely classical."""
    if isinstance(model, (QuantumModel, ExtentedQuantumModel)):
        return model
    inner = getattr(model, "quantum_model", None)
    if isinstance(inner, (QuantumModel, ExtentedQuantumModel)):
        return inner
    return None


def circuit_specs(qmodel, h, w):
    """Circuits evaluated per image plus the expanded gate count of one circuit."""
    p = qmodel.patch_size
    n_qubits = qmodel.n_qubits

    if isinstance(qmodel, ExtentedQuantumModel):
        (out_h, out_w), _ = qmodel.quanv.calculate_output_shape(h, w)
        qnode = qmodel.quanv.qnode
        args = (torch.zeros(n_qubits), torch.zeros(qmodel.quanv.n_layers, n_qubits))
    else:
        # QuantumModel pads up to whole patches, then scans a non-overlapping grid.
        out_h = -(-h // p)
        out_w = -(-w // p)
        qnode = qmodel.circuit.circuit
        args = ([0.0] * n_qubits,)

    try:
        specs = qml.specs(qnode, level="device")(*args)
        res = specs.resources
        gates, depth = int(res.num_gates), int(res.depth)
        gate_types = ", ".join(f"{k}:{v}" for k, v in sorted(res.gate_types.items()))
    except Exception as exc:  # specs is a reporting nicety, never fatal
        gates = depth = 0
        gate_types = f"<unavailable: {exc}>"

    return {
        "circuits_per_image": out_h * out_w,
        "feature_map": f"{out_h}x{out_w}x{n_qubits}",
        "n_qubits": n_qubits,
        "circuit_gates": gates,
        "circuit_depth": depth,
        "gate_types": gate_types,
    }


def param_counts(model):
    """Total/trainable params, split into the quantum branch and everything else."""
    qmod = _quantum_submodule(model)
    q_ids = {id(p) for p in qmod.parameters()} if qmod is not None else set()

    total = trainable = q_total = c_total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        if id(p) in q_ids:
            q_total += n
        else:
            c_total += n
    return {"params_total": total, "params_trainable": trainable,
            "params_quantum": q_total, "params_classical": c_total}


# --------------------------------------------------------------------------
# (B) Measured wall-clock
# --------------------------------------------------------------------------

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def global_warmup(device):
    """Absorb torch's one-off backend initialisation before anything is timed.

    Without this the first model measured pays for lazy thread-pool/oneDNN (and,
    on CUDA, context) setup and comes out spuriously slower than models measured
    later — enough to rank ShallowCNN above ResNet18.
    """
    x = torch.randn(16, 1, 28, 28, device=device)
    net = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                        nn.Flatten(), nn.Linear(32 * 28 * 28, 2)).to(device)
    opt = torch.optim.Adam(net.parameters())
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(net(x), torch.zeros(16, dtype=torch.long, device=device)).backward()
        opt.step()
    with torch.no_grad():
        for _ in range(10):
            net(x)
    _sync(device)


def time_calls(fn, device, warmup, max_repeats, budget):
    """Run `fn` warmup times, then up to max_repeats timed calls within `budget` seconds.

    Returns a stats dict in milliseconds. The headline figure is the *median*:
    wall-clock on a shared machine has a long right tail (scheduler preemption,
    CPU contention), and a single stall can move the mean by more than the effect
    being measured — the first pass here produced std > mean on three of three
    models. Mean and std are kept alongside it for reporting conventions, and min
    approximates the contention-free cost.

    Repeats stop early once `budget` seconds have elapsed, so a slow model
    degrades to fewer samples instead of stalling the run; at least 3 timed calls
    are always taken.

    Warmup is adaptive for the same reason. A fixed warmup is priced in units of
    the thing being measured, and a quantum training step costs tens of seconds
    (backward through thousands of circuits), so `--warmup 3` alone can burn
    minutes per model before a single sample is taken. One probe call decides:
    cheap models get the full warmup, expensive ones get the probe as their only
    warmup, since backend init is already absorbed by global_warmup().
    """
    probe_start = time.perf_counter()
    fn()
    _sync(device)
    probe = time.perf_counter() - probe_start

    for _ in range(warmup - 1 if probe < 1.0 else 0):
        fn()
    _sync(device)

    samples = []
    started = time.perf_counter()
    for i in range(max_repeats):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if i + 1 >= 3 and (time.perf_counter() - started) > budget:
            break

    return {
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "n": len(samples),
    }


def measure(name, builder, cfg, device, args):
    h = w = cfg["image_size"]
    batch = torch.randn(cfg["batch_size"], cfg["input_channels"], h, w, device=device)
    labels = torch.randint(0, cfg["num_classes"], (cfg["batch_size"],), device=device)

    model = builder(cfg, device)

    # One forward before anything is counted: both quantum models and the fusion
    # projections build their heads lazily on first use, so parameters do not all
    # exist until now.
    model.eval()
    with torch.no_grad():
        model(batch)

    row = {"model": name, "family": MODELS[name][1], "device": device.type}
    row.update(param_counts(model))
    row["macs_classical"] = count_macs(model, batch)

    qmod = _quantum_submodule(model)
    if qmod is not None:
        row.update(circuit_specs(qmod, h, w))
        row["circuits_per_batch"] = row["circuits_per_image"] * cfg["batch_size"]
    else:
        row.update({"circuits_per_image": 0, "circuits_per_batch": 0, "feature_map": "-",
                    "n_qubits": 0, "circuit_gates": 0, "circuit_depth": 0, "gate_types": "-"})

    # Inference
    def infer():
        with torch.no_grad():
            model(batch)

    model.eval()
    s = time_calls(infer, device, args.warmup, args.repeats, args.budget)
    row.update({"infer_ms_batch": s["median"], "infer_ms_batch_mean": s["mean"],
                "infer_ms_batch_std": s["std"], "infer_ms_batch_min": s["min"],
                "infer_ms_image": s["median"] / cfg["batch_size"], "infer_n": s["n"]})

    # Full training step, which is what an epoch is actually made of.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()

    model.train()
    s = time_calls(train_step, device, args.warmup, args.repeats, args.budget)
    row.update({"train_ms_step": s["median"], "train_ms_step_mean": s["mean"],
                "train_ms_step_std": s["std"], "train_ms_step_min": s["min"],
                "train_n": s["n"]})

    if args.batches_per_epoch:
        row["est_epoch_s"] = s["median"] * args.batches_per_epoch / 1000.0
    else:
        row["est_epoch_s"] = None

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def add_slowdowns(rows):
    """Slowdown against the fastest classical baseline on the same device.

    The reference is chosen per metric: the model with the cheapest inference is
    not necessarily the one with the cheapest training step, and mixing the two
    produces ratios below 1 for a baseline compared against itself.
    """
    for device_type in {r["device"] for r in rows}:
        same = [r for r in rows if r["device"] == device_type]
        classical = [r for r in same if r["family"] == "classical"]
        if not classical:
            for r in same:
                r["infer_slowdown"] = r["train_slowdown"] = None
            continue
        infer_ref = min(classical, key=lambda r: r["infer_ms_batch"])
        train_ref = min(classical, key=lambda r: r["train_ms_step"])
        for r in same:
            r["infer_ref_model"] = infer_ref["model"]
            r["train_ref_model"] = train_ref["model"]
            r["infer_slowdown"] = r["infer_ms_batch"] / infer_ref["infer_ms_batch"]
            r["train_slowdown"] = r["train_ms_step"] / train_ref["train_ms_step"]


def _fmt(value, spec=".2f", dash="-"):
    return dash if value is None else format(value, spec)


def write_txt(rows, path, cfg, args):
    lines = []
    w = lines.append
    w("=" * 100)
    w("RUNTIME AND COMPUTATIONAL-COST BENCHMARK")
    w("=" * 100)
    w("")
    w(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"Torch: {torch.__version__} | PennyLane: {qml.__version__}")
    w(f"CPU threads: {torch.get_num_threads()}")
    if torch.cuda.is_available():
        w(f"GPU: {torch.cuda.get_device_name(0)}")
    w(f"Input: {cfg['batch_size']}x{cfg['input_channels']}x{cfg['image_size']}x{cfg['image_size']}")
    w(f"Quantum: n_qubits={cfg['n_qubits']} n_layers={cfg['n_layers']} "
      f"patch_size={cfg['patch_size']} | simulator: lightning.qubit (CPU-only)")
    w(f"Warmup: {args.warmup} | max repeats: {args.repeats} | budget: {args.budget}s per measurement")
    if args.batches_per_epoch:
        w(f"Batches per epoch: {args.batches_per_epoch}"
          + (f" (from --dataset {args.dataset})" if args.dataset else " (from --batches-per-epoch)"))
    w("")

    w("-" * 100)
    w("(A) ANALYTICAL COST  [hardware-independent]")
    w("-" * 100)
    head = f"{'Model':<14}{'Params':>11}{'Trainable':>11}{'Quantum p.':>11}{'MACs/img':>14}{'Circuits/img':>14}{'Gates':>7}{'Depth':>7}"
    w(head)
    w("-" * len(head))
    seen = set()
    for r in rows:
        if r["model"] in seen:
            continue
        seen.add(r["model"])
        w(f"{r['model']:<14}{r['params_total']:>11,}{r['params_trainable']:>11,}"
          f"{r['params_quantum']:>11,}{r['macs_classical'] // cfg['batch_size']:>14,}"
          f"{r['circuits_per_image']:>14,}{r['circuit_gates']:>7}{r['circuit_depth']:>7}")
    w("")
    for r in rows:
        if r["circuits_per_image"] and r["model"] in seen:
            seen.discard(r["model"])
            w(f"  {r['model']}: quanv map {r['feature_map']}, "
              f"{r['circuits_per_batch']:,} circuits per batch, gates [{r['gate_types']}]")
    w("")

    w("-" * 100)
    w("(B) MEASURED WALL-CLOCK  [median over timed repeats; mean/std/min in the CSV]")
    w("-" * 100)
    head = (f"{'Model':<14}{'Dev':>5}{'Infer ms/batch':>16}{'ms/img':>10}"
            f"{'Train ms/step':>16}{'Slowdown':>10}{'Epoch s':>10}{'n':>5}")
    w(head)
    w("-" * len(head))
    for r in rows:
        w(f"{r['model']:<14}{r['device']:>5}{r['infer_ms_batch']:>16.2f}"
          f"{r['infer_ms_image']:>10.2f}{r['train_ms_step']:>16.2f}"
          f"{_fmt(r.get('train_slowdown'), '.1f') + 'x':>10}"
          f"{_fmt(r.get('est_epoch_s'), '.1f'):>10}{r['train_n']:>5}")
    w("")
    w("Dispersion (same runs, to show the timing is stable):")
    head = f"{'Model':<14}{'Dev':>5}{'Infer mean+-std':>22}{'Infer min':>12}{'Train mean+-std':>22}{'Train min':>12}"
    w(head)
    w("-" * len(head))
    for r in rows:
        infer = f"{r['infer_ms_batch_mean']:.2f} +- {r['infer_ms_batch_std']:.2f}"
        train = f"{r['train_ms_step_mean']:.2f} +- {r['train_ms_step_std']:.2f}"
        w(f"{r['model']:<14}{r['device']:>5}{infer:>22}{r['infer_ms_batch_min']:>12.2f}"
          f"{train:>22}{r['train_ms_step_min']:>12.2f}")
    w("")
    train_refs = {r.get("train_ref_model") for r in rows if r.get("train_ref_model")}
    if train_refs:
        w(f"Slowdown is the median training step relative to the cheapest classical "
          f"baseline on the same device ({', '.join(sorted(train_refs))}).")
    if any(r["circuits_per_image"] and r["family"] == "quantum" for r in rows):
        w("")
        w("Note: lightning.qubit simulates on CPU, so the quantum branch pays the same")
        w("cost with or without a GPU — the overhead does not shrink on better hardware.")
        w("The non-trainable `quantum` circuit is fixed, so its features can be")
        w("precomputed once and cached, turning a per-epoch cost into a one-off")
        w("preprocessing pass: dataset_size x (infer ms/img) milliseconds.")
    w("")
    path.write_text("\n".join(lines) + "\n")


def write_csv(rows, path):
    fields = ["model", "family", "device", "params_total", "params_trainable",
              "params_classical", "params_quantum", "macs_classical",
              "circuits_per_image", "circuits_per_batch", "feature_map", "n_qubits",
              "circuit_gates", "circuit_depth",
              "infer_ms_batch", "infer_ms_batch_mean", "infer_ms_batch_std",
              "infer_ms_batch_min", "infer_ms_image", "infer_n",
              "train_ms_step", "train_ms_step_mean", "train_ms_step_std",
              "train_ms_step_min", "train_n",
              "est_epoch_s", "infer_slowdown", "train_slowdown"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, path, cfg):
    lines = ["# Computational cost", "",
             f"Batch {cfg['batch_size']}, {cfg['image_size']}x{cfg['image_size']}, "
             f"{cfg['n_qubits']} qubits, {cfg['n_layers']} layer(s), patch {cfg['patch_size']}. "
             "Simulator: `lightning.qubit` (CPU).", "",
             "| Model | Device | Params (trainable) | MACs/img | Circuits/img | "
             "Inference ms/img | Train ms/step | Slowdown |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['device']} | {r['params_total']:,} ({r['params_trainable']:,}) "
            f"| {r['macs_classical'] // cfg['batch_size']:,} "
            f"| {r['circuits_per_image']:,} "
            f"| {r['infer_ms_image']:.2f} "
            f"| {r['train_ms_step']:.1f} "
            f"| {_fmt(r.get('train_slowdown'), '.1f')}× |")
    lines += ["", "Median over timed repeats; `lightning.qubit` runs on CPU, so the "
                  "quantum branch does not benefit from the GPU."]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="NAME", choices=sorted(MODELS),
                   help=f"benchmark only these models (default: all). Choices: {', '.join(sorted(MODELS))}")
    p.add_argument("--device", choices=["cpu", "cuda", "both"], default="both",
                   help="device(s) to measure on (default: both, if CUDA is available)")
    p.add_argument("--repeats", type=int, default=30, help="max timed repeats (default: 30)")
    p.add_argument("--warmup", type=int, default=3, help="untimed warmup calls (default: 3)")
    p.add_argument("--budget", type=float, default=30.0,
                   help="seconds cap per measurement, min 3 repeats (default: 30)")
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--image-size", type=int, default=DEFAULTS["image_size"])
    p.add_argument("--n-qubits", type=int, default=DEFAULTS["n_qubits"])
    p.add_argument("--n-layers", type=int, default=DEFAULTS["n_layers"])
    p.add_argument("--patch-size", type=int, default=DEFAULTS["patch_size"])
    p.add_argument("--dataset", choices=sorted(DATASETS),
                   help="read the real batches/epoch from this dataset's train loader")
    p.add_argument("--batches-per-epoch", type=int,
                   help="batches per epoch for the epoch-time estimate (overrides --dataset)")
    p.add_argument("--outdir", default=str(PROJECT_ROOT / "reports" / "cost"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_batches_per_epoch(args):
    """Real train-loader length when --dataset is given; never fatal if data is missing."""
    if args.batches_per_epoch or not args.dataset:
        return args.batches_per_epoch
    try:
        from src.utils.dataloader import (BCDRLoader, BreastMNISTLoader,
                                          BUSUCLMLoader, INbreastLoader)
        kwargs = {"batch_size": args.batch_size, "image_size": args.image_size}
        if args.dataset == "bcdr":
            loader = BCDRLoader(root_dir=str(PROJECT_ROOT / "src/data/BCDR/png"), **kwargs)
        elif args.dataset == "inbreast":
            loader = INbreastLoader(root_dir=str(PROJECT_ROOT / "src/data/INbreast/png"), **kwargs)
        elif args.dataset == "busuclm":
            loader = BUSUCLMLoader(root_dir=str(PROJECT_ROOT / "src/data/BUS-UCLM/png"), **kwargs)
        else:
            loader = BreastMNISTLoader(**kwargs)
        train_loader = loader.get_dataloaders()[0]
        return len(train_loader)
    except Exception as exc:
        print(f"[warn] could not read batches/epoch from --dataset {args.dataset}: {exc}")
        print("[warn] epoch-time estimate will be omitted; pass --batches-per-epoch instead")
        return None


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = {**DEFAULTS, "batch_size": args.batch_size, "image_size": args.image_size,
           "n_qubits": args.n_qubits, "n_layers": args.n_layers,
           "patch_size": args.patch_size}
    args.batches_per_epoch = resolve_batches_per_epoch(args)

    if args.device == "both":
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available")
        devices = [torch.device(args.device)]

    names = args.only or list(MODELS)
    rows = []
    for device in devices:
        print(f"[{device.type}] warming up backend ...", flush=True)
        global_warmup(device)
        for name in names:
            builder, _ = MODELS[name]
            print(f"[{device.type}] {name} ...", flush=True)
            t0 = time.perf_counter()
            try:
                rows.append(measure(name, builder, cfg, device, args))
            except Exception as exc:
                print(f"[{device.type}] {name} FAILED: {exc}")
                continue
            r = rows[-1]
            print(f"[{device.type}] {name}: infer {r['infer_ms_batch']:.1f} ms/batch, "
                  f"train {r['train_ms_step']:.1f} ms/step "
                  f"({time.perf_counter() - t0:.0f}s total)", flush=True)

    if not rows:
        raise SystemExit("no model was benchmarked successfully")

    add_slowdowns(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_txt(rows, outdir / "cost_benchmark.txt", cfg, args)
    write_csv(rows, outdir / "cost_benchmark.csv")
    write_md(rows, outdir / "cost_benchmark.md", cfg)

    print(f"\nWrote {outdir}/cost_benchmark.{{txt,csv,md}}")
    print((outdir / "cost_benchmark.txt").read_text())


if __name__ == "__main__":
    main()

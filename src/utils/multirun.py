"""Multi-run evaluation shared by the fusion notebooks and the `run-*.py` scripts.

Trains a model `n_runs` times from scratch (seeds `0..n_runs-1`) and reports
mean +/- std plus a 95% confidence interval (Student's t) for every test metric,
together with per-run and aggregated confusion matrices, PR curves and ROC
curves.

Two entry points:

  run_multi_seed(...)      trains every run, then evaluates.
  evaluate_checkpoints(...) skips training and evaluates the `best_*.pth`
                           checkpoints already on disk, so a finished
                           experiment can be re-tested independently.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import t as t_dist
from sklearn.metrics import (
    auc as sk_auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.utils.training import test_model, training_loop

# Computer Modern through mathtext, so figures match the paper without a LaTeX install.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"

BASE_METRIC_KEYS = ["loss", "accuracy", "precision", "recall", "f1", "auc", "ap"]
_CURVE_GRID = np.linspace(0.0, 1.0, 201)


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def get_predictions(model, dataloader, device):
    """Return (y_true, y_pred, y_prob) where y_prob is P(class 1)."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            y_prob.extend(probs[:, 1].cpu().numpy().tolist() if outputs.shape[1] > 1
                          else probs[:, 0].cpu().numpy().tolist())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(np.asarray(labels).squeeze().reshape(-1).tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm, save_path, title, class_names=None, normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        cell_fmt = lambda v: f"{v:.1%}"
    else:
        disp = cm
        cell_fmt = lambda v: f"{int(round(v))}"
    n = cm.shape[0]
    names = class_names if class_names is not None else [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(1.4 * n + 2.5, 1.4 * n + 2.5))
    im = ax.imshow(disp, cmap="Blues", vmin=0)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    thresh = disp.max() / 2 if disp.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cell_fmt(disp[i, j]), ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _interp_pr(y_true, y_prob):
    """Precision interpolated on a fixed recall grid (recall ascending)."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    order = np.argsort(recall)
    return np.interp(_CURVE_GRID, recall[order], precision[order])


def _interp_roc(y_true, y_prob):
    """TPR interpolated on a fixed FPR grid."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    interp = np.interp(_CURVE_GRID, fpr, tpr)
    interp[0] = 0.0
    return interp


def plot_pr_curves(curves, save_path, title):
    """Per-run PR curves + mean curve with a +/-1 std band.

    `curves` is a list of (y_true, y_prob) arrays, one per run.
    Returns (mean_ap, std_ap).
    """
    aps = [average_precision_score(y, p) for y, p in curves]
    interps = np.stack([_interp_pr(y, p) for y, p in curves])
    mean_p = interps.mean(axis=0)
    std_p = interps.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_p)
    baseline = float(np.mean([np.mean(y) for y, _ in curves]))

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, ((y, p), ap) in enumerate(zip(curves, aps)):
        precision, recall, _ = precision_recall_curve(y, p)
        ax.plot(recall, precision, lw=1, alpha=0.35,
                label=f"Run {i + 1} (AP = {ap:.3f})")
    ax.plot(_CURVE_GRID, mean_p, color="#B2182B", lw=2.2,
            label=f"Mean (AP = {np.mean(aps):.3f} $\\pm$ {np.std(aps, ddof=1) if len(aps) > 1 else 0.0:.3f})")
    ax.fill_between(_CURVE_GRID, np.clip(mean_p - std_p, 0, 1), np.clip(mean_p + std_p, 0, 1),
                    color="#B2182B", alpha=0.15, label="$\\pm$ 1 std")
    ax.axhline(baseline, ls="--", color="grey", lw=1,
               label=f"Chance (prevalence = {baseline:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(np.mean(aps)), float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0


def plot_roc_curves(curves, save_path, title):
    """Per-run ROC curves + mean curve with a +/-1 std band. Returns (mean_auc, std_auc)."""
    interps, aucs = [], []
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, (y, p) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y, p)
        run_auc = sk_auc(fpr, tpr)
        aucs.append(run_auc)
        interps.append(_interp_roc(y, p))
        ax.plot(fpr, tpr, lw=1, alpha=0.35, label=f"Run {i + 1} (AUC = {run_auc:.3f})")

    interps = np.stack(interps)
    mean_tpr = interps.mean(axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = interps.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_tpr)
    ax.plot(_CURVE_GRID, mean_tpr, color="#2166AC", lw=2.2,
            label=f"Mean (AUC = {np.mean(aucs):.3f} $\\pm$ {np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0:.3f})")
    ax.fill_between(_CURVE_GRID, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1),
                    color="#2166AC", alpha=0.15, label="$\\pm$ 1 std")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0


# ---------------------------------------------------------------------------
# Statistics + report
# ---------------------------------------------------------------------------

def summarize_metrics(run_results, metric_keys, confidence_level=0.95):
    stats = {}
    for key in metric_keys:
        values = np.array([r[key] for r in run_results], dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        if n > 1:
            t_crit = float(t_dist.ppf((1 + confidence_level) / 2, df=n - 1))
            margin = t_crit * std / np.sqrt(n)
        else:
            margin = 0.0
        stats[key] = {
            "mean": mean, "std": std,
            "min": float(np.min(values)), "max": float(np.max(values)),
            "ci_lower": mean - margin, "ci_upper": mean + margin,
        }
    return stats


def _write_report(stats_path, run_results, stats, metric_keys, cm_sum, class_names,
                  report_meta, n_runs, confidence_level, figure_paths):
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MULTI-RUN STATISTICS ({n_runs} runs)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for k, v in report_meta.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("PER-RUN TEST RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Run':<5}" + "".join(f"{k.capitalize():<12}" for k in metric_keys) + "\n")
        for r in run_results:
            f.write(f"{r['run']:<5}" + "".join(f"{r[k]:<12.4f}" for k in metric_keys) + "\n")
        f.write("\n")

        f.write(f"SUMMARY STATISTICS (mean +/- std, min, max, "
                f"{int(confidence_level * 100)}% CI [t-distribution])\n")
        f.write("-" * 80 + "\n")
        for key in metric_keys:
            s = stats[key]
            f.write(f"{key.capitalize():<12}: {s['mean']:.4f} +/- {s['std']:.4f}"
                    f"  (min: {s['min']:.4f}, max: {s['max']:.4f})"
                    f"  {int(confidence_level * 100)}% CI: [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]\n")
        f.write("\n")

        f.write(f"CONFUSION MATRIX - SUMMED OVER {n_runs} RUNS (rows = true, cols = predicted)\n")
        f.write("-" * 80 + "\n")
        f.write("            " + "".join(f"{lbl:>12}" for lbl in class_names) + "\n")
        for lbl, row in zip(class_names, cm_sum.astype(int)):
            f.write(f"{lbl:>12}" + "".join(f"{v:>12d}" for v in row) + "\n")
        f.write("\n")

        f.write(f"CONFUSION MATRIX - ROW-NORMALIZED MEAN OVER {n_runs} RUNS\n")
        f.write("-" * 80 + "\n")
        cm_norm = cm_sum.astype(float) / cm_sum.sum(axis=1, keepdims=True)
        f.write("            " + "".join(f"{lbl:>12}" for lbl in class_names) + "\n")
        for lbl, row in zip(class_names, cm_norm):
            f.write(f"{lbl:>12}" + "".join(f"{v:>12.2%}" for v in row) + "\n")
        f.write("\n")

        f.write("FIGURES\n")
        f.write("-" * 80 + "\n")
        for name, path in figure_paths.items():
            f.write(f"{name}: {path}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _finalize(run_results, run_confusion_matrices, run_curves, *, reports_dir,
              multi_run_dir, run_tag, n_runs, num_classes, class_names,
              report_meta, confidence_level, title_prefix):
    metric_keys = list(BASE_METRIC_KEYS)
    if all("gamma" in r for r in run_results):
        metric_keys.append("gamma")

    stats = summarize_metrics(run_results, metric_keys, confidence_level)

    cm_sum = np.stack(run_confusion_matrices).sum(axis=0)
    figure_paths = {
        "Confusion matrix (sum)": os.path.join(multi_run_dir, "confusion_matrix_sum.png"),
        "Confusion matrix (row-normalized)": os.path.join(multi_run_dir, "confusion_matrix_mean_normalized.png"),
        "Precision-Recall curves": os.path.join(multi_run_dir, "pr_curve.png"),
        "ROC curves": os.path.join(multi_run_dir, "roc_curve.png"),
    }
    plot_confusion_matrix(cm_sum.astype(int), figure_paths["Confusion matrix (sum)"],
                          f"Confusion Matrix - Sum over {n_runs} runs", class_names)
    plot_confusion_matrix(cm_sum, figure_paths["Confusion matrix (row-normalized)"],
                          f"Confusion Matrix - Row-Normalized (mean over {n_runs} runs)",
                          class_names, normalize=True)
    plot_pr_curves(run_curves, figure_paths["Precision-Recall curves"],
                   f"{title_prefix} - Precision-Recall ({n_runs} runs)")
    plot_roc_curves(run_curves, figure_paths["ROC curves"],
                    f"{title_prefix} - ROC ({n_runs} runs)")

    # Raw curve data, so the figures can be re-plotted / merged without retraining.
    np.savez(os.path.join(multi_run_dir, "curve_data.npz"),
             **{f"y_true_run{i + 1}": y for i, (y, _) in enumerate(run_curves)},
             **{f"y_prob_run{i + 1}": p for i, (_, p) in enumerate(run_curves)})

    stats_path = os.path.join(reports_dir, f"multi_run_statistics_{run_tag}.txt")
    os.makedirs(reports_dir, exist_ok=True)
    _write_report(stats_path, run_results, stats, metric_keys, cm_sum, class_names,
                  report_meta, n_runs, confidence_level, figure_paths)

    print(f"\n✓ Multi-run statistics saved to: {stats_path}")
    for key in metric_keys:
        s = stats[key]
        print(f"  {key.capitalize():<12}: {s['mean']:.4f} +/- {s['std']:.4f}"
              f"  (min: {s['min']:.4f}, max: {s['max']:.4f})"
              f"  {int(confidence_level * 100)}% CI: [{s['ci_lower']:.4f}, {s['ci_upper']:.4f}]")
    print("\nSummed confusion matrix:")
    print(cm_sum.astype(int))

    return {
        "run_results": run_results,
        "stats": stats,
        "metric_keys": metric_keys,
        "cm_sum": cm_sum,
        "curves": run_curves,
        "stats_path": stats_path,
        "figure_paths": figure_paths,
        "multi_run_dir": multi_run_dir,
    }


def restore_quantum_rand_params(model, checkpoint_path):
    """Rebuild the fixed quantum circuit from the sidecar saved next to a checkpoint.

    `NonTrainableLayer.rand_params` is not a tensor and never reaches the
    state_dict, and its qnode closes over the array passed at construction — so
    reloading weights alone would evaluate a *different* random circuit than the
    one that was trained. `training_loop` writes the array to
    `<checkpoint>_rand_params.npy`; here we read it back and reconstruct the layer.
    """
    sidecar = f"{os.path.splitext(checkpoint_path)[0]}_rand_params.npy"
    if not os.path.isfile(sidecar):
        return False

    quantum = getattr(model, "quantum_model", model)
    circuit = getattr(quantum, "circuit", None)
    if circuit is None or not hasattr(circuit, "rand_params"):
        return False

    rand_params = np.load(sidecar)
    quantum.circuit = type(circuit)(circuit.n_qubits, circuit.n_layers, rand_params)
    return True


def _evaluate_one(model, test_loader, device, criterion, hyperparams, log_path,
                  num_classes, run_idx, cm_path=None, class_names=None):
    if os.path.isfile(log_path):
        os.remove(log_path)

    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_model(
        model=model, dataloader=test_loader, criterion=criterion, device=device,
        hyperparams=hyperparams, output_file=log_path,
        extract_features=False, compute_detailed_metrics=True,
    )

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    if cm_path is not None:
        plot_confusion_matrix(cm, cm_path, f"Confusion Matrix - Run {run_idx + 1}", class_names)

    record = {
        "run": run_idx + 1,
        "loss": test_loss, "accuracy": test_acc, "precision": test_prec,
        "recall": test_rec, "f1": test_f1, "auc": test_auc,
        "ap": float(average_precision_score(y_true, y_prob)),
    }
    gamma = getattr(model, "gamma", None)
    if gamma is not None:
        record["gamma"] = float(gamma.item()) if hasattr(gamma, "item") else float(gamma)
    return record, cm, (y_true, y_prob)


def run_multi_seed(build_fn, train_loader, val_loader, test_loader, device, *,
                   reports_dir, run_tag, n_runs=5, epochs=10, num_classes=2,
                   hyperparams=None, report_meta=None, class_names=None,
                   confidence_level=0.95, title_prefix=None, multi_run_dir=None,
                   selection_metric="accuracy"):
    """Train `n_runs` models from scratch (seeds 0..n_runs-1) and aggregate the results.

    `build_fn(seed)` must return `(model, optimizer, model_name)` — a freshly
    built model with its lazy layers already materialized.

    `selection_metric` is the validation metric each run's `best_*.pth` is chosen on
    (see `src.utils.training.SELECTION_METRICS`); on an imbalanced validation split
    prefer `balanced_accuracy` or `auc` over the `accuracy` default.
    """
    multi_run_dir = multi_run_dir or os.path.join(reports_dir, "multi_run")
    os.makedirs(multi_run_dir, exist_ok=True)
    class_names = class_names or [str(i) for i in range(num_classes)]
    hyperparams = hyperparams or {}
    report_meta = report_meta or {}
    title_prefix = title_prefix or run_tag

    run_results, run_confusion_matrices, run_curves = [], [], []

    for run_idx in range(n_runs):
        seed = run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n{'=' * 80}")
        print(f"RUN {run_idx + 1}/{n_runs} (seed={seed}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 80}\n")

        model, optimizer, model_name = build_fn(seed)
        criterion = nn.CrossEntropyLoss()

        best_path = os.path.join(multi_run_dir, f"best_{model_name}.pth")
        if os.path.isfile(best_path):
            os.remove(best_path)

        training_loop(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=criterion, optimizer=optimizer, device=device,
            num_epochs=epochs, save_dir=multi_run_dir, model_name=model_name,
            extract_features=False, save_best_only=True, save_all_epochs=False,
            selection_metric=selection_metric,
        )

        if os.path.isfile(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        else:
            print(f"⚠ Warning: run {run_idx + 1} produced no best checkpoint — "
                  f"testing with final-epoch weights instead.")

        record, cm, curve = _evaluate_one(
            model, test_loader, device, criterion, hyperparams,
            os.path.join(multi_run_dir, f"test_log_run{run_idx + 1}.txt"),
            num_classes, run_idx,
            cm_path=os.path.join(multi_run_dir, f"confusion_matrix_run{run_idx + 1}.png"),
            class_names=class_names)
        run_results.append(record)
        run_confusion_matrices.append(cm)
        run_curves.append(curve)

        print(f"\n✓ Run {run_idx + 1} test accuracy: {record['accuracy']:.4f} | AP: {record['ap']:.4f}")

    print(f"\n{'=' * 80}")
    print(f"All {n_runs} runs complete")
    print(f"{'=' * 80}")

    return _finalize(run_results, run_confusion_matrices, run_curves,
                     reports_dir=reports_dir, multi_run_dir=multi_run_dir,
                     run_tag=run_tag, n_runs=n_runs, num_classes=num_classes,
                     class_names=class_names, report_meta=report_meta,
                     confidence_level=confidence_level, title_prefix=title_prefix)


def evaluate_checkpoints(build_fn, test_loader, device, *, reports_dir, run_tag,
                         n_runs=5, num_classes=2, hyperparams=None, report_meta=None,
                         class_names=None, confidence_level=0.95, title_prefix=None,
                         multi_run_dir=None):
    """Test-only counterpart of `run_multi_seed`: no training, reuses `best_*.pth`.

    Rebuilds each run's architecture with `build_fn(seed)`, loads the checkpoint
    saved by a previous training pass, and regenerates every metric and figure.
    Runs whose checkpoint is missing are skipped with a warning.
    """
    multi_run_dir = multi_run_dir or os.path.join(reports_dir, "multi_run")
    class_names = class_names or [str(i) for i in range(num_classes)]
    hyperparams = hyperparams or {}
    report_meta = dict(report_meta or {})
    report_meta["Mode"] = "test-only (checkpoints reloaded, no training)"
    title_prefix = title_prefix or run_tag
    criterion = nn.CrossEntropyLoss()

    run_results, run_confusion_matrices, run_curves = [], [], []

    for run_idx in range(n_runs):
        # Same seeds as run_multi_seed, so anything build_fn draws at random
        # (e.g. the fixed quantum circuit) matches what was trained.
        seed = run_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model, _, model_name = build_fn(run_idx)
        best_path = os.path.join(multi_run_dir, f"best_{model_name}.pth")
        if not os.path.isfile(best_path):
            print(f"⚠ Skipping run {run_idx + 1}: no checkpoint at {best_path}")
            continue

        print(f"\n{'=' * 80}")
        print(f"TEST-ONLY RUN {run_idx + 1}/{n_runs} - {best_path}")
        print(f"{'=' * 80}\n")
        if restore_quantum_rand_params(model, best_path):
            print(f"✓ Fixed quantum circuit restored from the rand_params sidecar")
        model.load_state_dict(torch.load(best_path, map_location=device))

        record, cm, curve = _evaluate_one(
            model, test_loader, device, criterion, hyperparams,
            os.path.join(multi_run_dir, f"test_log_run{run_idx + 1}.txt"),
            num_classes, run_idx,
            cm_path=os.path.join(multi_run_dir, f"confusion_matrix_run{run_idx + 1}.png"),
            class_names=class_names)
        run_results.append(record)
        run_confusion_matrices.append(cm)
        run_curves.append(curve)

        print(f"\n✓ Run {run_idx + 1} test accuracy: {record['accuracy']:.4f} | AP: {record['ap']:.4f}")

    if not run_results:
        raise FileNotFoundError(
            f"No checkpoints found under {multi_run_dir} — train first with run_multi_seed().")

    return _finalize(run_results, run_confusion_matrices, run_curves,
                     reports_dir=reports_dir, multi_run_dir=multi_run_dir,
                     run_tag=f"{run_tag}_testonly", n_runs=len(run_results),
                     num_classes=num_classes, class_names=class_names,
                     report_meta=report_meta, confidence_level=confidence_level,
                     title_prefix=title_prefix)


__all__ = [
    "get_predictions",
    "restore_quantum_rand_params",
    "plot_confusion_matrix",
    "plot_pr_curves",
    "plot_roc_curves",
    "summarize_metrics",
    "run_multi_seed",
    "evaluate_checkpoints",
]

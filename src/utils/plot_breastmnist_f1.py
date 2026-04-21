from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Use Computer Modern through mathtext so it works without a LaTeX install.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 15


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


data = [
    {"Paradigm": "Classical", "Architecture": "SCNN", "Split": "Val", "F1": 0.9138},
    {"Paradigm": "Classical", "Architecture": "SCNN", "Split": "Test", "F1": 0.8814},
    {"Paradigm": "Classical", "Architecture": "ResNet", "Split": "Val", "F1": 0.9204},
    {"Paradigm": "Classical", "Architecture": "ResNet", "Split": "Test", "F1": 0.8879},

    {"Paradigm": "Quantum", "Architecture": "NT", "Split": "Val", "F1": 0.9000},
    {"Paradigm": "Quantum", "Architecture": "NT", "Split": "Test", "F1": 0.8703},
    {"Paradigm": "Quantum", "Architecture": "T", "Split": "Val", "F1": 0.8491},
    {"Paradigm": "Quantum", "Architecture": "T", "Split": "Test", "F1": 0.8482},

    {"Paradigm": "DHF", "Architecture": "SCNN (NT)", "Split": "Val", "F1": 0.9333},
    {"Paradigm": "DHF", "Architecture": "SCNN (NT)", "Split": "Test", "F1": 0.8879},
    {"Paradigm": "DHF", "Architecture": "SCNN (T)", "Split": "Val", "F1": 0.8966},
    {"Paradigm": "DHF", "Architecture": "SCNN (T)", "Split": "Test", "F1": 0.8908},
    {"Paradigm": "DHF", "Architecture": "ResNet (NT)", "Split": "Val", "F1": 0.9310},
    {"Paradigm": "DHF", "Architecture": "ResNet (NT)", "Split": "Test", "F1": 0.9035},
    {"Paradigm": "DHF", "Architecture": "ResNet (T)", "Split": "Val", "F1": 0.9333},
    {"Paradigm": "DHF", "Architecture": "ResNet (T)", "Split": "Test", "F1": 0.9004},

    {"Paradigm": "TSHF", "Architecture": "SCNN (NT)", "Split": "Val", "F1": 0.9153},
    {"Paradigm": "TSHF", "Architecture": "SCNN (NT)", "Split": "Test", "F1": 0.8947},
    {"Paradigm": "TSHF", "Architecture": "SCNN (T)", "Split": "Val", "F1": 0.8929},
    {"Paradigm": "TSHF", "Architecture": "SCNN (T)", "Split": "Test", "F1": 0.8584},
    {"Paradigm": "TSHF", "Architecture": "ResNet (NT)", "Split": "Val", "F1": 0.9322},
    {"Paradigm": "TSHF", "Architecture": "ResNet (NT)", "Split": "Test", "F1": 0.9177},
    {"Paradigm": "TSHF", "Architecture": "ResNet (T)", "Split": "Val", "F1": 0.9421},
    {"Paradigm": "TSHF", "Architecture": "ResNet (T)", "Split": "Test", "F1": 0.9145},

    {"Paradigm": "SHF", "Architecture": "SCNN (NT)", "Split": "Val", "F1": 0.8125},
    {"Paradigm": "SHF", "Architecture": "SCNN (NT)", "Split": "Test", "F1": 0.8425},
    {"Paradigm": "SHF", "Architecture": "SCNN (T)", "Split": "Val", "F1": 0.8397},
    {"Paradigm": "SHF", "Architecture": "SCNN (T)", "Split": "Test", "F1": 0.8308},
    {"Paradigm": "SHF", "Architecture": "ResNet (NT)", "Split": "Val", "F1": 0.9322},
    {"Paradigm": "SHF", "Architecture": "ResNet (NT)", "Split": "Test", "F1": 0.9123},
    {"Paradigm": "SHF", "Architecture": "ResNet (T)", "Split": "Val", "F1": 0.9333},
    {"Paradigm": "SHF", "Architecture": "ResNet (T)", "Split": "Test", "F1": 0.9068},
]


def build_model_color_map():
    """Build a consistent color map for all models across plots."""
    models = [
        "SCNN",
        "ResNet",
        "NT",
        "T",
        "SCNN (NT)",
        "SCNN (T)",
        "ResNet (NT)",
        "ResNet (T)",
    ]
    colors = [
        "#FF6B35",
        "#FF8C42",
        "#D62828",
        "#F77F00",
        "#FF9F2C",
        "#FCBF49",
        "#EAE2B7",
        "#C1121F",
    ]
    return {model: color for model, color in zip(models, colors)}


def get_x_limits(scores, padding=0.02, floor=0.78, ceiling=1.0):
    """Compute tighter x-axis limits around the plotted F1 scores."""
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    lower = max(floor, min_score - padding)
    upper = min(ceiling, max_score + padding)
    return lower, upper


def plot_separated_by_paradigm(df):
    """Generate separate plots for each paradigm using only test metrics."""
    color_map = build_model_color_map()
    paradigms = ["Classical", "Quantum", "SHF", "DHF", "TSHF"]

    for paradigm in paradigms:
        df_paradigm = df[df["Paradigm"] == paradigm].copy()
        architectures = df_paradigm["Architecture"].tolist()
        scores = df_paradigm["F1"].tolist()
        
        fig, ax = plt.subplots(figsize=(10, 6))

        y = np.arange(len(architectures))
        bar_height = 0.6

        for i, (arch, test_f1) in enumerate(zip(architectures, scores)):
            color = color_map.get(arch, "#FF6B35")

            ax.barh(
                i,
                test_f1,
                height=bar_height,
                color=color,
                alpha=0.85,
                edgecolor="black",
                linewidth=1.5,
            )

            if test_f1 > 0:
                ax.text(test_f1 + 0.003, i, f"{test_f1:.4f}", va="center", fontsize=29)

        ax.set_yticks(y)
        ax.set_yticklabels(architectures, fontsize=29)
        ax.set_xlabel("F1-score (Test)", fontsize=29)
        ax.set_xlim(*get_x_limits(scores))
        ax.tick_params(axis="x", labelsize=20)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        output_path = OUTPUT_DIR / f"breastmnist_f1_{paradigm.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved {paradigm} plot to {output_path}")


def plot_combined(df):
    """Generate a combined plot with all paradigms and architectures using test metrics."""
    color_map = build_model_color_map()

    df["Configuration"] = df["Paradigm"] + " | " + df["Architecture"]
    paradigm_order = ["TSHF", "DHF", "SHF", "Quantum", "Classical"]

    order = []
    for paradigm in paradigm_order:
        df_paradigm = df[df["Paradigm"] == paradigm]
        for arch in df_paradigm["Architecture"].tolist():
            order.append(f"{paradigm} | {arch}")

    fig, ax = plt.subplots(figsize=(16, 10))

    y = np.arange(len(order))
    bar_height = 0.6
    scores = []

    for i, config in enumerate(order):
        df_config = df[df["Configuration"] == config]
        test_f1 = df_config["F1"].values[0]
        scores.append(test_f1)

        arch = config.split(" | ")[1]
        color = color_map.get(arch, "#FF6B35")

        ax.barh(
            i,
            test_f1,
            height=bar_height,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=1,
        )

        if test_f1 > 0:
            ax.text(test_f1 + 0.003, i, f"{test_f1:.4f}", va="center", fontsize=29)

    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=29)
    ax.set_xlabel("F1-score (Test)", fontsize=29)
    ax.set_xlim(*get_x_limits(scores))
    ax.tick_params(axis="x", labelsize=20)

    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "breastmnist_f1_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved combined plot to {output_path}")


def main() -> None:
    df = pd.DataFrame(data)
    df = df[df["Split"] == "Test"].copy()

    print("=" * 70)
    print("Generating BreastMNIST Test F1-score plots...")
    print("=" * 70)

    plot_separated_by_paradigm(df)
    plot_combined(df)

    print("=" * 70)
    print("✓ All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
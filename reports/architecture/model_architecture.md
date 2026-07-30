# Model architecture specification

Everything below was read off the code and verified by instantiating the models
(`n_qubits=4`, `n_layers=1`, `patch_size=2`, `image_size=28`, `num_classes=2`,
`feature_dim=128`). Parameter counts are exact (`sum(p.numel())`).

Sources: [classical.py](../../src/models/main/classical.py),
[models.py](../../src/models/main/models.py), [layers.py](../../src/models/main/layers.py),
[fusion.py](../../src/models/fusion.py), [classification.py](../../src/models/classification.py),
[fusion_experiment.py](../../src/utils/fusion_experiment.py).

## Table 1 — Architecture of the hybrid fusion pipeline

| Block | Layer / setting | Configuration | Output shape | Params |
|---|---|---|---|---|
| **Input** | Preprocessing | grayscale, resize 28×28, `Normalize(µ=0.5, σ=0.5)`, batch 16 | 1×28×28 | – |
| **SCNN** (scratch) | Conv block 1 | `Conv(1→32, 3×3, s=2, p=1)` + BN + ReLU + `MaxPool(2,2)` | 32×7×7 | 384 |
| | Conv block 2 | `Conv(32→64, 3×3, s=2, p=1)` + BN + ReLU + `MaxPool(2,2)` | 64×2×2 | 18,624 |
| | Conv block 3 | `Conv(64→128, 3×3, s=2, p=1)` + BN + ReLU + `AdaptiveAvgPool(1×1)` | 128×1×1 | 74,112 |
| | Feature tap | flatten (`extract_features`) | 128 | – |
| | Baseline head | `Linear(128→2)` | 2 | 258 |
| | *Backbone total* | | | **93,378** |
| **ResNet-18** (ImageNet init) | Stem adaptation | `conv1` replaced by `Conv(1→64, 7×7, s=2, p=3, bias=False)`; 3-channel stem discarded | 64×14×14 | 3,136 |
| | layer1–layer4 | unchanged BasicBlocks [2,2,2,2], widths 64→128→256→512 | 512×1×1 (after GAP) | 11,166,912 |
| | Head adaptation | `fc` replaced by `Linear(512→2)` (was `Linear(512→1000)`) | 2 | 1,026 |
| | Feature tap | `children[:-1]` → flatten, evaluated under `torch.no_grad()` | 512 | – |
| | *Backbone total* | | | **11,171,266** |
| **Quantum circuit** (shared) | Patching | non-overlapping 2×2 patches, stride 2, zero-pad to grid | 196 patches/image | – |
| | Qubits / layers | 4 qubits (1 per patch pixel), 1 variational layer | – | – |
| | Simulator | PennyLane `lightning.qubit`, analytic (no shots), `interface="torch"`, CPU-only | – | – |
| | Measurement | ⟨Z_j⟩, j = 1…4 → 4 channels | 4×14×14 | – |
| | Post-processing | bilinear resize to fixed map + ReLU + flatten | 784 | – |
| **QC-fixed** (`QuantumModel`) | Encoding | `RY(π·x_j)` on wire j | – | – |
| | Ansatz | `RandomLayers(Θ)`, Θ ~ U(0, 2π), shape (1,4), **frozen** | – | 0 |
| | Head | `Linear(784→2)` | 2 | 1,570 |
| | *Branch total* | | | **1,570** (0 quantum) |
| **QC-trainable** (`ExtentedQuantumModel`) | Encoding | `AngleEmbedding(x, rotation="Y")` on 4 wires | – | – |
| | Ansatz | `RandomLayers(W)`, `weight_shapes={"weights": (1,4)}`, trained via `qml.qnn.TorchLayer` | – | 4 |
| | Circuit size | sampled instance: 9 gates, depth 4 — {RY 5, RX 2, RZ 1, CNOT 1} | – | – |
| | Head | `Linear(784→2)` | 2 | 1,570 |
| | *Branch total* | | | **1,574** (4 quantum) |
| **Projection heads** | Classical | `Linear(128→128)` (SCNN) / `Linear(512→128)` (ResNet-18), bias, no activation | 128 | 16,512 / 65,664 |
| | Quantum | `Linear(784→128)`, bias, no activation | 128 | 100,480 |
| | Normalisation | per-branch L2, `F.normalize(·, p=2, dim=1)` → unit sphere | 128 | 0 |
| | Instantiation | lazy: dimensions inferred at the first forward pass | – | – |
| **Fusion layer** (SHF / TSHF) | Temperature | γ = exp(log γ); modes `none` (γ=1, frozen), `fixed`, `trainable` | scalar | 1 |
| | Concatenation | [h_C ; γ·h_Q] | 256 | – |
| | Normalisation | `LayerNorm(256, elementwise_affine=False)` — no affine gain, so γ stays identifiable | 256 | 0 |
| | Activation | ReLU | 256 | – |
| | Classifier | `Linear(256→2)` | 2 | 514 |
| **Classification handler** (offline path) | MLP | `Linear(d→128)` → ReLU → `Dropout(0.5)` → `Linear(128→2)` | 2 | 16,770 (d=128) / 65,922 (d=512) / 100,738 (d=784) / 33,154 (d=256) |
| | Optimisation | Adam, lr 1e-3, cross-entropy | – | – |
| **Offline fusion operators** | Concatenation | [h_Q ; h_C] | 2D | 0 |
| | Sum / Hadamard | h_Q + h_C, h_Q ⊙ h_C | D | 0 |
| | Cosine gate | α = σ(cos(h_Q,h_C)); α h_Q + (1−α) h_C | D | 0 |
| | Self- / cross-attention | scaled dot-product over the 2-token sequence {h_Q, h_C}, mean-pooled | D | 0 |
| | Learned gate | `Linear(2D→max(64,D))` → ReLU → `Linear(→D)` → σ, element-wise gating | D | 2D·h + h·D |
| **Optimisation** | Optimiser | Adam, per-group learning rates | – | – |
| | Learning rates | classical 1e-4, quantum 1e-2, fusion 1e-3 (1e-1 in the TSHF sweep), γ 1e-2 | – | – |
| | Schedule | batch 16, 10 epochs, cross-entropy, 5 seeds per configuration | – | – |

## Table 2 — Parameter budget of the assembled fusion models

| Configuration | Classical | Quantum | Projections + fusion | Total trainable |
|---|---|---|---|---|
| SCNN + QC-trainable (TSHF) | 93,378 | 1,574 | 117,507 | **210,889** |
| ResNet-18 + QC-trainable (TSHF) | 11,171,266 | 1,574 | 166,659 | **11,337,929** |

The SHF variants are identical minus the single γ parameter (210,888 / 11,337,928).
"Projections + fusion" = classical head + quantum head + `LayerNorm` (0) + `Linear(256→2)` + γ.

## Analytical cost (from `reports/cost/cost_benchmark.txt`)

| Model | Params | MACs/image | Circuits/image | Gates | Depth |
|---|---|---|---|---|---|
| SCNN | 93,378 | 425,344 | 0 | – | – |
| ResNet-18 | 11,171,266 | 33,006,848 | 0 | – | – |
| QC-trainable | 1,574 | 1,568 | 196 | 9 | 4 |

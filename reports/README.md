# Reports

Multi-run evaluation outputs for each experiment. A multi-run trains the model
`N_RUNS` times from scratch (seeds `0..N_RUNS-1`), then reports **mean ± std**
and a **95% confidence interval** (Student's t-distribution) for every test
metric, plus per-run and aggregated confusion matrices, precision-recall curves
and ROC curves.

The machinery lives in `src/utils/multirun.py` and the experiment setup in
`src/utils/fusion_experiment.py`, so the notebooks and the `run-*.py` scripts
produce byte-identical layouts from the same config dict.

## Where results come from

| Folder                          | Source                                                | Model evaluated                                   |
|---------------------------------|-------------------------------------------------------|---------------------------------------------------|
| `<Dataset>/classical/`          | `run-classical.py` / `classical.ipynb`                | ResNet / ShallowCNN (end-to-end)                  |
| `<Dataset>/quantum/`            | `run-quantum.py` / `quantum.ipynb`                    | QuantumModel / ExtentedQuantumModel               |
| `shf/<classical>+<quantum>/`    | `run-shf.py`, `notebooks/online/classify.ipynb`       | `OnlineFusionModel`                               |
| `tshf/<gamma_tag>/<classical>+<quantum>/` | `run-tshf.py`, `notebooks/online/temperature.ipynb` | `TemperatureFusionModel` (γ)                |
| `offline_classify/`             | `notebooks/offline/classify.ipynb`                    | Classifier head over pre-extracted fused features |
| `online_classify/`              | legacy output of the old `classify.ipynb`             | `OnlineFusionModel`                               |
| `cost/`                         | `benchmark_cost.py`                                   | Runtime / computational cost of every model       |

`<gamma_tag>` encodes the temperature condition: `gamma_none`,
`gamma_fixed_0p5`, `gamma_trainable_init1`, …

## Running everything sequentially

```bash
python run-shf.py            # resnet+quantum, then shallow+quantum
python run-tshf.py           # both backbones x {trainable, none, fixed 0.5, fixed 2.0}

python run-shf.py --only shallow            # a single config
python run-tshf.py --only-gamma trainable   # a single gamma condition, both backbones
python run-shf.py --test-only               # no training: re-test the checkpoints on disk
```

Each config is a separate model with its own freeze policy (resnet is frozen and
only the fusion head is learned; shallow is fine-tuned end-to-end), so the runs
never share state — `--only` isolates one exactly as a standalone script would.

## Files produced per folder

```
<folder>/
├── multi_run_statistics_<tag>.txt            # human-readable summary report
├── multi_run_statistics_<tag>_testonly.txt   # same, from a --test-only pass
└── multi_run/
    ├── best_<tag>_run{k}.pth                 # best checkpoint per run
    ├── test_log_run{k}.txt                   # test_model() log per run
    ├── confusion_matrix_run{k}.png           # per-run confusion matrix
    ├── confusion_matrix_sum.png              # summed over all runs
    ├── confusion_matrix_mean_normalized.png  # row-normalized mean
    ├── pr_curve.png                          # per-run + mean PR curve, ±1 std band
    ├── roc_curve.png                         # per-run + mean ROC curve, ±1 std band
    ├── curve_data.npz                        # raw y_true / y_prob per run
    └── umap_run{k}.png                       # written by the notebook's UMAP cell
```

## Cost benchmark (`cost/`)

`benchmark_cost.py` answers the "what does the quantum simulation cost?"
question without touching the training pipeline. It writes
`cost_benchmark.{txt,csv,md}`, each holding two independent kinds of evidence:

- **Analytical**, hardware-independent: trainable parameters split into the
  classical and quantum branches, classical multiply-accumulates per image,
  circuit evaluations per image, and the expanded gate count and depth of one
  circuit (`qml.specs(..., level="device")`, so `RandomLayers` is counted as the
  gates it decomposes into rather than as a single operation).
- **Measured wall-clock**: inference forwards and full training steps
  (forward + backward + optimizer step), reported as the **median** over timed
  repeats, with mean/std/min alongside. The median is the headline because a
  shared machine has a long right tail — a single scheduler stall moves the mean
  by more than the effect being measured.

```bash
python benchmark_cost.py                            # all models, CPU and GPU
python benchmark_cost.py --only resnet shf-resnet   # a subset
python benchmark_cost.py --dataset breastmnist      # real batches/epoch for the epoch estimate
python benchmark_cost.py --budget 300               # more samples for the slow quantum models
```

`--budget` caps the seconds spent on each measurement (at least 3 samples are
always taken), which is what keeps the non-trainable `quantum` config — one
PennyLane circuit per patch in a Python loop — from dominating the run.

Two results matter for the paper: `lightning.qubit` simulates on CPU, so the
quantum branch costs the same with or without a GPU; and the non-trainable
circuit is fixed, so its features can be precomputed once and cached, turning a
per-epoch cost into a one-off preprocessing pass.

`curve_data.npz` holds the raw labels and positive-class probabilities per run,
so PR/ROC figures can be re-plotted or merged across experiments without
retraining.

The `multi_run_statistics_<tag>.txt` report contains: run configuration
(including the pre-trained checkpoints used to initialize each branch), a
per-run metrics table (loss, accuracy, precision, recall, F1, AUC, AP — plus γ
for the temperature model), summary statistics with 95% CI, the summed /
row-normalized confusion matrices in text form, and the paths of every figure.

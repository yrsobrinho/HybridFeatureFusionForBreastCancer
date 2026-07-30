"""Shared setup for the online-fusion experiments (SHF and TSHF).

The notebooks (`notebooks/online/classify.ipynb`, `notebooks/online/temperature.ipynb`)
and the scripts (`run-shf.py`, `run-tshf.py`) build the very same objects from
the very same config dicts, so a result produced by one is reproducible by the
other.

A config is a plain dict; `DEFAULTS` below fills in everything not overridden.
Setting `gamma_mode` turns the run into TSHF (TemperatureFusionModel); leaving it
out gives plain SHF (OnlineFusionModel).
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from src.models.fusion import OnlineFusionModel, TemperatureFusionModel
from src.models.main.classical import ClassicalModel, ShallowCNN
from src.models.main.models import ExtentedQuantumModel, QuantumModel
from src.utils.dataloader import BUSUCLMLoader, BreastMNISTLoader, INbreastLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Backbones pre-trained by run-classical.py / run-quantum.py, reused as fusion init.
# The quantum entries are keyed by `quantum_model`, so switching the branch also
# switches which checkpoint initializes it.
PRETRAINED = {
    "shallow": PROJECT_ROOT / "reports/BreastMNIST/classical/shallow/multi-run/best_shallow_no_encoder_run5.pth",
    "resnet": PROJECT_ROOT / "reports/BreastMNIST/classical/resnet/multi_run/best_resnet_run5.pth",
    "quantum": PROJECT_ROOT / "reports/BreastMNIST/quantum/non-trainable/multi_run/best_quantum_run4.pth",
    "extended_quantum": PROJECT_ROOT / "reports/BreastMNIST/quantum/trainable/multi_run/best_extended_quantum_run4.pth",
}

DEFAULTS = {
    "dataset": "BreastMNIST",
    "classical_model": "shallow",
    "quantum_model": "quantum",
    "classical_pretrained": None,
    "quantum_pretrained": None,
    "freeze_classical": False,
    "freeze_quantum": False,
    "batch_size": 16,
    "image_size": 28,
    "num_epochs": 10,
    "feature_dim": 128,
    "num_classes": 2,
    "lr_classical": 1e-4,
    "lr_quantum": 1e-2,
    "lr_fusion": 1e-3,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
    # TSHF only — absent means plain SHF.
    "gamma_mode": None,     # 'none' | 'fixed' | 'trainable'
    "init_gamma": 1.0,
    "lr_gamma": 1e-2,
}


def resolve(config):
    """Merge a partial config over DEFAULTS and fill in the derived fields."""
    cfg = {**DEFAULTS, **config}
    if cfg["classical_pretrained"] is None:
        cfg["classical_pretrained"] = str(PRETRAINED[cfg["classical_model"]])
    if cfg["quantum_pretrained"] is None:
        cfg["quantum_pretrained"] = str(PRETRAINED[cfg["quantum_model"]])
    cfg["gamma_tag"] = gamma_tag(cfg)
    cfg.setdefault("name", f"{cfg['classical_model']}+{cfg['quantum_model']}")
    return cfg


def gamma_tag(cfg):
    """Folder-safe tag for the gamma condition, or None for plain SHF."""
    mode = cfg.get("gamma_mode")
    if mode is None:
        return None
    if mode == "none":
        return "gamma_none"
    if mode == "fixed":
        return f"gamma_fixed_{cfg['init_gamma']:g}".replace(".", "p")
    if mode == "trainable":
        return f"gamma_trainable_init{cfg['init_gamma']:g}".replace(".", "p")
    raise ValueError(f"gamma_mode invalido: {mode!r}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloaders(cfg):
    """Return (train_loader, test_loader, val_loader) for cfg['dataset']."""
    dataset = cfg["dataset"]
    kwargs = {"batch_size": cfg["batch_size"], "image_size": cfg["image_size"]}
    if dataset == "BreastMNIST":
        loader = BreastMNISTLoader(**kwargs)
    elif dataset == "INbreast":
        loader = INbreastLoader(root_dir=str(PROJECT_ROOT / "src/data/INbreast/png"), **kwargs)
    elif dataset == "BUSUCLM":
        loader = BUSUCLMLoader(root_dir=str(PROJECT_ROOT / "src/data/BUS-UCLM/png"), **kwargs)
    else:
        raise ValueError(f"Dataset nao suportado: {dataset}")
    return loader.get_dataloaders()


def build_model(cfg, device):
    """Build the fusion model, load the pre-trained backbones, materialize lazy layers."""
    if cfg["classical_model"] == "shallow":
        classical = ShallowCNN(num_classes=cfg["num_classes"], input_channels=1)
    elif cfg["classical_model"] == "resnet":
        classical = ClassicalModel(num_classes=cfg["num_classes"])
    else:
        raise ValueError(f"Modelo classico nao suportado: {cfg['classical_model']}")

    if cfg["quantum_model"] == "quantum":
        rand_params = np.random.uniform(high=2 * np.pi, size=(cfg["n_qubits"], cfg["n_layers"]))
        side = cfg["image_size"] // cfg["patch_size"]
        quantum = QuantumModel(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"],
                               rand_params=rand_params, output_size=(side, side),
                               patch_size=cfg["patch_size"], num_classes=cfg["num_classes"])
    elif cfg["quantum_model"] == "extended_quantum":
        quantum = ExtentedQuantumModel(n_qubits=cfg["n_qubits"], n_layers=cfg["n_layers"],
                                       num_classes=cfg["num_classes"], patch_size=cfg["patch_size"],
                                       mode="2d", stride=cfg["patch_size"], use_parallel=True)
    else:
        raise ValueError(f"Modelo quantico nao suportado: {cfg['quantum_model']}")

    shared = dict(classical_model=classical, quantum_model=quantum,
                  feature_dim=cfg["feature_dim"], num_classes=cfg["num_classes"],
                  freeze_classical=cfg["freeze_classical"], freeze_quantum=cfg["freeze_quantum"])

    if cfg.get("gamma_mode") is None:
        model = OnlineFusionModel(**shared)
    else:
        model = TemperatureFusionModel(
            **shared,
            init_gamma=1.0 if cfg["gamma_mode"] == "none" else float(cfg["init_gamma"]),
            trainable_gamma=(cfg["gamma_mode"] == "trainable"))
        assert len(list(model.fusion_norm.parameters())) == 0, \
            "fusion_norm ainda tem ganho afim - ele pode cancelar o gamma e invalidar a ablacao"

    if cfg["classical_pretrained"] or cfg["quantum_pretrained"]:
        model.load_pretrained_weights(classical_path=cfg["classical_pretrained"],
                                      quantum_path=cfg["quantum_pretrained"], device=device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():  # materializes the lazy projection heads
        _ = model(torch.randn(1, 1, cfg["image_size"], cfg["image_size"]).to(device))
    model.train()
    return model


def build_optimizer(model, cfg):
    if cfg.get("gamma_mode") is None:
        groups = model.get_parameter_groups(lr_classical=cfg["lr_classical"],
                                            lr_quantum=cfg["lr_quantum"],
                                            lr_fusion=cfg["lr_fusion"])
    else:
        groups = model.get_parameter_groups(lr_classical=cfg["lr_classical"],
                                            lr_quantum=cfg["lr_quantum"],
                                            lr_fusion=cfg["lr_fusion"],
                                            lr_gamma=cfg["lr_gamma"])
    return optim.Adam(groups)


def run_tag(cfg):
    """Tag used in report filenames and per-run checkpoint names."""
    tag = f"fusion_{cfg['classical_model']}_{cfg['quantum_model']}"
    if cfg.get("gamma_tag"):
        tag += f"_{cfg['gamma_tag']}"
    return tag


def reports_dir(cfg, family):
    """`reports/<family>/[<gamma_tag>/]<classical>+<quantum>` under the project root."""
    parts = [str(PROJECT_ROOT), "reports", family]
    if cfg.get("gamma_tag"):
        parts.append(cfg["gamma_tag"])
    parts.append(f"{cfg['classical_model']}+{cfg['quantum_model']}")
    return os.path.join(*parts)


def make_build_fn(cfg, device):
    """Return `build_fn(seed) -> (model, optimizer, model_name)` for src.utils.multirun."""
    tag = run_tag(cfg)

    def build_fn(seed):
        model = build_model(cfg, device)
        return model, build_optimizer(model, cfg), f"{tag}_run{seed + 1}"

    return build_fn


def hyperparams(cfg):
    """The subset of cfg worth writing into the test logs."""
    keys = ["dataset", "classical_model", "quantum_model", "feature_dim", "batch_size",
            "lr_classical", "lr_quantum", "lr_fusion", "n_qubits", "n_layers", "patch_size",
            "freeze_classical", "freeze_quantum"]
    hp = {k: cfg[k] for k in keys}
    if cfg.get("gamma_mode") is not None:
        hp.update({k: cfg[k] for k in ["gamma_mode", "init_gamma", "lr_gamma"]})
    return hp


def report_meta(cfg):
    """Header block of the multi-run statistics report."""
    meta = {
        "Dataset": cfg["dataset"],
        "Classical Model": cfg["classical_model"],
        "Quantum Model": cfg["quantum_model"],
        "Feature Dim": cfg["feature_dim"],
        "Epochs per run": cfg["num_epochs"],
        "Batch Size": cfg["batch_size"],
        "LR (classical/quantum/fusion)": f"{cfg['lr_classical']}/{cfg['lr_quantum']}/{cfg['lr_fusion']}",
        "Freeze (classical/quantum)": f"{cfg['freeze_classical']}/{cfg['freeze_quantum']}",
        "Classical checkpoint": cfg["classical_pretrained"],
        "Quantum checkpoint": cfg["quantum_pretrained"],
    }
    if cfg.get("gamma_mode") is not None:
        meta["Gamma mode"] = cfg["gamma_mode"]
        meta["Gamma value (init/fixed)"] = cfg["init_gamma"]
        meta["LR gamma"] = cfg["lr_gamma"]
    return meta


def describe(cfg):
    line = (f"{cfg['classical_model']}+{cfg['quantum_model']} | {cfg['dataset']} | "
            f"epochs={cfg['num_epochs']} | freeze c/q={cfg['freeze_classical']}/{cfg['freeze_quantum']}")
    if cfg.get("gamma_mode") is not None:
        line += f" | gamma={cfg['gamma_mode']} (init {cfg['init_gamma']:g})"
    return line


__all__ = [
    "PROJECT_ROOT", "PRETRAINED", "DEFAULTS",
    "resolve", "gamma_tag", "get_device", "build_dataloaders", "build_model",
    "build_optimizer", "run_tag", "reports_dir", "make_build_fn",
    "hyperparams", "report_meta", "describe",
]

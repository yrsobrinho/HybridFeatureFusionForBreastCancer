import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import warnings

warnings.filterwarnings("ignore")

# Use Computer Modern through mathtext so it works without a LaTeX install.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 15

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.main.classical import ShallowCNN, ClassicalModel
from src.models.main.models import QuantumModel, ExtentedQuantumModel
from src.models.classification import ClassificationModelHandler
from src.models.fusion import OnlineFusionModel
from src.utils.dataloader import BreastMNISTLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base configurations
BASE_CONFIG = {
    "batch_size": 32,
    "num_classes": 2,
    "feature_dim": 128,
    "n_qubits": 4,
    "n_layers": 1,
    "patch_size": 2,
    "base_output_dir": str(project_root / "experiments" / "models"),
    "image_size": 28,
}

# Reproducibility for NT quantum random parameters.
# This helps keep ROC numbers stable across runs.
QUANTUM_RANDOM_SEED = 42
QUANTUM_BASIC_BEST_RAND_PARAMS_PATH = project_root / "notebooks" / "extra" / "best_model_rand_params.npy"

CLASSICAL_CONFIGS = [
    ("SCNN", "shallow"),
    ("ResNet-18", "resnet"),
]

QUANTUM_CONFIGS = [
    ("Quantum NT", "basic"),
    ("Quantum T", "T"),
]

FUSION_ONLINE_CONFIGS = [
    ("DHF + SCNN + Quantum", "shallow", "basic"),
    ("DHF + SCNN + T", "shallow", "T"),
    ("DHF + ResNet + Quantum", "resnet", "basic"),
    ("DHF + ResNet + T", "resnet", "T"),
]

FUSION_TEMPERATURE_CONFIGS = [
    ("TSHF + SCNN + Quantum", "shallow", "basic"),
    ("TSHF + SCNN + T", "shallow", "T"),
    ("TSHF + ResNet + Quantum", "resnet", "basic"),
    ("TSHF + ResNet + T", "resnet", "T"),
]

CLASSICAL_TYPE_TO_NAME = {classical_type: model_name for model_name, classical_type in CLASSICAL_CONFIGS}
QUANTUM_TYPE_TO_NAME = {quantum_type: model_name for model_name, quantum_type in QUANTUM_CONFIGS}
FUSION_ONLINE_MAP = {
    model_name: (classical_type, quantum_type)
    for model_name, classical_type, quantum_type in FUSION_ONLINE_CONFIGS
}
FUSION_TEMPERATURE_MAP = {
    model_name: (classical_type, quantum_type)
    for model_name, classical_type, quantum_type in FUSION_TEMPERATURE_CONFIGS
}

OFFLINE_SHF_VARIANTS = {
    "basic_shallow": {
        "feature_dir": "fusion/offline/{dataset}/concatenated/basic_shallow",
        "classifier_dir": "classification/offline/{dataset}/basic_shallow",
        "classifier_file": "best_classification_{dataset_lower}_basic_shallow.pth",
    },
    "basic_resnet": {
        "feature_dir": "fusion/offline/{dataset}/concatenated/basic_resnet",
        "classifier_dir": "classification/offline/{dataset}/basic_resnet",
        "classifier_file": "best_classification_{dataset_lower}_basic_resnet.pth",
    },
    "trainable_shallow": {
        "feature_dir": "fusion/offline/{dataset}/concatenated/trainable_shallow",
        "classifier_dir": "classification/offline/{dataset}/trainable_shallow",
        "classifier_file": "best_classification_{dataset_lower}_trainable_shallow.pth",
    },
    "trainable_resnet": {
        "feature_dir": "fusion/offline/{dataset}/concatenated/trainable_resnet",
        "classifier_dir": "classification/offline/{dataset}/trainable_resnet",
        "classifier_file": "best_classification_{dataset_lower}_trainable_resnet.pth",
    },
}

FUSION_NAME_ALIASES = {
    # Online fusion (DHF)
    "DHF (SCNN + NT)": "DHF + SCNN + Quantum",
    "DHF (SCNN + T)": "DHF + SCNN + T",
    "DHF (ResNet + NT)": "DHF + ResNet + Quantum",
    "DHF (ResNet + T)": "DHF + ResNet + T",
    # Temperature fusion (TSHF)
    "TSHF (SCNN + NT)": "TSHF + SCNN + Quantum",
    "TSHF (SCNN + T)": "TSHF + SCNN + T",
    "TSHF (ResNet + NT)": "TSHF + ResNet + Quantum",
    "TSHF (ResNet + T)": "TSHF + ResNet + T",
}

# Explicit checkpoint paths (relative to experiments/models).
# For fusion online and fusion with temperature, paths are manual-only.
BEST_MODEL_PATHS = {
    "classical": {
        "shallow": "classical/BreastMNIST/shallow/best_shallow.pth",
        "resnet": "classical/BreastMNIST/resnet/best_resnet.pth",
    },
    "quantum": {
        "basic": "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/notebooks/extra/best_model.pth",
        "T": "quantum/trainable/BreastMNIST/checkpoints/model_epoch15.pth",
    },
    "fusion_online": {
        "DHF (SCNN + NT)": "fusion/online/BreastMNIST/shallow_quantum/best_fusion_shallow_quantum.pth",
        "DHF (SCNN + T)": "fusion/online/BreastMNIST/shallow_extended_quantum/best_fusion_shallow_extended_quantum.pth",
        "DHF (ResNet + NT)": "fusion/online/BreastMNIST/resnet_quantum/best_fusion_resnet_quantum.pth",
        "DHF (ResNet + T)": "fusion/online/BreastMNIST/resnet_extended_quantum/best_fusion_resnet_extended_quantum.pth",
    },
    "fusion_online_temperature": {
        "TSHF (SCNN + NT)": "fusion/online/BreastMNIST/shallow_quantum/temperature/best_fusion_shallow_quantum_temperature.pth",
        "TSHF (SCNN + T)": "fusion/online/BreastMNIST/shallow_extended_quantum/temperature/best_fusion_shallow_extended_quantum_temperature.pth",
        "TSHF (ResNet + NT)": "fusion/online/BreastMNIST/resnet_quantum/temperature/best_fusion_resnet_quantum_temperature.pth",
        "TSHF (ResNet + T)": "fusion/online/BreastMNIST/resnet_extended_quantum/temperature/best_fusion_resnet_extended_quantum_temperature.pth",
    },
}


def _resolve_checkpoint_path(relative_path):
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(BASE_CONFIG["base_output_dir"], relative_path)


def get_classical_checkpoint_path(classical_type):
    custom_path = BEST_MODEL_PATHS["classical"].get(classical_type)
    if not custom_path:
        print(
            f"  ✗ Caminho nao configurado para classical '{classical_type}' em BEST_MODEL_PATHS['classical']."
        )
        return None
    return _resolve_checkpoint_path(custom_path)


def get_quantum_checkpoint_path(quantum_type):
    custom_path = BEST_MODEL_PATHS["quantum"].get(quantum_type)
    if not custom_path:
        print(
            f"  ✗ Caminho nao configurado para quantum '{quantum_type}' em BEST_MODEL_PATHS['quantum']."
        )
        return None
    return _resolve_checkpoint_path(custom_path)


def get_fusion_checkpoint_path(model_name, with_temperature=False):
    fusion_key = "fusion_online_temperature" if with_temperature else "fusion_online"
    custom_path = BEST_MODEL_PATHS[fusion_key].get(model_name)
    if not custom_path:
        print(
            f"  ✗ Caminho nao configurado para '{model_name}' em BEST_MODEL_PATHS['{fusion_key}']."
        )
        return None
    return _resolve_checkpoint_path(custom_path)


def resolve_fusion_config(model_name, with_temperature=False):
    """Resolve fusion config accepting both legacy and alias model names."""
    fusion_map = FUSION_TEMPERATURE_MAP if with_temperature else FUSION_ONLINE_MAP

    # Direct match first.
    config = fusion_map.get(model_name)
    if config:
        return config

    # Alias -> canonical name.
    canonical_name = FUSION_NAME_ALIASES.get(model_name)
    if canonical_name:
        return fusion_map.get(canonical_name)

    # Canonical -> alias name (useful when keys are inverted).
    for alias_name, canonical in FUSION_NAME_ALIASES.items():
        if canonical == model_name:
            return fusion_map.get(canonical) or fusion_map.get(alias_name)

    return None


def load_quantum_rand_params(checkpoint_path):
    """
    Prioridade de carregamento dos rand_params para quantum não treinável:
    1) arquivo fixo de melhor run (notebooks/extra/best_model_rand_params.npy);
    2) campo 'quantum_rand_params' dentro do checkpoint;
    3) arquivo sidecar '<checkpoint>_rand_params.npy';
    4) None (caller usa fallback com seed fixa).
    """
    if QUANTUM_BASIC_BEST_RAND_PARAMS_PATH.exists():
        try:
            rand_params = np.load(str(QUANTUM_BASIC_BEST_RAND_PARAMS_PATH))
            print(
                f"  ✓ rand_params carregado do arquivo fixo: {QUANTUM_BASIC_BEST_RAND_PARAMS_PATH}"
            )
            return np.asarray(rand_params, dtype=np.float32)
        except Exception as e:
            print(
                f"  ⚠ Falha ao carregar arquivo fixo de rand_params ({QUANTUM_BASIC_BEST_RAND_PARAMS_PATH}): {e}"
            )

    if not checkpoint_path:
        return None

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "quantum_rand_params" in checkpoint:
            rand_params = checkpoint["quantum_rand_params"]
            if rand_params is not None:
                print("  ✓ rand_params carregado do checkpoint (quantum_rand_params)")
                return np.asarray(rand_params, dtype=np.float32)
    except Exception:
        # Silencioso: sidecar/fallback ainda podem resolver.
        pass

    sidecar_path = str(Path(checkpoint_path).with_suffix("")) + "_rand_params.npy"
    if os.path.exists(sidecar_path):
        try:
            rand_params = np.load(sidecar_path)
            print(f"  ✓ rand_params carregado do sidecar: {sidecar_path}")
            return np.asarray(rand_params, dtype=np.float32)
        except Exception as e:
            print(f"  ⚠ Falha ao carregar sidecar rand_params ({sidecar_path}): {e}")

    print("  ⚠ rand_params não encontrado em checkpoint/sidecar; usando fallback com seed fixa.")
    return None


def load_checkpoint_state_dict(model, checkpoint_path, strict=False, ignored_keys=None):
    """Carrega checkpoint aceitando tanto state_dict puro quanto dicionário de treino."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Se for dict com model_state_dict, extrai apenas isso
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if ignored_keys:
            state_dict = {
                k: v for k, v in state_dict.items() if k not in set(ignored_keys)
            }

        load_result = model.load_state_dict(state_dict, strict=strict)
        if not strict:
            if load_result.missing_keys:
                print(f"  ⚠ Missing keys ignoradas: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"  ⚠ Unexpected keys ignoradas: {load_result.unexpected_keys}")
        return True
    except Exception as e:
        print(f"Erro ao carregar {checkpoint_path}: {e}")
        return False


def get_test_loader():
    """Retorna apenas o dataloader de teste do BreastMNIST."""
    loader = BreastMNISTLoader(
        batch_size=BASE_CONFIG["batch_size"],
        image_size=BASE_CONFIG["image_size"],
    )
    _, test_loader, _ = loader.get_dataloaders()
    return test_loader


def get_dataset_labels_path(dataset):
    return project_root / "experiments" / "labels" / dataset / "test_labels.npy"


def get_offline_shf_paths(dataset, variant):
    variant_config = OFFLINE_SHF_VARIANTS.get(variant)
    if not variant_config:
        print(f"  ✗ Variante offline SHF nao configurada: {variant}")
        return None, None, None

    feature_dir = project_root / "experiments" / "models" / variant_config["feature_dir"].format(dataset=dataset)
    classifier_dir = project_root / "experiments" / "models" / variant_config["classifier_dir"].format(dataset=dataset)
    classifier_path = classifier_dir / variant_config["classifier_file"].format(dataset_lower=dataset.lower())
    features_path = feature_dir / "test_features_concat.npy"
    labels_path = get_dataset_labels_path(dataset)

    return str(features_path), str(labels_path), str(classifier_path)


def warmup_lazy_layers(model, dataloader):
    """Inicializa camadas criadas dinamicamente (ex.: fc em modelos quânticos)."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            _ = model(imgs)
            break


def get_predictions_and_labels(model, dataloader):
    """Extrai predições e labels do modelo."""
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(imgs)
            
            # Extrai probabilidades da classe positiva
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)[:, 1]
            else:
                probs = torch.sigmoid(outputs).squeeze()
            
            all_preds.append(probs.detach().cpu().numpy().reshape(-1))
            all_labels.append(labels.detach().cpu().numpy().reshape(-1))
    
    return np.concatenate(all_preds), np.concatenate(all_labels)


def evaluate_model_for_roc(
    model,
    checkpoint_path,
    test_loader,
    strict_load=False,
    use_warmup=False,
    ignored_checkpoint_keys=None,
):
    """
    Pipeline único para ROC:
    1) carregar modelo;
    2) rodar no conjunto de teste;
    3) obter probabilidades e labels.
    """
    if checkpoint_path is None:
        return None

    if not os.path.exists(checkpoint_path):
        print(f"  ✗ Checkpoint não encontrado: {checkpoint_path}")
        return None

    if use_warmup:
        # Alguns modelos possuem camadas lazy que precisam existir antes do load.
        warmup_lazy_layers(model, test_loader)

    if not load_checkpoint_state_dict(
        model,
        checkpoint_path,
        strict=strict_load,
        ignored_keys=ignored_checkpoint_keys,
    ):
        return None

    preds, labels = get_predictions_and_labels(model, test_loader)
    return preds, labels


def make_roc_result(preds, labels, paradigm):
    """Calcula FPR/TPR/AUC a partir das probabilidades."""
    fpr, tpr, _ = roc_curve(labels, preds)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc(fpr, tpr),
        "paradigm": paradigm,
    }


def create_classical_model(classical_type):
    """Cria modelo clássico."""
    if classical_type == "shallow":
        return ShallowCNN(num_classes=2, input_channels=1)
    elif classical_type == "resnet":
        return ClassicalModel(num_classes=2)
    else:
        raise ValueError(f"Classical type {classical_type} not supported")


def create_quantum_model(quantum_type, rand_params_override=None):
    """Cria modelo quantum."""
    if quantum_type == "basic":
        if rand_params_override is not None:
            rand_params = rand_params_override
        else:
            np.random.seed(QUANTUM_RANDOM_SEED)
            rand_params = np.random.uniform(high=2*np.pi, size=(4, 1))
        return QuantumModel(
            n_qubits=4,
            n_layers=1,
            rand_params=rand_params,
            output_size=(14, 14),
            patch_size=2,
            num_classes=2
        )
    elif quantum_type == "T":
        return ExtentedQuantumModel(
            n_qubits=4,
            n_layers=1,
            num_classes=2,
            patch_size=2,
            mode="2d",
            stride=None,
            use_parallel=True
        )
    else:
        raise ValueError(f"Quantum type {quantum_type} not supported")


def process_classical_models():
    """Processa todos os modelos clássicos."""
    results = {}
    test_loader = get_test_loader()
    
    configured_classical = BEST_MODEL_PATHS.get("classical", {})
    for classical_type in configured_classical:
        model_name = CLASSICAL_TYPE_TO_NAME.get(classical_type, classical_type)
        print(f"Processando Classical: {model_name}...")
        
        try:
            model = create_classical_model(classical_type)
            checkpoint_path = get_classical_checkpoint_path(classical_type)
            evaluated = evaluate_model_for_roc(
                model,
                checkpoint_path,
                test_loader,
                strict_load=False,
                use_warmup=False,
            )
            if evaluated is None:
                continue

            preds, labels = evaluated
            results[f"Classical - {model_name}"] = make_roc_result(preds, labels, "Classical")
            print(f"  ✓ AUC: {results[f'Classical - {model_name}']['auc']:.4f}")
        except Exception as e:
            print(f"  ✗ Erro ao processar {model_name}: {e}")
    
    return results


def process_quantum_models():
    """Processa todos os modelos quantum."""
    results = {}
    test_loader = get_test_loader()
    
    configured_quantum = BEST_MODEL_PATHS.get("quantum", {})
    for quantum_type in configured_quantum:
        model_name = QUANTUM_TYPE_TO_NAME.get(quantum_type, quantum_type)
        print(f"Processando Quantum: {model_name}...")
        
        try:
            checkpoint_path = get_quantum_checkpoint_path(quantum_type)

            rand_params_override = None
            if quantum_type == "basic" and checkpoint_path and os.path.exists(checkpoint_path):
                rand_params_override = load_quantum_rand_params(checkpoint_path)

            model = create_quantum_model(
                quantum_type,
                rand_params_override=rand_params_override,
            )
            evaluated = evaluate_model_for_roc(
                model,
                checkpoint_path,
                test_loader,
                strict_load=True,
                use_warmup=True,
            )
            if evaluated is None:
                continue

            preds, labels = evaluated
            results[f"Quantum - {model_name}"] = make_roc_result(preds, labels, "Quantum")
            print(f"  ✓ AUC: {results[f'Quantum - {model_name}']['auc']:.4f}")
        except Exception as e:
            print(f"  ✗ Erro ao processar {model_name}: {e}")
    
    return results


def process_fusion_models():
    """Processa todos os modelos de fusão."""
    results = {}
    test_loader = get_test_loader()

    fusion_models_to_load = []

    for model_name in BEST_MODEL_PATHS.get("fusion_online", {}):
        config = resolve_fusion_config(model_name, with_temperature=False)
        if not config:
            print(f"  ✗ Configuração não encontrada para fusion_online: {model_name}")
            continue
        classical_type, quantum_type = config
        fusion_models_to_load.append((model_name, classical_type, quantum_type, False))

    for model_name in BEST_MODEL_PATHS.get("fusion_online_temperature", {}):
        config = resolve_fusion_config(model_name, with_temperature=True)
        if not config:
            print(f"  ✗ Configuração não encontrada para fusion_online_temperature: {model_name}")
            continue
        classical_type, quantum_type = config
        fusion_models_to_load.append((model_name, classical_type, quantum_type, True))

    for model_name, classical_type, quantum_type, with_temperature in fusion_models_to_load:
        print(f"Processando Fusion: {model_name}...")
        
        try:
            classical_model = create_classical_model(classical_type)
            quantum_model = create_quantum_model(quantum_type)
            
            fusion_model = OnlineFusionModel(
                classical_model=classical_model,
                quantum_model=quantum_model,
                feature_dim=128,
                num_classes=2
            )
            
            checkpoint_path = get_fusion_checkpoint_path(model_name, with_temperature=with_temperature)
            evaluated = evaluate_model_for_roc(
                fusion_model,
                checkpoint_path,
                test_loader,
                strict_load=True,
                use_warmup=True,
                ignored_checkpoint_keys=["log_gamma"] if with_temperature else None,
            )
            if evaluated is None:
                continue

            preds, labels = evaluated
            results[f"Fusion - {model_name}"] = make_roc_result(preds, labels, "Fusion")
            print(f"  ✓ AUC: {results[f'Fusion - {model_name}']['auc']:.4f}")
        except Exception as e:
            print(f"  ✗ Erro ao processar {model_name}: {str(e)[:100]}")
    
    return results


def process_offline_shf_models():
    """Processa os modelos SHF offline baseados em features concatenadas."""
    results = {}
    dataset_name = "BreastMNIST"

    offline_variants = [
        ("SHF (SCNN + NT)", "basic_shallow"),
        ("SHF (ResNet + NT)", "basic_resnet"),
        ("SHF (SCNN + T)", "trainable_shallow"),
        ("SHF (ResNet + T)", "trainable_resnet"),
    ]

    for model_name, variant in offline_variants:
        print(f"Processando SHF: {model_name}...")
        try:
            features_path, labels_path, classifier_path = get_offline_shf_paths(dataset_name, variant)
            if not os.path.exists(features_path):
                print(f"  ✗ Features nao encontradas: {features_path}")
                continue
            if not os.path.exists(labels_path):
                print(f"  ✗ Labels nao encontrados: {labels_path}")
                continue
            if not os.path.exists(classifier_path):
                print(f"  ✗ Checkpoint nao encontrado: {classifier_path}")
                continue

            features = np.load(features_path)
            labels = np.load(labels_path).squeeze()

            if features.ndim > 2:
                features = features.reshape(features.shape[0], -1)

            classifier = ClassificationModelHandler(
                device=device,
                num_classes=2,
                sequential_hyperparams={"input_dim": features.shape[1], "learning_rate": 1e-3},
            )
            classifier.model.load_state_dict(torch.load(classifier_path, map_location=device), strict=True)
            classifier.model.eval()

            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(features),
                torch.LongTensor(labels),
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=BASE_CONFIG["batch_size"], shuffle=False)

            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_features, batch_labels in dataloader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    outputs = classifier.model(batch_features)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    all_preds.append(probs.detach().cpu().numpy().reshape(-1))
                    all_labels.append(batch_labels.detach().cpu().numpy().reshape(-1))

            preds = np.concatenate(all_preds)
            labels = np.concatenate(all_labels)
            results[f"{model_name}"] = make_roc_result(preds, labels, "SHF")
            print(f"  ✓ AUC: {results[f'{model_name}']['auc']:.4f}")
        except Exception as e:
            print(f"  ✗ Erro ao processar {model_name}: {e}")

    return results


def build_model_color_map(all_results):
    """Cria um mapa global de cores aleatorias para os modelos."""
    rng = np.random.default_rng(42)
    model_keys = sorted(all_results.keys())
    color_map = {}

    for key in model_keys:
        # Gera cor RGB aleatoria evitando tons muito claros para manter contraste.
        rgb = rng.integers(30, 226, size=3)
        color_map[key] = "#{:02X}{:02X}{:02X}".format(*rgb)

    return color_map


def plot_roc_curves(all_results, model_color_map):
    """Plota todas as curvas ROC."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    
    paradigms = ["Classical", "Quantum", "Fusion", "SHF"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    
    paradigm_results = {p: {} for p in paradigms}
    for key, result in all_results.items():
        paradigm = result["paradigm"]
        paradigm_results[paradigm][key] = result
    
    for idx, paradigm in enumerate(paradigms):
        ax = axes[idx]
        results = paradigm_results[paradigm]
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier", alpha=0.5)
        
        # Plot curves
        for i, (key, result) in enumerate(results.items()):
            color = model_color_map.get(key, "#D7301F")
            marker = markers[i % len(markers)]
            label = key.replace(f"{paradigm} - ", "")
            ax.plot(
                result["fpr"],
                result["tpr"],
                color=color,
                lw=2.5,
                marker=marker,
                markevery=0.15,
                ms=5,
                label=f"{label} (AUC = {result['auc']:.3f})",
                alpha=0.9,
            )
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.legend(loc="lower right", fontsize=22, framealpha=0.95)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_aspect("equal")
    
    plt.tight_layout()
    
    # Salvar figura
    output_dir = Path(BASE_CONFIG["base_output_dir"]) / ".." / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "breastmnist_roc_curves.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"\n✓ Figura salva em: {output_path}")
    plt.close()


def plot_combined_roc(all_results, model_color_map):
    """Plota todas as curvas ROC em um único gráfico."""
    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier", alpha=0.5)
    
    linestyle_map = {"Classical": "-", "Quantum": "--", "Fusion": "-.", "SHF": ":"}
    marker_map = {"Classical": "o", "Quantum": "s", "Fusion": "^", "SHF": "D"}
    
    for key, result in sorted(all_results.items()):
        paradigm = result["paradigm"]

        color = model_color_map.get(key, "#D7301F")
        linestyle = linestyle_map[paradigm]
        label = key
        
        ax.plot(
            result["fpr"],
            result["tpr"],
            color=color,
            lw=2.5,
            linestyle=linestyle,
            marker=marker_map[paradigm],
            markevery=0.14,
            ms=5,
            label=f"{label} (AUC = {result['auc']:.3f})",
            alpha=0.9,
        )
    
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=20)
    ax.set_ylabel("True Positive Rate", fontsize=20)
    ax.legend(loc="lower right", fontsize=20, framealpha=0.95, ncol=2)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_aspect("equal")
    
    plt.tight_layout()
    
    # Salvar figura
    output_dir = Path(BASE_CONFIG["base_output_dir"]) / ".." / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "breastmnist_roc_curves_combined.png"
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    print(f"✓ Figura combinada salva em: {output_path}")
    plt.close()


def plot_roc_separated(all_results, model_color_map):
    """
    Gera 5 graficos separados por categoria:
    1. Modelos Quantum
    2. Modelos Classical
    3. Modelos DHF (Fusion Online)
    4. Modelos TSHF (Fusion com Temperatura)
    5. Modelos SHF
    """
    output_dir = Path(BASE_CONFIG["base_output_dir"]) / ".." / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    marker_cycle = ["o", "s", "^", "D", "v", "P", "X"]
    
    # Separar resultados por categoria
    quantum_results = {}
    classical_results = {}
    dhf_results = {}
    tshf_results = {}
    shf_results = {}
    
    for key, result in all_results.items():
        if "Quantum" in key:
            quantum_results[key] = result
        elif "Classical" in key:
            classical_results[key] = result
        elif "DHF" in key:
            dhf_results[key] = result
        elif "TSHF" in key:
            tshf_results[key] = result
        elif "SHF" in key:
            shf_results[key] = result
    
    categories = [
        ("Quantum Models", quantum_results, "breastmnist_roc_quantum.png"),
        ("Classical Models", classical_results, "breastmnist_roc_classical.png"),
        ("DHF Models", dhf_results, "breastmnist_roc_dhf.png"),
        ("TSHF Models", tshf_results, "breastmnist_roc_tshf.png"),
        ("SHF Models", shf_results, "breastmnist_roc_shf.png"),
    ]
    
    for title, results_dict, filename in categories:
        if not results_dict:
            continue
            
        fig, ax = plt.subplots(figsize=(10.5, 8.5))
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Random Classifier", alpha=0.4)
        
        # Plot curves com cores aleatorias definidas no mapa global
        for i, (key, result) in enumerate(results_dict.items()):
            color = model_color_map.get(key, "#D7301F")
            marker = marker_cycle[i % len(marker_cycle)]
            label = key.replace("Quantum - ", "").replace("Classical - ", "").replace("Fusion - ", "")
            ax.plot(
                result["fpr"],
                result["tpr"],
                color=color,
                lw=3,
                marker=marker,
                markevery=0.12,
                ms=5,
                label=f"{label} (AUC = {result['auc']:.3f})",
                alpha=0.9,
            )
        
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)
        ax.legend(loc="lower right", fontsize=20, framealpha=0.97, edgecolor="black", fancybox=True)
        ax.tick_params(axis="both", labelsize=20)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.7)
        ax.set_aspect("equal")
        
        # Estilo mais visual
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        
        plt.tight_layout()
        
        # Salvar figura
        output_path = output_dir / filename
        plt.savefig(str(output_path), dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ {title} salvo em: {output_path}")
        plt.close()


def main():
    print("="*70)
    print("ROC CURVE GENERATION - BreastMNIST")
    print("="*70)
    print(f"Device: {device}\n")
    
    all_results = {}
    
    print("\n[1/3] CLASSICAL MODELS")
    print("-" * 70)
    all_results.update(process_classical_models())
    
    print("\n[2/3] QUANTUM MODELS")
    print("-" * 70)
    all_results.update(process_quantum_models())
    
    print("\n[3/3] FUSION MODELS")
    print("-" * 70)
    all_results.update(process_fusion_models())

    print("\n[3/3b] SHF MODELS")
    print("-" * 70)
    all_results.update(process_offline_shf_models())

    print("\n" + "="*70)
    print(f"Total de modelos processados: {len(all_results)}")
    print("="*70)
    
    if all_results:
        print("\nGerando gráficos...")
        model_color_map = build_model_color_map(all_results)
        plot_roc_curves(all_results, model_color_map)
        plot_combined_roc(all_results, model_color_map)
        plot_roc_separated(all_results, model_color_map)
        print("\n✓ Processamento concluído!")
    else:
        print("\n✗ Nenhum modelo foi processado com sucesso.")


if __name__ == "__main__":
    main()

"""Auto-exported from 09_v1_1_PaddyDoctor_Calibration_Reliability_TemperatureScaling_FIXED.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 09
# ## Calibración probabilística y confiabilidad bajo cambio de dominio
# 
# ### Objetivo científico
# Evaluar si los modelos no solo pierden capacidad de clasificación al cambiar de dominio/cultivar, sino también **confiabilidad probabilística**.
# 
# Se integran P1, P4 y P5.
# 
# ### Preguntas
# 1. ¿Aumentan **ECE, MCE, NLL y Brier score** bajo domain shift?
# 2. ¿Los modelos se vuelven **sobreconfiados** fuera de dominio?
# 3. ¿El **temperature scaling**, ajustado únicamente con validation, mejora la calibración?
# 4. ¿Una temperatura aprendida en el dominio fuente se transfiere al dominio objetivo?
# 5. ¿La pérdida de calibración persiste en P5 después del control exacto de edad?
# 
# ### Regla anti-leakage
# El parámetro de temperatura se ajusta **exclusivamente en validation**.  
# El test **nunca** se utiliza para seleccionar o ajustar la temperatura.
# 
# ### Carga computacional
# No hay entrenamiento. Se cargan los checkpoints ya guardados y se hace una inferencia de validation:
# 
# - P1: 12 checkpoints.
# - P4: 24 checkpoints.
# - P5: 24 checkpoints.
# - Total: **60 inferencias de validation**.
# 
# > Recomendado: **GPU L4**. El notebook puede funcionar en CPU, pero será más lento.

# %% [markdown] Cell 2
# ## 0. Montaje robusto de Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path
import time
import os

try:
    drive.mount("/content/drive", force_remount=False)
except Exception:
    drive.mount("/content/drive", force_remount=True)

MYDRIVE = Path("/content/drive/MyDrive")
PROJECT = MYDRIVE / "PADDY_DOCTOR_Q1"

DATA_DIR = PROJECT / "data"
SPLITS_DIR = DATA_DIR / "splits"
CONFIG_DIR = PROJECT / "config"

RESULTS_DIR = PROJECT / "results"
MODELS_DIR = PROJECT / "models"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"

P1_RESULTS = RESULTS_DIR / "05_P1_30ep_noES"
P4_RESULTS = RESULTS_DIR / "06_P4_cross_cultivar_30ep_noES"
P5_RESULTS = RESULTS_DIR / "07_P5_exact_age_matched_30ep_noES"

P1_MODELS = MODELS_DIR / "05_P1_30ep_noES"
P4_MODELS = MODELS_DIR / "06_P4_cross_cultivar_30ep_noES"
P5_MODELS = MODELS_DIR / "07_P5_exact_age_matched_30ep_noES"

RUN_ROOT = RESULTS_DIR / "09_calibration_reliability"
FIGURE_ROOT = FIGURES_DIR / "09_calibration_reliability"
CACHE_ROOT = RUN_ROOT / "validation_cache"

for folder in [RUN_ROOT, FIGURE_ROOT, CACHE_ROOT, EXPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("✅ DRIVE CONECTADO")
print("=" * 80)
print("Proyecto:", PROJECT)
print("Resultados N09:", RUN_ROOT)

# %% [markdown] Cell 4
# ## 1. Dependencias, Kaggle y dispositivo

# %% Cell 5
import sys
import subprocess
import shutil

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.28"],
    check=True
)

import json
import gc
import math
import random
import warnings
import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
import kagglehub

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax

DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"
LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"

if DRIVE_KAGGLE_JSON.exists():
    shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
    os.chmod(LOCAL_KAGGLE_JSON, 0o600)
    print("✅ kaggle.json restaurado desde Drive.")
else:
    print("⚠️ No encontré config/kaggle.json.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dispositivo:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CPU activo. Recomiendo L4 para acelerar las 60 inferencias.")

print("timm:", timm.__version__)
print("torch:", torch.__version__)

# %% [markdown] Cell 6
# ## 2. Configuración científica

# %% Cell 7
MODELS = [
    "resnet50",
    "convnext_tiny",
    "vit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
]

MODEL_SHORT = {
    "resnet50": "ResNet-50",
    "convnext_tiny": "ConvNeXt-Tiny",
    "vit_small_patch16_224": "ViT-S/16",
    "swin_tiny_patch4_window7_224": "Swin-Tiny",
}

SEEDS = [2026, 2027, 2028]

IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 64 if DEVICE.type == "cuda" else 16
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

N_BINS = 15
EPS = 1e-12

P4_PROTOCOL_MAP = {
    ("ADT45", "OTHER"): "P4A_ADT45_to_OTHER_matched",
    ("ADT45", "ADT45"): "P4B_ADT45_to_ADT45_control",
    ("OTHER", "ADT45"): "P4C_OTHER_to_ADT45_matched",
    ("OTHER", "OTHER"): "P4D_OTHER_to_OTHER_control",
}

P5_PROTOCOL_MAP = {
    ("ADT45", "OTHER"): "P5A_ADT45_to_OTHER_age_matched",
    ("ADT45", "ADT45"): "P5B_ADT45_to_ADT45_age_control",
    ("OTHER", "ADT45"): "P5C_OTHER_to_ADT45_age_matched",
    ("OTHER", "OTHER"): "P5D_OTHER_to_OTHER_age_control",
}

config = {
    "notebook": "09_calibration_reliability",
    "n_checkpoints_validation_inference": 60,
    "temperature_scaling_fit": "validation_only",
    "temperature_objective": "multiclass_NLL",
    "ece_bins": N_BINS,
    "batch_size": BATCH_SIZE,
    "device": str(DEVICE),
    "models": MODELS,
    "seeds": SEEDS,
}

with open(RUN_ROOT / "09_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))

# %% [markdown] Cell 8
# ## 3. Verificar resultados, modelos y splits anteriores

# %% Cell 9
expected_complete = {
    "P1": (P1_RESULTS, 12),
    "P4": (P4_RESULTS, 24),
    "P5": (P5_RESULTS, 24),
}

for exp, (root, expected_n) in expected_complete.items():
    if not root.exists():
        raise FileNotFoundError(f"❌ Falta {exp}: {root}")

    completed = len(list(root.rglob("COMPLETE.json")))
    print(f"{exp}: COMPLETE.json {completed}/{expected_n}")

    if completed != expected_n:
        raise RuntimeError(f"❌ {exp} no está completo.")

checkpoint_counts = {
    "P1": len(list(P1_MODELS.rglob("best_model.pt"))),
    "P4": len(list(P4_MODELS.rglob("best_model.pt"))),
    "P5": len(list(P5_MODELS.rglob("best_model.pt"))),
}

print("\nCheckpoints:", checkpoint_counts)

if checkpoint_counts != {"P1": 12, "P4": 24, "P5": 24}:
    raise RuntimeError("❌ No están presentes los 60 checkpoints esperados.")

required_splits = [
    SPLITS_DIR / "P1_group_clean.csv",
    SPLITS_DIR / "P4A_ADT45_to_OTHER_matched.csv",
    SPLITS_DIR / "P4B_ADT45_to_ADT45_control.csv",
    SPLITS_DIR / "P4C_OTHER_to_ADT45_matched.csv",
    SPLITS_DIR / "P4D_OTHER_to_OTHER_control.csv",
    SPLITS_DIR / "P5A_ADT45_to_OTHER_age_matched.csv",
    SPLITS_DIR / "P5B_ADT45_to_ADT45_age_control.csv",
    SPLITS_DIR / "P5C_OTHER_to_ADT45_age_matched.csv",
    SPLITS_DIR / "P5D_OTHER_to_OTHER_age_control.csv",
]

missing_splits = [str(p) for p in required_splits if not p.exists()]

if missing_splits:
    raise FileNotFoundError(f"❌ Faltan splits: {missing_splits}")

print("✅ Checkpoints y splits verificados.")

# %% [markdown] Cell 10
# ## 4. Recuperar Paddy Doctor y preparar validation

# %% Cell 11
DATASET_PATH = Path(
    "/root/.cache/kagglehub/competitions/paddy-disease-classification"
)

TRAIN_IMAGES = DATASET_PATH / "train_images"

if not TRAIN_IMAGES.exists():
    print("⬇️ Descargando Paddy Doctor...")
    DATASET_PATH = Path(
        kagglehub.competition_download("paddy-disease-classification")
    )
    TRAIN_IMAGES = DATASET_PATH / "train_images"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

image_lookup = {
    p.name: p
    for p in TRAIN_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
}

print("Imágenes indexadas:", len(image_lookup))

p1_split = pd.read_csv(SPLITS_DIR / "P1_group_clean.csv")
P1_VAL = p1_split[p1_split["split"] == "validation"].copy()

p4a = pd.read_csv(SPLITS_DIR / "P4A_ADT45_to_OTHER_matched.csv")
p4c = pd.read_csv(SPLITS_DIR / "P4C_OTHER_to_ADT45_matched.csv")

P4_VAL = {
    "ADT45": p4a[p4a["split"] == "validation"].copy(),
    "OTHER": p4c[p4c["split"] == "validation"].copy(),
}

p5a = pd.read_csv(SPLITS_DIR / "P5A_ADT45_to_OTHER_age_matched.csv")
p5c = pd.read_csv(SPLITS_DIR / "P5C_OTHER_to_ADT45_age_matched.csv")

P5_VAL = {
    "ADT45": p5a[p5a["split"] == "validation"].copy(),
    "OTHER": p5c[p5c["split"] == "validation"].copy(),
}

if len(P1_VAL) != 1476:
    raise RuntimeError(f"❌ P1 validation inesperado: {len(P1_VAL)}")

for d in ["ADT45", "OTHER"]:
    if len(P4_VAL[d]) != 616:
        raise RuntimeError(f"❌ P4 {d} validation inesperado: {len(P4_VAL[d])}")
    if len(P5_VAL[d]) != 100:
        raise RuntimeError(f"❌ P5 {d} validation inesperado: {len(P5_VAL[d])}")
    if P5_VAL[d]["label"].nunique() != 5:
        raise RuntimeError(f"❌ P5 {d} no tiene 5 clases.")

print("=" * 80)
print("✅ VALIDATION LISTO")
print("=" * 80)
print("P1:", len(P1_VAL))
print("P4 ADT45 / OTHER:", len(P4_VAL["ADT45"]), len(P4_VAL["OTHER"]))
print("P5 ADT45 / OTHER:", len(P5_VAL["ADT45"]), len(P5_VAL["OTHER"]))

# %% [markdown] Cell 12
# ## 5. Dataset y transformación de evaluación

# %% Cell 13
eval_transform = transforms.Compose([
    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class PaddyEvalDataset(Dataset):
    def __init__(self, dataframe, image_lookup, class_to_idx):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_lookup = image_lookup
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_id = row["image_id"]
        label_name = row["label"]
        path = self.image_lookup[image_id]

        with Image.open(path) as img:
            image = img.convert("RGB")

        image = eval_transform(image)
        target = self.class_to_idx[label_name]

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "image_id": image_id,
        }

def make_val_loader(dataframe, class_to_idx):
    missing = [
        image_id for image_id in dataframe["image_id"]
        if image_id not in image_lookup
    ]

    if missing:
        raise RuntimeError(f"❌ Faltan imágenes físicas. Ejemplos: {missing[:5]}")

    ds = PaddyEvalDataset(
        dataframe=dataframe,
        image_lookup=image_lookup,
        class_to_idx=class_to_idx,
    )

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )

print("✅ Dataset de evaluación definido.")

# %% [markdown] Cell 14
# ## 6. Métricas de calibración

# %% Cell 15
def multiclass_nll(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0)

    return float(
        -np.mean(
            np.log(p[np.arange(len(y_true)), y_true])
        )
    )

def multiclass_brier(y_true, probabilities, n_classes):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    one_hot = np.eye(n_classes, dtype=float)[y_true]

    return float(
        np.mean(np.sum((p - one_hot) ** 2, axis=1))
    )

def calibration_bins(y_true, probabilities, n_bins=N_BINS):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    confidence = p.max(axis=1)
    prediction = p.argmax(axis=1)
    correct = (prediction == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []

    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]

        if i == 0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence > lo) & (confidence <= hi)

        n = int(mask.sum())

        if n == 0:
            rows.append({
                "bin": i + 1,
                "bin_lower": lo,
                "bin_upper": hi,
                "n": 0,
                "accuracy": np.nan,
                "mean_confidence": np.nan,
                "abs_gap": np.nan,
            })
            continue

        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())

        rows.append({
            "bin": i + 1,
            "bin_lower": lo,
            "bin_upper": hi,
            "n": n,
            "accuracy": acc,
            "mean_confidence": conf,
            "abs_gap": abs(acc - conf),
        })

    return pd.DataFrame(rows)

def ece_mce(y_true, probabilities, n_bins=N_BINS):
    bins = calibration_bins(y_true, probabilities, n_bins=n_bins)
    total = bins["n"].sum()
    nonempty = bins[bins["n"] > 0].copy()

    if total == 0:
        return np.nan, np.nan

    ece = float(
        np.sum((nonempty["n"] / total) * nonempty["abs_gap"])
    )
    mce = float(nonempty["abs_gap"].max())

    return ece, mce

def calibration_metrics(y_true, probabilities):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    pred = p.argmax(axis=1)
    conf = p.max(axis=1)

    accuracy = float(np.mean(pred == y_true))
    mean_confidence = float(np.mean(conf))
    ece, mce = ece_mce(y_true, p, n_bins=N_BINS)

    return {
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "signed_confidence_minus_accuracy": mean_confidence - accuracy,
        "ece_15": ece,
        "mce_15": mce,
        "nll": multiclass_nll(y_true, p),
        "brier": multiclass_brier(y_true, p, n_classes=p.shape[1]),
    }

print("✅ ECE, MCE, NLL, Brier y overconfidence definidos.")

# %% [markdown] Cell 16
# ## 7. Temperature scaling sin fuga de información

# %% Cell 17
def nll_from_logits(y_true, logits, temperature=1.0):
    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=float)

    scaled = logits / float(temperature)
    log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)

    return float(
        -np.mean(log_probs[np.arange(len(y_true)), y_true])
    )

def fit_temperature(y_true, logits):
    y_true = np.asarray(y_true, dtype=int)
    logits = np.asarray(logits, dtype=float)

    def objective(log_temperature):
        temperature = float(np.exp(log_temperature))
        return nll_from_logits(
            y_true=y_true,
            logits=logits,
            temperature=temperature,
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
        options={"xatol": 1e-5},
    )

    if not result.success:
        raise RuntimeError(
            f"❌ Falló temperature scaling: {result.message}"
        )

    temperature = float(np.exp(result.x))

    return temperature, float(result.fun)

def logits_to_probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=float)
    return softmax(logits / float(temperature), axis=1)

def probabilities_temperature_scale(probabilities, temperature):
    # Si p = softmax(z), entonces log(p) = z - constante.
    # softmax(log(p)/T) es equivalente a softmax(z/T).
    p = np.clip(
        np.asarray(probabilities, dtype=float),
        EPS,
        1.0
    )

    pseudo_logits = np.log(p)

    return softmax(
        pseudo_logits / float(temperature),
        axis=1
    )

print("✅ Temperature scaling definido.")
print("✅ T se ajustará SOLO con validation.")

# %% [markdown] Cell 18
# ## 8. Inferencia de validation y ajuste de temperatura para los 60 checkpoints

# %% Cell 19
@torch.no_grad()
def infer_logits(model, loader):
    model.eval()

    all_logits = []
    all_targets = []
    all_ids = []

    use_amp = DEVICE.type == "cuda"

    for batch in loader:
        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        if use_amp:
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16
            ):
                logits = model(images)
        else:
            logits = model(images)

        all_logits.append(
            logits.float().cpu().numpy()
        )

        all_targets.append(
            batch["target"].numpy()
        )

        all_ids.extend(
            list(batch["image_id"])
        )

    return (
        np.concatenate(all_targets),
        np.concatenate(all_logits),
        np.asarray(all_ids, dtype=str),
    )

def checkpoint_and_validation(experiment, model_key, seed, train_domain=None):
    if experiment == "P1":
        ckpt = (
            P1_MODELS /
            f"{model_key}__seed_{seed}" /
            "best_model.pt"
        )
        val_df = P1_VAL
        run_id = f"P1__{model_key}__seed_{seed}"

    elif experiment == "P4":
        ckpt = (
            P4_MODELS /
            f"train_{train_domain}__{model_key}__seed_{seed}" /
            "best_model.pt"
        )
        val_df = P4_VAL[train_domain]
        run_id = (
            f"P4__train_{train_domain}__"
            f"{model_key}__seed_{seed}"
        )

    elif experiment == "P5":
        ckpt = (
            P5_MODELS /
            f"train_{train_domain}__{model_key}__seed_{seed}" /
            "best_model.pt"
        )
        val_df = P5_VAL[train_domain]
        run_id = (
            f"P5__train_{train_domain}__"
            f"{model_key}__seed_{seed}"
        )

    else:
        raise ValueError(experiment)

    return ckpt, val_df, run_id

def process_validation_checkpoint(
    experiment,
    model_key,
    seed,
    train_domain=None,
):
    ckpt_path, val_df, run_id = checkpoint_and_validation(
        experiment=experiment,
        model_key=model_key,
        seed=seed,
        train_domain=train_domain,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cache_npz = CACHE_ROOT / f"{run_id}.npz"
    cache_json = CACHE_ROOT / f"{run_id}.json"

    if cache_npz.exists() and cache_json.exists():
        arrays = np.load(cache_npz, allow_pickle=False)

        with open(cache_json, "r", encoding="utf-8") as f:
            summary = json.load(f)

        return summary, arrays["y_true"], arrays["logits"], arrays["image_ids"]

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False
    )

    class_to_idx = {
        str(k): int(v)
        for k, v in checkpoint["class_to_idx"].items()
    }

    idx_to_class = {
        int(v): str(k)
        for k, v in class_to_idx.items()
    }

    class_names = [
        idx_to_class[i]
        for i in range(len(idx_to_class))
    ]

    timm_name = checkpoint.get(
        "timm_name",
        model_key
    )

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    loader = make_val_loader(
        dataframe=val_df,
        class_to_idx=class_to_idx,
    )

    y_true, logits, image_ids = infer_logits(
        model,
        loader
    )

    raw_probs = logits_to_probabilities(
        logits,
        temperature=1.0
    )

    temperature, calibrated_val_nll = fit_temperature(
        y_true,
        logits
    )

    calibrated_probs = logits_to_probabilities(
        logits,
        temperature=temperature
    )

    raw_metrics = calibration_metrics(
        y_true,
        raw_probs
    )

    calibrated_metrics = calibration_metrics(
        y_true,
        calibrated_probs
    )

    summary = {
        "experiment": experiment,
        "model_key": model_key,
        "seed": int(seed),
        "train_domain": (
            "MIXED"
            if experiment == "P1"
            else train_domain
        ),
        "run_id": run_id,
        "checkpoint": str(ckpt_path),
        "timm_name": timm_name,
        "n_validation": int(len(y_true)),
        "n_classes": int(len(class_names)),
        "class_names_json": json.dumps(class_names),
        "temperature": float(temperature),
        "validation_raw_nll": float(raw_metrics["nll"]),
        "validation_calibrated_nll": float(calibrated_val_nll),
        "validation_raw_ece": float(raw_metrics["ece_15"]),
        "validation_calibrated_ece": float(calibrated_metrics["ece_15"]),
    }

    np.savez_compressed(
        cache_npz,
        y_true=y_true.astype(np.int16),
        logits=logits.astype(np.float32),
        image_ids=image_ids,
    )

    with open(
        cache_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    del model
    del loader
    del checkpoint

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return summary, y_true, logits, image_ids


validation_rows = []

tasks = []

for model_key in MODELS:
    for seed in SEEDS:
        tasks.append(
            ("P1", model_key, seed, None)
        )

for experiment in ["P4", "P5"]:
    for train_domain in ["ADT45", "OTHER"]:
        for model_key in MODELS:
            for seed in SEEDS:
                tasks.append(
                    (
                        experiment,
                        model_key,
                        seed,
                        train_domain
                    )
                )

print("Total checkpoints:", len(tasks))

start_validation = time.time()

for i, (
    experiment,
    model_key,
    seed,
    train_domain
) in enumerate(tasks, start=1):

    label = (
        f"{experiment} | "
        f"{train_domain or 'MIXED'} | "
        f"{MODEL_SHORT[model_key]} | "
        f"seed {seed}"
    )

    print(
        f"[{i:02d}/{len(tasks)}] {label}"
    )

    summary, _, _, _ = process_validation_checkpoint(
        experiment=experiment,
        model_key=model_key,
        seed=seed,
        train_domain=train_domain,
    )

    validation_rows.append(summary)

validation_parameters = pd.DataFrame(
    validation_rows
)

validation_parameters.to_csv(
    RUN_ROOT /
    "09_temperature_parameters_validation.csv",
    index=False
)

elapsed_minutes = (
    time.time() - start_validation
) / 60.0

print("=" * 80)
print("✅ 60 VALIDATION INFERENCES COMPLETADAS")
print("=" * 80)
print(f"Tiempo: {elapsed_minutes:.1f} min")

display(
    validation_parameters[
        [
            "experiment",
            "train_domain",
            "model_key",
            "seed",
            "temperature",
            "validation_raw_nll",
            "validation_calibrated_nll",
            "validation_raw_ece",
            "validation_calibrated_ece",
        ]
    ].round(5)
)

# %% [markdown] Cell 20
# ## 9. Aplicar la temperatura de validation a los tests ya guardados

# %% Cell 21
def probability_columns_from_class_names(df, class_names):
    cols = [
        f"prob__{class_name}"
        for class_name in class_names
    ]

    missing = [
        col
        for col in cols
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"❌ Faltan columnas de probabilidad: {missing}"
        )

    return cols

def run_lookup(
    experiment,
    model_key,
    seed,
    train_domain
):
    mask = (
        (validation_parameters["experiment"] == experiment)
        & (validation_parameters["model_key"] == model_key)
        & (validation_parameters["seed"] == seed)
    )

    if experiment != "P1":
        mask &= (
            validation_parameters["train_domain"] == train_domain
        )

    rows = validation_parameters[mask]

    if len(rows) != 1:
        raise RuntimeError(
            f"❌ Parámetro T no único: {experiment}, "
            f"{model_key}, {seed}, {train_domain}"
        )

    return rows.iloc[0]

def test_prediction_path(
    experiment,
    model_key,
    seed,
    train_domain=None,
    eval_domain=None,
):
    if experiment == "P1":
        return (
            P1_RESULTS /
            f"{model_key}__seed_{seed}" /
            "test_predictions.csv"
        )

    root = (
        P4_RESULTS
        if experiment == "P4"
        else P5_RESULTS
    )

    return (
        root /
        f"train_{train_domain}__{model_key}__seed_{seed}" /
        f"eval__{eval_domain}" /
        "test_predictions.csv"
    )

test_metric_rows = []
bin_rows = []
per_image_rows = []

test_tasks = []

for model_key in MODELS:
    for seed in SEEDS:
        test_tasks.append(
            ("P1", model_key, seed, "MIXED", "MIXED", "P1_IID")
        )

for experiment, protocol_map in [
    ("P4", P4_PROTOCOL_MAP),
    ("P5", P5_PROTOCOL_MAP),
]:
    for train_domain in ["ADT45", "OTHER"]:
        for eval_domain in ["ADT45", "OTHER"]:
            for model_key in MODELS:
                for seed in SEEDS:
                    test_tasks.append(
                        (
                            experiment,
                            model_key,
                            seed,
                            train_domain,
                            eval_domain,
                            protocol_map[
                                (train_domain, eval_domain)
                            ],
                        )
                    )

print("Evaluaciones test:", len(test_tasks))

for i, (
    experiment,
    model_key,
    seed,
    train_domain,
    eval_domain,
    protocol,
) in enumerate(test_tasks, start=1):

    lookup_domain = (
        None
        if experiment == "P1"
        else train_domain
    )

    row = run_lookup(
        experiment=experiment,
        model_key=model_key,
        seed=seed,
        train_domain=lookup_domain,
    )

    temperature = float(
        row["temperature"]
    )

    class_names = json.loads(
        row["class_names_json"]
    )

    path = test_prediction_path(
        experiment=experiment,
        model_key=model_key,
        seed=seed,
        train_domain=None if experiment == "P1" else train_domain,
        eval_domain=None if experiment == "P1" else eval_domain,
    )

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    prob_cols = probability_columns_from_class_names(
        df,
        class_names
    )

    raw_probs = df[prob_cols].to_numpy(
        dtype=float
    )

    raw_probs = raw_probs / raw_probs.sum(
        axis=1,
        keepdims=True
    )

    calibrated_probs = probabilities_temperature_scale(
        raw_probs,
        temperature=temperature
    )

    y_true = df["y_true_idx"].to_numpy(
        dtype=int
    )

    raw_metrics = calibration_metrics(
        y_true,
        raw_probs
    )

    calibrated_metrics = calibration_metrics(
        y_true,
        calibrated_probs
    )

    common = {
        "experiment": experiment,
        "protocol": protocol,
        "model_key": model_key,
        "seed": int(seed),
        "train_domain": train_domain,
        "eval_domain": eval_domain,
        "eval_type": (
            "IID"
            if experiment == "P1"
            else (
                "control"
                if train_domain == eval_domain
                else "cross"
            )
        ),
        "temperature": temperature,
        "n_test": int(len(df)),
    }

    metric_row = common.copy()

    for k, v in raw_metrics.items():
        metric_row[f"raw_{k}"] = float(v)

    for k, v in calibrated_metrics.items():
        metric_row[f"ts_{k}"] = float(v)

    metric_row["delta_ece_ts_minus_raw"] = (
        calibrated_metrics["ece_15"]
        - raw_metrics["ece_15"]
    )

    metric_row["delta_nll_ts_minus_raw"] = (
        calibrated_metrics["nll"]
        - raw_metrics["nll"]
    )

    metric_row["delta_brier_ts_minus_raw"] = (
        calibrated_metrics["brier"]
        - raw_metrics["brier"]
    )

    test_metric_rows.append(metric_row)

    # Reliability bins.
    for method, probs in [
        ("raw", raw_probs),
        ("temperature_scaled", calibrated_probs),
    ]:
        bins = calibration_bins(
            y_true,
            probs,
            n_bins=N_BINS
        )

        for _, b in bins.iterrows():
            bin_rows.append({
                **common,
                "method": method,
                "bin": int(b["bin"]),
                "bin_lower": float(b["bin_lower"]),
                "bin_upper": float(b["bin_upper"]),
                "n": int(b["n"]),
                "accuracy": (
                    np.nan
                    if pd.isna(b["accuracy"])
                    else float(b["accuracy"])
                ),
                "mean_confidence": (
                    np.nan
                    if pd.isna(b["mean_confidence"])
                    else float(b["mean_confidence"])
                ),
                "abs_gap": (
                    np.nan
                    if pd.isna(b["abs_gap"])
                    else float(b["abs_gap"])
                ),
            })

    raw_true_prob = raw_probs[
        np.arange(len(y_true)),
        y_true
    ]

    ts_true_prob = calibrated_probs[
        np.arange(len(y_true)),
        y_true
    ]

    raw_pred = raw_probs.argmax(axis=1)
    ts_pred = calibrated_probs.argmax(axis=1)

    for j in range(len(df)):
        per_image_rows.append({
            **common,
            "image_id": str(df.iloc[j]["image_id"]),
            "y_true_idx": int(y_true[j]),
            "raw_correct": int(raw_pred[j] == y_true[j]),
            "ts_correct": int(ts_pred[j] == y_true[j]),
            "raw_confidence": float(raw_probs[j].max()),
            "ts_confidence": float(calibrated_probs[j].max()),
            "raw_nll": float(-np.log(max(raw_true_prob[j], EPS))),
            "ts_nll": float(-np.log(max(ts_true_prob[j], EPS))),
        })

    if i % 12 == 0 or i == len(test_tasks):
        print(
            f"✅ Test {i}/{len(test_tasks)} procesados"
        )


test_calibration = pd.DataFrame(
    test_metric_rows
)

reliability_bins_all = pd.DataFrame(
    bin_rows
)

per_image_calibration = pd.DataFrame(
    per_image_rows
)

test_calibration.to_csv(
    RUN_ROOT /
    "09_test_calibration_by_run.csv",
    index=False
)

reliability_bins_all.to_csv(
    RUN_ROOT /
    "09_reliability_bins_by_run.csv",
    index=False
)

per_image_calibration.to_csv(
    RUN_ROOT /
    "09_test_calibration_per_image.csv",
    index=False
)

print("✅ Test calibrado sin usar test para ajustar T.")
display(
    test_calibration.head().round(5)
)

# %% [markdown] Cell 22
# ## 10. Resumen agregado: raw vs temperature scaling

# %% Cell 23
aggregate = (
    test_calibration
    .groupby(
        [
            "experiment",
            "protocol",
            "model_key",
            "train_domain",
            "eval_domain",
            "eval_type",
        ],
        dropna=False
    )
    .agg(
        n_seeds=("seed", "nunique"),
        temperature_mean=("temperature", "mean"),
        temperature_sd=("temperature", "std"),
        raw_accuracy_mean=("raw_accuracy", "mean"),
        raw_ece_mean=("raw_ece_15", "mean"),
        raw_ece_sd=("raw_ece_15", "std"),
        ts_ece_mean=("ts_ece_15", "mean"),
        ts_ece_sd=("ts_ece_15", "std"),
        raw_mce_mean=("raw_mce_15", "mean"),
        ts_mce_mean=("ts_mce_15", "mean"),
        raw_nll_mean=("raw_nll", "mean"),
        raw_nll_sd=("raw_nll", "std"),
        ts_nll_mean=("ts_nll", "mean"),
        ts_nll_sd=("ts_nll", "std"),
        raw_brier_mean=("raw_brier", "mean"),
        ts_brier_mean=("ts_brier", "mean"),
        raw_confidence_mean=("raw_mean_confidence", "mean"),
        ts_confidence_mean=("ts_mean_confidence", "mean"),
        raw_overconfidence_mean=(
            "raw_signed_confidence_minus_accuracy",
            "mean"
        ),
        ts_overconfidence_mean=(
            "ts_signed_confidence_minus_accuracy",
            "mean"
        ),
        delta_ece_mean=("delta_ece_ts_minus_raw", "mean"),
        delta_nll_mean=("delta_nll_ts_minus_raw", "mean"),
        delta_brier_mean=("delta_brier_ts_minus_raw", "mean"),
    )
    .reset_index()
)

aggregate["model"] = (
    aggregate["model_key"]
    .map(MODEL_SHORT)
)

aggregate.to_csv(
    RUN_ROOT /
    "09_calibration_aggregate_summary.csv",
    index=False
)

display(
    aggregate[
        [
            "experiment",
            "protocol",
            "model",
            "raw_ece_mean",
            "ts_ece_mean",
            "raw_nll_mean",
            "ts_nll_mean",
            "raw_overconfidence_mean",
            "ts_overconfidence_mean",
        ]
    ].round(5)
)

# %% [markdown] Cell 24
# ## 11. Deterioro de calibración bajo domain shift con target fijo

# %% Cell 25
fixed_target_rows = []

for experiment in ["P4", "P5"]:
    exp_df = test_calibration[
        test_calibration["experiment"] == experiment
    ].copy()

    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        for model_key in MODELS:
            for seed in SEEDS:
                control = exp_df[
                    (exp_df["model_key"] == model_key)
                    & (exp_df["seed"] == seed)
                    & (exp_df["train_domain"] == control_source)
                    & (exp_df["eval_domain"] == target_domain)
                ]

                cross = exp_df[
                    (exp_df["model_key"] == model_key)
                    & (exp_df["seed"] == seed)
                    & (exp_df["train_domain"] == cross_source)
                    & (exp_df["eval_domain"] == target_domain)
                ]

                if len(control) != 1 or len(cross) != 1:
                    raise RuntimeError(
                        f"❌ Fixed-target no único: "
                        f"{experiment}, {target_domain}, {model_key}, {seed}"
                    )

                c = control.iloc[0]
                x = cross.iloc[0]

                fixed_target_rows.append({
                    "experiment": experiment,
                    "target_domain": target_domain,
                    "model_key": model_key,
                    "seed": int(seed),

                    "raw_ece_control": c["raw_ece_15"],
                    "raw_ece_cross": x["raw_ece_15"],
                    "raw_ece_shift_increase": (
                        x["raw_ece_15"]
                        - c["raw_ece_15"]
                    ),

                    "ts_ece_control": c["ts_ece_15"],
                    "ts_ece_cross": x["ts_ece_15"],
                    "ts_ece_shift_increase": (
                        x["ts_ece_15"]
                        - c["ts_ece_15"]
                    ),

                    "raw_nll_control": c["raw_nll"],
                    "raw_nll_cross": x["raw_nll"],
                    "raw_nll_shift_increase": (
                        x["raw_nll"]
                        - c["raw_nll"]
                    ),

                    "ts_nll_control": c["ts_nll"],
                    "ts_nll_cross": x["ts_nll"],
                    "ts_nll_shift_increase": (
                        x["ts_nll"]
                        - c["ts_nll"]
                    ),

                    "raw_overconfidence_control": (
                        c["raw_signed_confidence_minus_accuracy"]
                    ),
                    "raw_overconfidence_cross": (
                        x["raw_signed_confidence_minus_accuracy"]
                    ),
                })


fixed_target = pd.DataFrame(
    fixed_target_rows
)

fixed_target_summary = (
    fixed_target
    .groupby(
        [
            "experiment",
            "target_domain",
            "model_key",
        ]
    )
    .agg(
        raw_ece_control_mean=("raw_ece_control", "mean"),
        raw_ece_cross_mean=("raw_ece_cross", "mean"),
        raw_ece_shift_increase_mean=("raw_ece_shift_increase", "mean"),
        ts_ece_control_mean=("ts_ece_control", "mean"),
        ts_ece_cross_mean=("ts_ece_cross", "mean"),
        ts_ece_shift_increase_mean=("ts_ece_shift_increase", "mean"),
        raw_nll_shift_increase_mean=("raw_nll_shift_increase", "mean"),
        ts_nll_shift_increase_mean=("ts_nll_shift_increase", "mean"),
        raw_overconfidence_control_mean=("raw_overconfidence_control", "mean"),
        raw_overconfidence_cross_mean=("raw_overconfidence_cross", "mean"),
    )
    .reset_index()
)

fixed_target.to_csv(
    RUN_ROOT /
    "09_fixed_target_calibration_shift_by_seed.csv",
    index=False
)

fixed_target_summary.to_csv(
    RUN_ROOT /
    "09_fixed_target_calibration_shift_summary.csv",
    index=False
)

display(
    fixed_target_summary.round(5)
)

# %% [markdown] Cell 26
# ## 12. ¿Temperature scaling se transfiere fuera de dominio?

# %% Cell 27
transfer_summary = (
    test_calibration
    .assign(
        ece_improved=lambda d: d["ts_ece_15"] < d["raw_ece_15"],
        nll_improved=lambda d: d["ts_nll"] < d["raw_nll"],
        brier_improved=lambda d: d["ts_brier"] < d["raw_brier"],
    )
    .groupby(
        [
            "experiment",
            "eval_type",
            "model_key",
        ]
    )
    .agg(
        n_runs=("seed", "count"),
        fraction_ece_improved=("ece_improved", "mean"),
        fraction_nll_improved=("nll_improved", "mean"),
        fraction_brier_improved=("brier_improved", "mean"),
        mean_delta_ece=("delta_ece_ts_minus_raw", "mean"),
        mean_delta_nll=("delta_nll_ts_minus_raw", "mean"),
        mean_delta_brier=("delta_brier_ts_minus_raw", "mean"),
    )
    .reset_index()
)

transfer_summary["model"] = (
    transfer_summary["model_key"]
    .map(MODEL_SHORT)
)

transfer_summary.to_csv(
    RUN_ROOT /
    "09_temperature_scaling_transfer_summary.csv",
    index=False
)

display(
    transfer_summary[
        [
            "experiment",
            "eval_type",
            "model",
            "fraction_ece_improved",
            "fraction_nll_improved",
            "mean_delta_ece",
            "mean_delta_nll",
        ]
    ].round(5)
)

# %% [markdown] Cell 28
# ## 13. Resumen de temperaturas aprendidas en validation

# %% Cell 29
temperature_summary = (
    validation_parameters
    .groupby(
        [
            "experiment",
            "train_domain",
            "model_key",
        ]
    )
    .agg(
        temperature_mean=("temperature", "mean"),
        temperature_sd=("temperature", "std"),
        validation_raw_nll_mean=("validation_raw_nll", "mean"),
        validation_calibrated_nll_mean=(
            "validation_calibrated_nll",
            "mean"
        ),
        validation_raw_ece_mean=("validation_raw_ece", "mean"),
        validation_calibrated_ece_mean=(
            "validation_calibrated_ece",
            "mean"
        ),
    )
    .reset_index()
)

temperature_summary["model"] = (
    temperature_summary["model_key"]
    .map(MODEL_SHORT)
)

temperature_summary.to_csv(
    RUN_ROOT /
    "09_temperature_validation_summary.csv",
    index=False
)

display(
    temperature_summary.round(5)
)

# %% [markdown] Cell 30
# ## 14. Reliability diagrams
# 
# Se generan diagramas para:
# 
# - P1 IID;
# - P4 cross-domain en ambas direcciones;
# - P5 exact-age-matched cross-domain en ambas direcciones.
# 
# Cada curva se obtiene agregando los bins de las tres semillas mediante ponderación por número de imágenes en cada bin.
# 
# > Estas figuras son descriptivas. Las semillas representan realizaciones repetidas del entrenamiento, no datasets independientes.

# %% Cell 31
SELECTED_PROTOCOLS = [
    "P1_IID",
    "P4A_ADT45_to_OTHER_matched",
    "P4C_OTHER_to_ADT45_matched",
    "P5A_ADT45_to_OTHER_age_matched",
    "P5C_OTHER_to_ADT45_age_matched",
]

def weighted_bin_average(group):
    g = group[group["n"] > 0].copy()

    if len(g) == 0:
        return pd.Series({
            "n_total": 0,
            "accuracy": np.nan,
            "mean_confidence": np.nan,
        })

    weights = g["n"].to_numpy(dtype=float)

    return pd.Series({
        "n_total": int(weights.sum()),
        "accuracy": float(
            np.average(
                g["accuracy"].to_numpy(dtype=float),
                weights=weights
            )
        ),
        "mean_confidence": float(
            np.average(
                g["mean_confidence"].to_numpy(dtype=float),
                weights=weights
            )
        ),
    })

reliability_aggregate = (
    reliability_bins_all[
        reliability_bins_all["protocol"].isin(
            SELECTED_PROTOCOLS
        )
    ]
    .groupby(
        [
            "experiment",
            "protocol",
            "model_key",
            "method",
            "bin",
            "bin_lower",
            "bin_upper",
        ],
        dropna=False
    )
    .apply(weighted_bin_average)
    .reset_index()
)

reliability_aggregate.to_csv(
    RUN_ROOT /
    "09_reliability_bins_aggregate_selected.csv",
    index=False
)

for protocol in SELECTED_PROTOCOLS:
    for model_key in MODELS:
        plot_df = reliability_aggregate[
            (reliability_aggregate["protocol"] == protocol)
            & (reliability_aggregate["model_key"] == model_key)
        ].copy()

        if len(plot_df) == 0:
            continue

        fig = plt.figure(figsize=(6.5, 6.0))

        for method in [
            "raw",
            "temperature_scaled",
        ]:
            sub = plot_df[
                plot_df["method"] == method
            ].sort_values("bin")

            sub = sub[
                sub["n_total"] > 0
            ]

            plt.plot(
                sub["mean_confidence"],
                sub["accuracy"],
                marker="o",
                label=method,
            )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="perfect calibration",
        )

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("Mean predicted confidence")
        plt.ylabel("Observed accuracy")
        plt.title(
            f"{MODEL_SHORT[model_key]} | {protocol}"
        )
        plt.legend()
        plt.tight_layout()

        safe_protocol = protocol.replace("/", "_")

        plt.savefig(
            FIGURE_ROOT /
            f"09_reliability__{safe_protocol}__{model_key}.png",
            dpi=250,
            bbox_inches="tight"
        )

        plt.close(fig)

print(
    "✅ Reliability diagrams guardados:",
    len(list(FIGURE_ROOT.glob("09_reliability__*.png")))
)

# %% [markdown] Cell 32
# ## 15. Figuras resumen de calibración

# %% Cell 33
# 1) ECE raw bajo los nueve protocolos.
plot_df = aggregate.copy()

protocol_order = [
    "P1_IID",
    "P4B_ADT45_to_ADT45_control",
    "P4A_ADT45_to_OTHER_matched",
    "P4D_OTHER_to_OTHER_control",
    "P4C_OTHER_to_ADT45_matched",
    "P5B_ADT45_to_ADT45_age_control",
    "P5A_ADT45_to_OTHER_age_matched",
    "P5D_OTHER_to_OTHER_age_control",
    "P5C_OTHER_to_ADT45_age_matched",
]

for model_key in MODELS:
    sub = (
        plot_df[
            plot_df["model_key"] == model_key
        ]
        .set_index("protocol")
        .reindex(protocol_order)
        .reset_index()
    )

    sub = sub[
        sub["raw_ece_mean"].notna()
    ]

    fig = plt.figure(figsize=(11, 5))

    x = np.arange(len(sub))

    plt.plot(
        x,
        sub["raw_ece_mean"],
        marker="o",
        label="raw ECE",
    )

    plt.plot(
        x,
        sub["ts_ece_mean"],
        marker="o",
        label="temperature-scaled ECE",
    )

    plt.xticks(
        x,
        sub["protocol"],
        rotation=55,
        ha="right"
    )

    plt.ylabel("ECE (15 bins)")
    plt.xlabel("Protocol")
    plt.title(
        f"Calibration across protocols — {MODEL_SHORT[model_key]}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT /
        f"09_ECE_across_protocols__{model_key}.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)


# 2) Incremento de ECE por domain shift, target fijo.
for experiment in ["P4", "P5"]:
    sub = fixed_target_summary[
        fixed_target_summary["experiment"] == experiment
    ].copy()

    sub["label"] = (
        sub["model_key"].map(MODEL_SHORT)
        + " | target "
        + sub["target_domain"]
    )

    fig = plt.figure(figsize=(10, 6))

    y = np.arange(len(sub))

    plt.plot(
        sub["raw_ece_shift_increase_mean"],
        y,
        marker="o",
        linestyle="",
        label="raw",
    )

    plt.plot(
        sub["ts_ece_shift_increase_mean"],
        y,
        marker="x",
        linestyle="",
        label="temperature-scaled",
    )

    plt.axvline(
        0.0,
        linestyle="--"
    )

    plt.yticks(
        y,
        sub["label"]
    )

    plt.xlabel(
        "ECE increase: cross-domain − matched-domain control"
    )
    plt.ylabel("Architecture / target")
    plt.title(
        f"{experiment} — calibration degradation under domain shift"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_ROOT /
        f"09_{experiment}_fixed_target_ECE_shift.png",
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(fig)

print("✅ Figuras resumen guardadas en:", FIGURE_ROOT)

# %% [markdown] Cell 34
# ## 16. Tabla de hallazgos clave
# 
# La tabla sintetiza:
# 
# - calibración IID;
# - calibración control in-domain;
# - calibración cross-domain;
# - efecto del temperature scaling;
# - overconfidence.
# 
# No se declara que temperature scaling “resuelva” el domain shift: se evalúa si corrige parcialmente la confianza probabilística sin cambiar la clase predicha.

# %% Cell 35
key_findings = (
    aggregate
    .groupby(
        [
            "experiment",
            "eval_type",
            "model_key",
        ]
    )
    .agg(
        raw_ece_mean=("raw_ece_mean", "mean"),
        ts_ece_mean=("ts_ece_mean", "mean"),
        raw_nll_mean=("raw_nll_mean", "mean"),
        ts_nll_mean=("ts_nll_mean", "mean"),
        raw_brier_mean=("raw_brier_mean", "mean"),
        ts_brier_mean=("ts_brier_mean", "mean"),
        raw_overconfidence_mean=(
            "raw_overconfidence_mean",
            "mean"
        ),
        ts_overconfidence_mean=(
            "ts_overconfidence_mean",
            "mean"
        ),
    )
    .reset_index()
)

key_findings["model"] = (
    key_findings["model_key"]
    .map(MODEL_SHORT)
)

key_findings["ece_reduction_raw_minus_ts"] = (
    key_findings["raw_ece_mean"]
    - key_findings["ts_ece_mean"]
)

key_findings["nll_reduction_raw_minus_ts"] = (
    key_findings["raw_nll_mean"]
    - key_findings["ts_nll_mean"]
)

key_findings.to_csv(
    RUN_ROOT /
    "09_key_calibration_findings.csv",
    index=False
)

display(
    key_findings.round(5)
)

# %% [markdown] Cell 36
# ## 17. Nota metodológica reproducible

# %% Cell 37
methodology_note = f'''
PADDY DOCTOR Q1 — NOTEBOOK 09
CALIBRATION AND RELIABILITY UNDER DOMAIN SHIFT

Experiments:
- P1 clean IID benchmark.
- P4 class-matched cross-cultivar benchmark, 10 classes.
- P5 exact disease-age matched sensitivity benchmark, 5 classes.

Models:
- ResNet-50
- ConvNeXt-Tiny
- ViT-S/16
- Swin-Tiny

Seeds:
- {SEEDS}

Calibration metrics:
- ECE with {N_BINS} equal-width confidence bins.
- MCE with the same bins.
- Multiclass negative log-likelihood.
- Multiclass Brier score.
- Mean confidence minus accuracy as signed overconfidence indicator.

Temperature scaling:
- One scalar temperature per trained checkpoint.
- Temperature fitted exclusively on the corresponding validation domain.
- Objective: multiclass validation NLL.
- Test data never used to fit temperature.
- For P4/P5, the same source-validation temperature is applied to both
  in-domain and cross-domain target tests.
- Applying temperature to stored softmax probabilities uses:
  softmax(log(p) / T), which is equivalent to softmax(z / T) because
  log softmax logits differ from logits only by an additive sample constant.

Interpretation:
- Temperature scaling changes confidence but not the argmax class.
- Improvement in source-domain calibration does not imply that calibration
  transfers under domain shift.
- P5 controls observable plant age exactly within matched disease pairs,
  but it does not identify a pure causal cultivar effect.
'''

with open(
    RUN_ROOT /
    "09_calibration_methodology_notes.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(methodology_note)

print(methodology_note)

# %% [markdown] Cell 38
# ## 18. Verificación final

# %% Cell 39
required_outputs = [
    "09_config.json",
    "09_temperature_parameters_validation.csv",
    "09_test_calibration_by_run.csv",
    "09_reliability_bins_by_run.csv",
    "09_test_calibration_per_image.csv",
    "09_calibration_aggregate_summary.csv",
    "09_fixed_target_calibration_shift_by_seed.csv",
    "09_fixed_target_calibration_shift_summary.csv",
    "09_temperature_scaling_transfer_summary.csv",
    "09_temperature_validation_summary.csv",
    "09_reliability_bins_aggregate_selected.csv",
    "09_key_calibration_findings.csv",
    "09_calibration_methodology_notes.txt",
]

missing = [
    name
    for name in required_outputs
    if not (RUN_ROOT / name).exists()
]

if missing:
    raise RuntimeError(
        f"❌ Faltan outputs de N09: {missing}"
    )

if len(validation_parameters) != 60:
    raise RuntimeError(
        f"❌ Esperaba 60 temperaturas, hay {len(validation_parameters)}"
    )

if len(test_calibration) != 108:
    raise RuntimeError(
        f"❌ Esperaba 108 evaluaciones test, hay {len(test_calibration)}"
    )

print("=" * 80)
print("✅ NOTEBOOK 09 COMPLETADO")
print("=" * 80)
print("60/60 validation checkpoints procesados.")
print("108/108 evaluaciones test calibradas.")
print("Temperature scaling: validation-only.")
print("P1 + P4 + P5 integrados.")
print("Figuras:", FIGURE_ROOT)

# %% [markdown] Cell 40
# ## 19. Exportar resultados a ZIP

# %% Cell 41
from google.colab import files

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

package_name = (
    "09_PaddyDoctor_Calibration_Reliability_"
    f"RESULTS_{timestamp}"
)

temp_package = (
    Path("/content") /
    package_name
)

if temp_package.exists():
    shutil.rmtree(temp_package)

(temp_package / "results").mkdir(
    parents=True,
    exist_ok=True
)

(temp_package / "figures").mkdir(
    parents=True,
    exist_ok=True
)

# No exportar validation_cache para mantener el ZIP compacto.
for path in RUN_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(
            path,
            temp_package / "results" / path.name
        )

for path in FIGURE_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(
            path,
            temp_package / "figures" / path.name
        )

with open(
    temp_package /
    "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 09\n"
        "Calibration and Reliability Under Domain Shift\n\n"
        "Temperature scaling fitted on validation only.\n"
        "Test never used for calibration fitting.\n"
        "Includes P1, P4 and P5.\n"
        "Metrics: ECE, MCE, NLL, Brier, confidence-accuracy gap.\n"
        "Includes fixed-target domain-shift calibration analysis.\n"
    )

zip_base = (
    EXPORTS_DIR /
    package_name
)

zip_path = shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=temp_package
)

size_mb = (
    Path(zip_path).stat().st_size
    / (1024 ** 2)
)

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print(zip_path)
print(f"Tamaño: {size_mb:.2f} MB")

files.download(zip_path)

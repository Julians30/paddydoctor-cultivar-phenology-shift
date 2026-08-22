"""Auto-exported from 05_v3_1_PaddyDoctor_P1_CNN_Transformer_Benchmark_30epochs_noEarlyStop_FIXED.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 05 v3.1
# ## P1 Clean IID Benchmark definitivo — 30 épocas completas, sin early stopping
# 
# ### Objetivo
# Ejecutar el benchmark **IID limpio** sobre `P1_group_clean` con un horizonte fijo de **30 épocas completas** para todas las combinaciones modelo–semilla.
# 
# La versión v2 confirmó que 20 épocas eran insuficientes para varias ejecuciones, pero también mostró que `early stopping` con paciencia 5 podía detener prematuramente algunos entrenamientos bajo el scheduler de 30 épocas. Por ello, esta versión elimina la parada anticipada para garantizar igualdad estricta de presupuesto de optimización.
# 
# Modelos preentrenados en ImageNet-1k:
# 
# - **ResNet-50** — CNN clásica.
# - **ConvNeXt-Tiny** — CNN moderna.
# - **ViT-S/16** — Vision Transformer global.
# - **Swin-Tiny** — Transformer jerárquico.
# 
# ### Diseño experimental definitivo
# - Protocolo: `P1_group_clean`.
# - Resolución: `224 × 224`.
# - Semillas: `2026, 2027, 2028`.
# - Fine-tuning completo.
# - **30 épocas completas para las 12 ejecuciones.**
# - **Sin early stopping.**
# - Warm-up: **2 épocas**.
# - Cosine annealing recalculado para el horizonte de 30 épocas.
# - Misma transformación, optimizador y presupuesto para todos los modelos.
# - Selección del checkpoint **solo por macro-F1 de validación**.
# - El test se evalúa únicamente después de cargar el mejor checkpoint.
# - AMP activado.
# - Resultados y checkpoints se guardan permanentemente en Google Drive.
# - Esta versión usa carpetas separadas de los experimentos v1 y v2.
# - El notebook es reanudable: ejecuciones ya completadas se omiten automáticamente.
# 
# ### Métricas
# - Accuracy.
# - Balanced accuracy.
# - Macro-F1 (**métrica principal**).
# - Weighted-F1.
# - Macro AUROC OvR.
# - NLL.
# - Multiclass Brier score.
# - ECE (15 bins).
# - Métricas por clase.
# - Matriz de confusión.
# - Tiempo de entrenamiento.
# - Memoria GPU máxima.
# - Número de parámetros.
# 
# > **Este notebook requiere GPU.**

# %% [markdown] Cell 2
# ## 0. Conexión obligatoria a Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

DATA_DIR = PROJECT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"

NOTEBOOKS_DIR = PROJECT / "notebooks"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"
CONFIG_DIR = PROJECT / "config"

RUN_ROOT = RESULTS_DIR / "05_P1_30ep_noES"
MODEL_ROOT = MODELS_DIR / "05_P1_30ep_noES"
FIGURE_ROOT = FIGURES_DIR / "05_P1_30ep_noES"

for folder in [
    PROJECT, DATA_DIR, METADATA_DIR, SPLITS_DIR,
    NOTEBOOKS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    EXPORTS_DIR, CONFIG_DIR, RUN_ROOT, MODEL_ROOT, FIGURE_ROOT
]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("✅ GOOGLE DRIVE CONECTADO")
print("=" * 80)
print("Proyecto:", PROJECT)
print("Resultados:", RUN_ROOT)
print("Modelos:", MODEL_ROOT)

# %% [markdown] Cell 4
# ## 1. Autenticación persistente de Kaggle

# %% Cell 5
from google.colab import files
import shutil
import os
import sys
import subprocess

DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"

LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"

if not DRIVE_KAGGLE_JSON.exists():
    print("🔑 Selecciona tu archivo kaggle.json")
    uploaded = files.upload()

    if "kaggle.json" not in uploaded:
        raise FileNotFoundError(
            "❌ Debes seleccionar exactamente kaggle.json."
        )

    with open(DRIVE_KAGGLE_JSON, "wb") as f:
        f.write(uploaded["kaggle.json"])

    print("✅ Credencial guardada en Drive.")
else:
    print("✅ Credencial persistente encontrada.")

shutil.copy2(
    DRIVE_KAGGLE_JSON,
    LOCAL_KAGGLE_JSON
)

os.chmod(
    LOCAL_KAGGLE_JSON,
    0o600
)

# Versiones fijadas para reproducibilidad del benchmark.
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "timm==1.0.28"
    ],
    check=True
)

import kagglehub
import timm

print("✅ Kaggle autenticado.")
print("✅ timm:", timm.__version__)

# %% [markdown] Cell 6
# ## 2. Verificar GPU
# Antes de continuar en Colab:
# 
# `Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU`

# %% Cell 7
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ NO HAY GPU ACTIVA. "
        "Activa GPU en Colab y vuelve a ejecutar el notebook."
    )

DEVICE = torch.device("cuda")

gpu_name = torch.cuda.get_device_name(0)
gpu_props = torch.cuda.get_device_properties(0)
gpu_memory_gb = gpu_props.total_memory / (1024 ** 3)

print("=" * 80)
print("✅ GPU DETECTADA")
print("=" * 80)
print("GPU:", gpu_name)
print(f"VRAM: {gpu_memory_gb:.2f} GB")
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)

# %% [markdown] Cell 8
# ## 3. Configuración congelada
# No modificar durante el benchmark salvo que se documente explícitamente una nueva versión del experimento.

# %% Cell 9
import json
import platform
import numpy as np
import pandas as pd
import sklearn
import torchvision
from PIL import Image

PROTOCOL = "P1_group_clean"

MODEL_SPECS = {
    "resnet50": {
        "timm_name": "resnet50",
        "family": "CNN_classic"
    },
    "convnext_tiny": {
        "timm_name": "convnext_tiny",
        "family": "CNN_modern"
    },
    "vit_small_patch16_224": {
        "timm_name": "vit_small_patch16_224",
        "family": "Transformer_global"
    },
    "swin_tiny_patch4_window7_224": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "family": "Transformer_hierarchical"
    }
}

SEEDS = [2026, 2027, 2028]

IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 30
USE_EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = None

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2

# Transformación fija y común.
TRAIN_CROP_SCALE = [0.85, 1.0]
HFLIP_P = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER = {
    "brightness": 0.10,
    "contrast": 0.10,
    "saturation": 0.10,
    "hue": 0.02
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

config = {
    "protocol": PROTOCOL,
    "experiment_version": "05_v3_1_definitive_30epochs_no_early_stopping_fixed",
    "pilot_reference": "05_P1_20epochs_and_05_v2_30epochs_with_early_stopping",
    "change_reason": "The 20-epoch pilot showed horizon censoring, while v2 showed that early stopping patience=5 could truncate some runs under the 30-epoch scheduler; v3 therefore uses 30 full epochs for every run.",
    "model_specs": MODEL_SPECS,
    "seeds": SEEDS,
    "image_size": IMAGE_SIZE,
    "resize_size": RESIZE_SIZE,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "max_epochs": MAX_EPOCHS,
    "use_early_stopping": USE_EARLY_STOPPING,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "2-epoch linear warmup + cosine annealing",
    "criterion": "CrossEntropyLoss",
    "primary_metric": "validation_macro_f1",
    "amp": True,
    "train_crop_scale": TRAIN_CROP_SCALE,
    "horizontal_flip_probability": HFLIP_P,
    "rotation_degrees": ROTATION_DEGREES,
    "color_jitter": COLOR_JITTER,
    "imagenet_mean": IMAGENET_MEAN,
    "imagenet_std": IMAGENET_STD,
    "gpu_name": gpu_name,
    "gpu_memory_gb": round(gpu_memory_gb, 3),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "torchvision_version": torchvision.__version__,
    "timm_version": timm.__version__,
    "sklearn_version": sklearn.__version__,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
}

with open(
    RUN_ROOT / "05_experiment_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))

# %% [markdown] Cell 10
# ## 4. Verificar disponibilidad de modelos preentrenados

# %% Cell 11
# timm >= 1.0 puede devolver tags de pesos por defecto.
# include_tags=False permite comprobar los nombres base de arquitectura.
available_pretrained = set(
    timm.list_models(
        pretrained=True,
        include_tags=False
    )
)

missing_models = []

for short_name, spec in MODEL_SPECS.items():
    model_name = spec["timm_name"]

    if model_name not in available_pretrained:
        missing_models.append(model_name)

    print(
        f"{short_name:32s} -> "
        f"{'✅' if model_name in available_pretrained else '❌'}"
    )

if missing_models:
    raise RuntimeError(
        "❌ Estos modelos no aparecen con pesos preentrenados "
        f"en timm {timm.__version__}: {missing_models}"
    )

print("\n✅ Todos los modelos están disponibles con pesos preentrenados.")

# %% [markdown] Cell 12
# ## 5. Recuperar Paddy Doctor y P1

# %% Cell 13
DATASET_PATH = Path(
    "/root/.cache/kagglehub/competitions/paddy-disease-classification"
)

TRAIN_IMAGES = DATASET_PATH / "train_images"

if not TRAIN_IMAGES.exists():
    print("⬇️ Runtime nuevo: descargando Paddy Doctor...")
    DATASET_PATH = Path(
        kagglehub.competition_download(
            "paddy-disease-classification"
        )
    )
    TRAIN_IMAGES = DATASET_PATH / "train_images"

P1_CSV = SPLITS_DIR / "P1_group_clean.csv"

if not P1_CSV.exists():
    raise FileNotFoundError(
        "❌ No existe data/splits/P1_group_clean.csv. "
        "Ejecuta primero el Notebook 03."
    )

p1 = pd.read_csv(P1_CSV)

print("P1 total:", len(p1))
print("\nSplits:")
print(p1["split"].value_counts())
print("\nClases:")
print(p1["label"].value_counts())

if set(p1["split"].unique()) != {
    "train", "validation", "test"
}:
    raise RuntimeError(
        "❌ P1 no contiene exactamente train/validation/test."
    )

if p1["label"].nunique() != 10:
    raise RuntimeError(
        "❌ P1 no contiene las 10 clases esperadas."
    )

# %% [markdown] Cell 14
# ## 6. Índice físico de imágenes y mapa de etiquetas

# %% Cell 15
VALID_EXTENSIONS = {
    ".jpg", ".jpeg", ".png"
}

image_lookup = {
    p.name: p
    for p in TRAIN_IMAGES.rglob("*")
    if (
        p.is_file() and
        p.suffix.lower() in VALID_EXTENSIONS
    )
}

missing_images = [
    image_id
    for image_id in p1["image_id"]
    if image_id not in image_lookup
]

print(
    "Imágenes físicas indexadas:",
    len(image_lookup)
)

print(
    "Imágenes de P1 no encontradas:",
    len(missing_images)
)

if missing_images:
    raise RuntimeError(
        "❌ Faltan imágenes físicas de P1."
    )

CLASS_NAMES = sorted(
    p1["label"].unique()
)

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(
        CLASS_NAMES
    )
}

IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}

NUM_CLASSES = len(CLASS_NAMES)

with open(
    RUN_ROOT / "05_label_map.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        {
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS
        },
        f,
        indent=2
    )

print("\nLabel map:")
for name, idx in CLASS_TO_IDX.items():
    print(idx, "->", name)

# %% [markdown] Cell 16
# ## 7. Transformaciones comunes
# La misma transformación se aplica a las cuatro arquitecturas para controlar el pipeline de entrada.

# %% Cell 17
from torchvision import transforms
from torchvision.transforms import InterpolationMode

train_transform = transforms.Compose([
    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=tuple(TRAIN_CROP_SCALE),
        ratio=(0.90, 1.10),
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(
        p=HFLIP_P
    ),
    transforms.RandomRotation(
        degrees=ROTATION_DEGREES,
        interpolation=InterpolationMode.BILINEAR
    ),
    transforms.ColorJitter(
        brightness=COLOR_JITTER["brightness"],
        contrast=COLOR_JITTER["contrast"],
        saturation=COLOR_JITTER["saturation"],
        hue=COLOR_JITTER["hue"]
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

eval_transform = transforms.Compose([
    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(
        IMAGE_SIZE
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

print("✅ Transformaciones definidas.")

# %% [markdown] Cell 18
# ## 8. Dataset y DataLoaders reproducibles

# %% Cell 19
from torch.utils.data import Dataset, DataLoader
import random

class PaddyDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_lookup,
        class_to_idx,
        transform
    ):
        self.df = dataframe.reset_index(
            drop=True
        ).copy()

        self.image_lookup = image_lookup
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = row["image_id"]
        label_name = row["label"]

        path = self.image_lookup[
            image_id
        ]

        with Image.open(path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        target = self.class_to_idx[
            label_name
        ]

        return {
            "image": image,
            "target": torch.tensor(
                target,
                dtype=torch.long
            ),
            "image_id": image_id
        }

def seed_worker(worker_id):
    worker_seed = (
        torch.initial_seed() %
        2**32
    )

    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loaders(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_df = p1[
        p1["split"] == "train"
    ].copy()

    val_df = p1[
        p1["split"] == "validation"
    ].copy()

    test_df = p1[
        p1["split"] == "test"
    ].copy()

    train_ds = PaddyDataset(
        train_df,
        image_lookup,
        CLASS_TO_IDX,
        train_transform
    )

    val_ds = PaddyDataset(
        val_df,
        image_lookup,
        CLASS_TO_IDX,
        eval_transform
    )

    test_ds = PaddyDataset(
        test_df,
        image_lookup,
        CLASS_TO_IDX,
        eval_transform
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(
            NUM_WORKERS > 0
        ),
        worker_init_fn=seed_worker
    )

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        drop_last=False,
        **loader_kwargs
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )

print("✅ Dataset y loader factory definidos.")

# %% [markdown] Cell 20
# ## 9. Utilidades de reproducibilidad y métricas

# %% Cell 21
import math
import time
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    log_loss
)

def set_global_seed(seed):
    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def multiclass_brier(
    y_true,
    probabilities,
    n_classes
):
    one_hot = np.eye(
        n_classes,
        dtype=np.float64
    )[y_true]

    return float(
        np.mean(
            np.sum(
                (
                    probabilities -
                    one_hot
                ) ** 2,
                axis=1
            )
        )
    )

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=15
):
    confidences = probabilities.max(
        axis=1
    )

    predictions = probabilities.argmax(
        axis=1
    )

    correctness = (
        predictions == y_true
    ).astype(float)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == 0:
            mask = (
                (confidences >= lo) &
                (confidences <= hi)
            )
        else:
            mask = (
                (confidences > lo) &
                (confidences <= hi)
            )

        if not np.any(mask):
            continue

        bin_accuracy = correctness[
            mask
        ].mean()

        bin_confidence = confidences[
            mask
        ].mean()

        ece += (
            mask.mean() *
            abs(
                bin_accuracy -
                bin_confidence
            )
        )

    return float(ece)

def compute_metrics(
    y_true,
    probabilities
):
    predictions = probabilities.argmax(
        axis=1
    )

    metrics = {
        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions
            ),

        "macro_f1":
            f1_score(
                y_true,
                predictions,
                average="macro"
            ),

        "weighted_f1":
            f1_score(
                y_true,
                predictions,
                average="weighted"
            ),

        "nll":
            log_loss(
                y_true,
                probabilities,
                labels=list(
                    range(NUM_CLASSES)
                )
            ),

        "brier":
            multiclass_brier(
                y_true,
                probabilities,
                NUM_CLASSES
            ),

        "ece_15":
            expected_calibration_error(
                y_true,
                probabilities,
                n_bins=15
            )
    }

    try:
        metrics[
            "macro_auroc_ovr"
        ] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=list(
                range(NUM_CLASSES)
            )
        )

    except Exception:
        metrics[
            "macro_auroc_ovr"
        ] = np.nan

    return metrics

# %% [markdown] Cell 22
# ## 10. Crear modelo, optimizador y scheduler

# %% Cell 23
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)

def build_model(
    timm_name,
    num_classes
):
    model = timm.create_model(
        timm_name,
        pretrained=True,
        num_classes=num_classes
    )

    return model

def count_trainable_parameters(
    model
):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def build_optimizer_scheduler(
    model
):
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.10,
        end_factor=1.0,
        total_iters=WARMUP_EPOCHS
    )

    cosine_epochs = max(
        1,
        MAX_EPOCHS -
        WARMUP_EPOCHS
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=LEARNING_RATE * 0.01
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            warmup,
            cosine
        ],
        milestones=[
            WARMUP_EPOCHS
        ]
    )

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()

print("✅ Factories de entrenamiento definidas.")

# %% [markdown] Cell 24
# ## 11. Funciones de entrenamiento y evaluación

# %% Cell 25
from tqdm.auto import tqdm

def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler
):
    model.train()

    running_loss = 0.0
    n_samples = 0

    for batch in tqdm(
        loader,
        leave=False,
        desc="train"
    ):
        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True
        )

        targets = batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(images)
            loss = criterion(
                logits,
                targets
            )

        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        n_samples += batch_size

    return (
        running_loss /
        max(
            1,
            n_samples
        )
    )

@torch.no_grad()
def predict_loader(
    model,
    loader
):
    model.eval()

    all_targets = []
    all_probs = []
    all_ids = []

    running_loss = 0.0
    n_samples = 0

    for batch in tqdm(
        loader,
        leave=False,
        desc="eval"
    ):
        images = batch[
            "image"
        ].to(
            DEVICE,
            non_blocking=True
        )

        targets = batch[
            "target"
        ].to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(images)
            loss = criterion(
                logits,
                targets
            )

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )

        batch_size = images.size(0)

        running_loss += (
            loss.item() *
            batch_size
        )

        n_samples += batch_size

        all_targets.append(
            targets.cpu().numpy()
        )

        all_probs.append(
            probabilities.cpu().numpy()
        )

        all_ids.extend(
            batch["image_id"]
        )

    y_true = np.concatenate(
        all_targets
    )

    probabilities = np.concatenate(
        all_probs
    )

    mean_loss = (
        running_loss /
        max(
            1,
            n_samples
        )
    )

    return (
        mean_loss,
        y_true,
        probabilities,
        all_ids
    )

# %% [markdown] Cell 26
# ## 12. Función de una ejecución completa
# Una ejecución corresponde a un modelo + una semilla.
# 
# Si existe `COMPLETE.json`, se omite automáticamente para facilitar la reanudación después de una desconexión de Colab.

# %% Cell 27
import gc

def run_experiment(
    model_key,
    seed
):
    spec = MODEL_SPECS[
        model_key
    ]

    timm_name = spec[
        "timm_name"
    ]

    family = spec[
        "family"
    ]

    run_name = (
        f"{model_key}__seed_{seed}"
    )

    run_dir = (
        RUN_ROOT /
        run_name
    )

    model_dir = (
        MODEL_ROOT /
        run_name
    )

    figure_dir = (
        FIGURE_ROOT /
        run_name
    )

    for folder in [
        run_dir,
        model_dir,
        figure_dir
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    complete_file = (
        run_dir /
        "COMPLETE.json"
    )

    if complete_file.exists():
        print(
            f"⏭️ Ya completado: "
            f"{run_name}"
        )

        with open(
            complete_file,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    print("\n" + "=" * 80)
    print(
        "🚀",
        run_name
    )
    print("=" * 80)

    set_global_seed(
        seed
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, test_loader = (
        make_loaders(seed)
    )

    model = build_model(
        timm_name,
        NUM_CLASSES
    ).to(DEVICE)

    n_parameters = (
        count_trainable_parameters(
            model
        )
    )

    optimizer, scheduler = (
        build_optimizer_scheduler(
            model
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda"
    )

    best_val_macro_f1 = -np.inf
    best_epoch = None
    epochs_without_improvement = 0

    history_rows = []

    checkpoint_path = (
        model_dir /
        "best_model.pt"
    )

    start_time = time.time()

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler
        )

        (
            val_loss,
            y_val,
            p_val,
            _
        ) = predict_loader(
            model,
            val_loader
        )

        val_metrics = (
            compute_metrics(
                y_val,
                p_val
            )
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        epoch_seconds = (
            time.time() -
            epoch_start
        )

        history_rows.append({
            "epoch":
                epoch,
            "train_loss":
                train_loss,
            "val_loss":
                val_loss,
            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],
            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],
            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],
            "val_weighted_f1":
                val_metrics[
                    "weighted_f1"
                ],
            "learning_rate":
                current_lr,
            "epoch_seconds":
                epoch_seconds
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macroF1="
            f"{val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if (
            val_metrics[
                "macro_f1"
            ] >
            best_val_macro_f1 + 1e-6
        ):
            best_val_macro_f1 = (
                val_metrics[
                    "macro_f1"
                ]
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "model_key":
                        model_key,
                    "timm_name":
                        timm_name,
                    "family":
                        family,
                    "seed":
                        seed,
                    "epoch":
                        epoch,
                    "best_val_macro_f1":
                        best_val_macro_f1,
                    "class_to_idx":
                        CLASS_TO_IDX,
                    "config":
                        config
                },
                checkpoint_path
            )

        else:
            epochs_without_improvement += 1

        scheduler.step()

        # v3.1: sin early stopping.
        # Todas las combinaciones reciben exactamente MAX_EPOCHS épocas.

    training_seconds = (
        time.time() -
        start_time
    )

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        run_dir /
        "history.csv",
        index=False
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "❌ No se generó checkpoint."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------
    # Evaluación final: validación
    # --------------------------------------------------
    (
        final_val_loss,
        y_val,
        p_val,
        val_ids
    ) = predict_loader(
        model,
        val_loader
    )

    final_val_metrics = (
        compute_metrics(
            y_val,
            p_val
        )
    )

    # --------------------------------------------------
    # Evaluación final: test
    # --------------------------------------------------
    (
        test_loss,
        y_test,
        p_test,
        test_ids
    ) = predict_loader(
        model,
        test_loader
    )

    test_metrics = (
        compute_metrics(
            y_test,
            p_test
        )
    )

    test_predictions = (
        p_test.argmax(
            axis=1
        )
    )

    # --------------------------------------------------
    # Predicciones por imagen
    # --------------------------------------------------
    prediction_data = {
        "image_id":
            test_ids,
        "y_true_idx":
            y_test,
        "y_true":
            [
                IDX_TO_CLASS[
                    int(x)
                ]
                for x in y_test
            ],
        "y_pred_idx":
            test_predictions,
        "y_pred":
            [
                IDX_TO_CLASS[
                    int(x)
                ]
                for x in test_predictions
            ],
        "confidence":
            p_test.max(
                axis=1
            )
    }

    for class_idx, class_name in (
        IDX_TO_CLASS.items()
    ):
        prediction_data[
            f"prob__{class_name}"
        ] = p_test[
            :,
            class_idx
        ]

    predictions_df = pd.DataFrame(
        prediction_data
    )

    predictions_df.to_csv(
        run_dir /
        "test_predictions.csv",
        index=False
    )

    # --------------------------------------------------
    # Reporte por clase
    # --------------------------------------------------
    report_dict = (
        classification_report(
            y_test,
            test_predictions,
            labels=list(
                range(NUM_CLASSES)
            ),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:
        values = report_dict[
            class_name
        ]

        per_class_rows.append({
            "class_name":
                class_name,
            "precision":
                values[
                    "precision"
                ],
            "recall":
                values[
                    "recall"
                ],
            "f1_score":
                values[
                    "f1-score"
                ],
            "support":
                values[
                    "support"
                ]
        })

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    per_class_df.to_csv(
        run_dir /
        "test_per_class_metrics.csv",
        index=False
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(
            range(NUM_CLASSES)
        )
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        run_dir /
        "test_confusion_matrix_counts.csv"
    )

    cm_norm = cm.astype(
        np.float64
    ) / np.maximum(
        cm.sum(
            axis=1,
            keepdims=True
        ),
        1
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_norm_df.to_csv(
        run_dir /
        "test_confusion_matrix_normalized.csv"
    )

    # --------------------------------------------------
    # Figuras
    # --------------------------------------------------
    import matplotlib.pyplot as plt

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="train loss"
    )

    plt.plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="validation loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "loss_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig = plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_df["epoch"],
        history_df[
            "val_macro_f1"
        ],
        label="validation macro-F1"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "macro_f1_curve.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig = plt.figure(
        figsize=(11, 9)
    )

    plt.imshow(
        cm_norm,
        aspect="auto"
    )

    plt.colorbar(
        label="Normalized recall"
    )

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=90
    )

    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "True"
    )

    plt.title(
        f"{run_name} — normalized confusion matrix"
    )

    plt.tight_layout()

    plt.savefig(
        figure_dir /
        "confusion_matrix_normalized.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # --------------------------------------------------
    # Resumen final
    # --------------------------------------------------
    max_gpu_memory_gb = (
        torch.cuda.max_memory_allocated() /
        (1024 ** 3)
    )

    result = {
        "protocol":
            PROTOCOL,
        "model_key":
            model_key,
        "timm_name":
            timm_name,
        "family":
            family,
        "seed":
            seed,
        "n_parameters":
            int(n_parameters),
        "best_epoch":
            int(best_epoch),
        "best_val_macro_f1":
            float(
                best_val_macro_f1
            ),
        "final_val_loss":
            float(
                final_val_loss
            ),
        "training_seconds":
            float(
                training_seconds
            ),
        "max_gpu_memory_gb":
            float(
                max_gpu_memory_gb
            )
    }

    for key, value in (
        final_val_metrics.items()
    ):
        result[
            f"val_{key}"
        ] = float(value)

    result[
        "test_loss"
    ] = float(
        test_loss
    )

    for key, value in (
        test_metrics.items()
    ):
        result[
            f"test_{key}"
        ] = float(value)

    with open(
        complete_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2
        )

    pd.DataFrame(
        [result]
    ).to_csv(
        run_dir /
        "run_summary.csv",
        index=False
    )

    print("\n✅ COMPLETADO")
    print(
        f"Model: {model_key} | "
        f"Seed: {seed}"
    )
    print(
        f"Best epoch: "
        f"{best_epoch}"
    )
    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    # Liberar GPU antes del siguiente modelo.
    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del val_loader
    del test_loader

    gc.collect()
    torch.cuda.empty_cache()

    return result

# %% [markdown] Cell 28
# ### Nota metodológica sobre las 30 épocas completas
# 
# Las ejecuciones anteriores se conservan como **pilotos metodológicos** y no se reutilizan sus checkpoints ni sus archivos `COMPLETE.json`.
# 
# La versión v3 vuelve a entrenar todas las combinaciones desde el inicio para mantener igualdad estricta de condiciones. El scheduler se construye usando `MAX_EPOCHS = 30`, por lo que el tramo coseno se recalcula automáticamente como `30 - WARMUP_EPOCHS`.
# 
# **No se aplica early stopping.** Cada una de las 12 combinaciones recibe exactamente 30 épocas de optimización. El mejor checkpoint se conserva exclusivamente según **macro-F1 de validación** y el conjunto test no interviene en selección, parada ni ajuste de hiperparámetros.

# %% [markdown] Cell 29
# ## 13. Ejecutar benchmark completo
# 
# ### Tiempo
# Son **12 ejecuciones**:
# 
# `4 arquitecturas × 3 semillas`
# 
# El notebook guarda cada resultado inmediatamente en Drive.  
# Si Colab se desconecta, vuelve a ejecutar desde arriba: las combinaciones ya terminadas se saltarán.

# %% Cell 30
all_results = []

total_runs = (
    len(MODEL_SPECS) *
    len(SEEDS)
)

counter = 0

for model_key in MODEL_SPECS:
    for seed in SEEDS:
        counter += 1

        print(
            "\n" + "#" * 80
        )

        print(
            f"RUN {counter}/{total_runs}"
        )

        print(
            "#" * 80
        )

        result = run_experiment(
            model_key,
            seed
        )

        all_results.append(
            result
        )

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    RUN_ROOT /
    "05_all_runs_summary.csv",
    index=False
)

display(
    results_df[
        [
            "model_key",
            "family",
            "seed",
            "best_epoch",
            "best_val_macro_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_macro_auroc_ovr",
            "test_ece_15",
            "training_seconds",
            "max_gpu_memory_gb"
        ]
    ]
)

# %% [markdown] Cell 31
# ## 14. Resumen por arquitectura
# No se selecciona ganador todavía. Se reportan media y desviación estándar entre semillas.

# %% Cell 32
aggregate = (
    results_df
    .groupby(
        [
            "model_key",
            "family"
        ]
    )
    .agg(
        n_seeds=(
            "seed",
            "nunique"
        ),
        params=(
            "n_parameters",
            "first"
        ),
        best_epoch_mean=(
            "best_epoch",
            "mean"
        ),
        test_accuracy_mean=(
            "test_accuracy",
            "mean"
        ),
        test_accuracy_std=(
            "test_accuracy",
            "std"
        ),
        test_balanced_accuracy_mean=(
            "test_balanced_accuracy",
            "mean"
        ),
        test_balanced_accuracy_std=(
            "test_balanced_accuracy",
            "std"
        ),
        test_macro_f1_mean=(
            "test_macro_f1",
            "mean"
        ),
        test_macro_f1_std=(
            "test_macro_f1",
            "std"
        ),
        test_macro_auroc_mean=(
            "test_macro_auroc_ovr",
            "mean"
        ),
        test_macro_auroc_std=(
            "test_macro_auroc_ovr",
            "std"
        ),
        test_nll_mean=(
            "test_nll",
            "mean"
        ),
        test_brier_mean=(
            "test_brier",
            "mean"
        ),
        test_ece_mean=(
            "test_ece_15",
            "mean"
        ),
        training_seconds_mean=(
            "training_seconds",
            "mean"
        ),
        max_gpu_memory_gb_mean=(
            "max_gpu_memory_gb",
            "mean"
        )
    )
    .reset_index()
    .round(5)
)

aggregate = aggregate.sort_values(
    "test_macro_f1_mean",
    ascending=False
)

display(
    aggregate
)

aggregate.to_csv(
    RUN_ROOT /
    "05_model_aggregate_summary.csv",
    index=False
)

# %% [markdown] Cell 33
# ## 15. Tabla por clase agregada entre semillas

# %% Cell 34
per_class_frames = []

for model_key in MODEL_SPECS:
    for seed in SEEDS:
        run_dir = (
            RUN_ROOT /
            f"{model_key}__seed_{seed}"
        )

        path = (
            run_dir /
            "test_per_class_metrics.csv"
        )

        if not path.exists():
            continue

        temp = pd.read_csv(
            path
        )

        temp[
            "model_key"
        ] = model_key

        temp[
            "seed"
        ] = seed

        per_class_frames.append(
            temp
        )

per_class_all = pd.concat(
    per_class_frames,
    ignore_index=True
)

per_class_all.to_csv(
    RUN_ROOT /
    "05_per_class_all_seeds.csv",
    index=False
)

per_class_summary = (
    per_class_all
    .groupby(
        [
            "model_key",
            "class_name"
        ]
    )
    .agg(
        f1_mean=(
            "f1_score",
            "mean"
        ),
        f1_std=(
            "f1_score",
            "std"
        ),
        recall_mean=(
            "recall",
            "mean"
        ),
        precision_mean=(
            "precision",
            "mean"
        ),
        support=(
            "support",
            "first"
        )
    )
    .reset_index()
    .round(5)
)

display(
    per_class_summary
)

per_class_summary.to_csv(
    RUN_ROOT /
    "05_per_class_aggregate_summary.csv",
    index=False
)

# %% [markdown] Cell 35
# ## 16. Figura comparativa de macro-F1

# %% Cell 36
import matplotlib.pyplot as plt

plot_df = aggregate.sort_values(
    "test_macro_f1_mean",
    ascending=True
)

fig = plt.figure(
    figsize=(9, 5)
)

plt.errorbar(
    x=plot_df[
        "test_macro_f1_mean"
    ],
    y=np.arange(
        len(plot_df)
    ),
    xerr=plot_df[
        "test_macro_f1_std"
    ].fillna(0),
    fmt="o",
    capsize=5
)

plt.yticks(
    np.arange(
        len(plot_df)
    ),
    plot_df[
        "model_key"
    ]
)

plt.xlabel(
    "Test macro-F1 (mean ± SD)"
)

plt.ylabel(
    "Architecture"
)

plt.title(
    "Paddy Doctor — P1 clean IID benchmark"
)

plt.tight_layout()

comparison_path = (
    FIGURE_ROOT /
    "05_P1_30ep_noES_macro_f1_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print(
    "✅ Figura:",
    comparison_path
)

# %% [markdown] Cell 37
# ## 17. Verificación final

# %% Cell 38
expected_runs = (
    len(MODEL_SPECS) *
    len(SEEDS)
)

completed_runs = len(
    results_df
)

print("=" * 80)
print("RESUMEN NOTEBOOK 05 v3 — 30 ÉPOCAS COMPLETAS")
print("=" * 80)
print(
    f"Ejecuciones esperadas: "
    f"{expected_runs}"
)
print(
    f"Ejecuciones completadas: "
    f"{completed_runs}"
)

if completed_runs != expected_runs:
    raise RuntimeError(
        "❌ El benchmark no está completo."
    )

if (
    results_df[
        "seed"
    ].nunique() !=
    len(SEEDS)
):
    raise RuntimeError(
        "❌ Faltan semillas."
    )

print(
    "✅ P1 benchmark definitivo de 30 épocas completas, sin early stopping, completo."
)

print(
    "✅ Checkpoints guardados en Drive."
)

print(
    "✅ Predicciones por imagen guardadas."
)

print(
    "✅ Ya podemos pasar a P4/P5 "
    "después de revisar este ZIP."
)

# %% [markdown] Cell 39
# ## 18. Exportar resultados a ZIP
# 
# El ZIP **no incluye los checkpoints `.pt`**, porque son pesados.  
# Los checkpoints permanecen guardados en:
# 
# `PADDY_DOCTOR_Q1/models/05_P1_30ep_noES/`
# 
# El ZIP contiene:
# - configuración;
# - resumen de todas las ejecuciones;
# - resultados agregados;
# - histories;
# - predicciones test;
# - métricas por clase;
# - matrices de confusión;
# - figuras.

# %% Cell 40
import time

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

PACKAGE_NAME = (
    "05_v3_1_PaddyDoctor_P1_Benchmark_30epochs_noEarlyStop_FIXED_"
    f"RESULTS_{timestamp}"
)

TEMP_PACKAGE = (
    Path("/content") /
    PACKAGE_NAME
)

if TEMP_PACKAGE.exists():
    shutil.rmtree(
        TEMP_PACKAGE
    )

TEMP_PACKAGE.mkdir(
    parents=True,
    exist_ok=True
)

# --------------------------------------------------
# Copiar resultados
# --------------------------------------------------
dest_results = (
    TEMP_PACKAGE /
    "results" /
    "05_P1_30ep_noES"
)

shutil.copytree(
    RUN_ROOT,
    dest_results,
    dirs_exist_ok=True
)

# --------------------------------------------------
# Copiar figuras
# --------------------------------------------------
dest_figures = (
    TEMP_PACKAGE /
    "figures" /
    "05_P1_30ep_noES"
)

shutil.copytree(
    FIGURE_ROOT,
    dest_figures,
    dirs_exist_ok=True
)

# --------------------------------------------------
# README
# --------------------------------------------------
with open(
    TEMP_PACKAGE /
    "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 05\n"
        "P1 Clean IID CNN vs Transformer Benchmark — definitive 30-epoch horizon\n"
        f"Timestamp: {timestamp}\n\n"
        "Checkpoints are NOT inside this ZIP.\n"
        "They remain in Google Drive under models/05_P1_30ep_noES/.\n"
    )

zip_base = (
    EXPORTS_DIR /
    PACKAGE_NAME
)

zip_path = shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=TEMP_PACKAGE
)

size_mb = (
    Path(zip_path).stat().st_size /
    (1024 ** 2)
)

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print(zip_path)
print(
    f"Tamaño: {size_mb:.2f} MB"
)

files.download(
    zip_path
)

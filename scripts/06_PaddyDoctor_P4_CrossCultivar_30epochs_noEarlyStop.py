"""Auto-exported from 06_PaddyDoctor_P4_CrossCultivar_30epochs_noEarlyStop.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 06
# ## P4 — Generalización entre cultivares: ADT45 ↔ OTHER
# 
# ### Objetivo científico
# Evaluar si el rendimiento observado en el benchmark IID limpio (P1) se mantiene cuando cambia el dominio de cultivar.
# 
# P4 utiliza los protocolos previamente congelados:
# 
# - **P4A:** entrenar/validar en ADT45 → evaluar en OTHER.
# - **P4B:** entrenar/validar en ADT45 → evaluar en ADT45 (control in-domain).
# - **P4C:** entrenar/validar en OTHER → evaluar en ADT45.
# - **P4D:** entrenar/validar en OTHER → evaluar en OTHER (control in-domain).
# 
# ### Optimización computacional
# P4A y P4B comparten exactamente el mismo entrenamiento y validación ADT45.  
# P4C y P4D comparten exactamente el mismo entrenamiento y validación OTHER.
# 
# Por ello, **no se entrenan 48 modelos**. Se entrenan:
# 
# \[
# 4\ arquitecturas \times 3\ semillas \times 2\ dominios\ de\ entrenamiento = 24\ entrenamientos
# \]
# 
# Cada checkpoint se evalúa después en **ambos dominios de test**, generando 48 evaluaciones finales.
# 
# ### Protocolo de entrenamiento congelado desde P1
# - ResNet-50, ConvNeXt-Tiny, ViT-S/16 y Swin-Tiny.
# - Semillas 2026, 2027 y 2028.
# - 224 × 224.
# - ImageNet-1k pretraining.
# - Fine-tuning completo.
# - AdamW, LR 3e-4, weight decay 0.05.
# - Warm-up de 2 épocas + cosine annealing.
# - **30 épocas completas, sin early stopping.**
# - Mejor checkpoint seleccionado **solo por macro-F1 de validación**.
# - Test utilizado únicamente después de cargar el mejor checkpoint.
# - AMP activado.
# 
# ### Interpretación
# P4 está balanceado por clase entre dominios, pero **no está emparejado exactamente por edad/fenología**.  
# Por tanto, P4 mide generalización cross-cultivar **potencialmente influida por confusión fenológica**.
# 
# La separación más estricta entre cultivar y edad se realizará en **P5 exact-age-matched**.

# %% [markdown] Cell 2
# ## 0. Montaje robusto de Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path
import time
import os

# Desmontar una conexión previa/inestable si existe.
try:
    drive.flush_and_unmount()
except Exception:
    pass

time.sleep(2)

drive.mount("/content/drive", force_remount=True)
time.sleep(2)

MYDRIVE = Path("/content/drive/MyDrive")

connected = False
for attempt in range(1, 6):
    try:
        if not MYDRIVE.exists():
            raise OSError("MyDrive todavía no responde.")
        _ = list(MYDRIVE.iterdir())[:3]
        connected = True
        print(f"✅ MyDrive responde correctamente — intento {attempt}")
        break
    except Exception as e:
        print(f"⚠️ Intento {attempt}/5: {e}")
        time.sleep(5)

if not connected:
    raise RuntimeError(
        "❌ Google Drive no responde. No inicies el entrenamiento."
    )

PROJECT = MYDRIVE / "PADDY_DOCTOR_Q1"

DATA_DIR = PROJECT / "data"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"

NOTEBOOKS_DIR = PROJECT / "notebooks"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"
CONFIG_DIR = PROJECT / "config"

RUN_ROOT = RESULTS_DIR / "06_P4_cross_cultivar_30ep_noES"
MODEL_ROOT = MODELS_DIR / "06_P4_cross_cultivar_30ep_noES"
FIGURE_ROOT = FIGURES_DIR / "06_P4_cross_cultivar_30ep_noES"

folders = [
    PROJECT, DATA_DIR, METADATA_DIR, SPLITS_DIR,
    NOTEBOOKS_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    EXPORTS_DIR, CONFIG_DIR, RUN_ROOT, MODEL_ROOT, FIGURE_ROOT
]

for folder in folders:
    success = False
    for attempt in range(1, 4):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            success = True
            break
        except (ConnectionAbortedError, OSError) as e:
            print(f"⚠️ Drive inestable en {folder.name}, intento {attempt}/3: {e}")
            time.sleep(5)
    if not success:
        raise RuntimeError(f"❌ No fue posible acceder/crear: {folder}")

print("=" * 80)
print("✅ GOOGLE DRIVE CONECTADO Y VERIFICADO")
print("=" * 80)
print("Proyecto:  ", PROJECT)
print("Resultados:", RUN_ROOT)
print("Modelos:   ", MODEL_ROOT)
print("Figuras:   ", FIGURE_ROOT)

# %% [markdown] Cell 4
# ## 1. Autenticación persistente de Kaggle y dependencias

# %% Cell 5
from google.colab import files
import shutil
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
        raise FileNotFoundError("❌ Debes seleccionar exactamente kaggle.json.")

    with open(DRIVE_KAGGLE_JSON, "wb") as f:
        f.write(uploaded["kaggle.json"])

    print("✅ Credencial guardada de forma persistente en Drive.")
else:
    print("✅ Credencial persistente encontrada.")

shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
os.chmod(LOCAL_KAGGLE_JSON, 0o600)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.28"],
    check=True
)

import kagglehub
import timm

print("✅ Kaggle autenticado.")
print("✅ timm:", timm.__version__)

# %% [markdown] Cell 6
# ## 2. Verificar GPU

# %% Cell 7
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ NO HAY GPU ACTIVA. Activa GPU en Colab antes de continuar."
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
# ## 3. Configuración experimental congelada

# %% Cell 9
import json
import platform
import random
import numpy as np
import pandas as pd
import sklearn
import torchvision
from PIL import Image

PROTOCOL = "P4_cross_cultivar_class_matched"

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
TRAIN_DOMAINS = ["ADT45", "OTHER"]
EVAL_DOMAINS = ["ADT45", "OTHER"]

PROTOCOL_MAP = {
    ("ADT45", "OTHER"): "P4A_ADT45_to_OTHER_matched",
    ("ADT45", "ADT45"): "P4B_ADT45_to_ADT45_control",
    ("OTHER", "ADT45"): "P4C_OTHER_to_ADT45_matched",
    ("OTHER", "OTHER"): "P4D_OTHER_to_OTHER_control",
}

IMAGE_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 2

MAX_EPOCHS = 30
USE_EARLY_STOPPING = False

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 2

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
    "experiment_version": "06_P4_cross_cultivar_30epochs_no_early_stopping",
    "scientific_note": (
        "P4 is class-matched across cultivar domains but is not exact-age-matched; "
        "phenological confounding is addressed separately in P5."
    ),
    "model_specs": MODEL_SPECS,
    "seeds": SEEDS,
    "train_domains": TRAIN_DOMAINS,
    "eval_domains": EVAL_DOMAINS,
    "protocol_map": {f"{k[0]}->{k[1]}": v for k, v in PROTOCOL_MAP.items()},
    "n_training_runs": len(MODEL_SPECS) * len(SEEDS) * len(TRAIN_DOMAINS),
    "n_final_evaluations": len(MODEL_SPECS) * len(SEEDS) * len(TRAIN_DOMAINS) * len(EVAL_DOMAINS),
    "image_size": IMAGE_SIZE,
    "resize_size": RESIZE_SIZE,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "max_epochs": MAX_EPOCHS,
    "use_early_stopping": USE_EARLY_STOPPING,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": "2-epoch linear warmup + cosine annealing",
    "criterion": "CrossEntropyLoss",
    "primary_selection_metric": "validation_macro_f1",
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

with open(RUN_ROOT / "06_experiment_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))

# %% [markdown] Cell 10
# ## 4. Verificar modelos preentrenados

# %% Cell 11
available_pretrained = set(
    timm.list_models(
        pretrained=True,
        include_tags=False
    )
)

missing_models = []

for short_name, spec in MODEL_SPECS.items():
    model_name = spec["timm_name"]
    ok = model_name in available_pretrained

    print(f"{short_name:32s} -> {'✅' if ok else '❌'}")

    if not ok:
        missing_models.append(model_name)

if missing_models:
    raise RuntimeError(
        f"❌ Modelos preentrenados no disponibles en timm {timm.__version__}: "
        f"{missing_models}"
    )

print("\n✅ Todos los modelos están disponibles.")

# %% [markdown] Cell 12
# ## 5. Recuperar Paddy Doctor y cargar los cuatro protocolos P4

# %% Cell 13
DATASET_PATH = Path(
    "/root/.cache/kagglehub/competitions/paddy-disease-classification"
)

TRAIN_IMAGES = DATASET_PATH / "train_images"

if not TRAIN_IMAGES.exists():
    print("⬇️ Runtime nuevo: descargando Paddy Doctor...")
    DATASET_PATH = Path(
        kagglehub.competition_download("paddy-disease-classification")
    )
    TRAIN_IMAGES = DATASET_PATH / "train_images"

P4_FILES = {
    "P4A": SPLITS_DIR / "P4A_ADT45_to_OTHER_matched.csv",
    "P4B": SPLITS_DIR / "P4B_ADT45_to_ADT45_control.csv",
    "P4C": SPLITS_DIR / "P4C_OTHER_to_ADT45_matched.csv",
    "P4D": SPLITS_DIR / "P4D_OTHER_to_OTHER_control.csv",
}

missing_split_files = [
    str(path) for path in P4_FILES.values() if not path.exists()
]

if missing_split_files:
    raise FileNotFoundError(
        "❌ Faltan archivos P4. Ejecuta primero Notebook 03. "
        f"Faltantes: {missing_split_files}"
    )

P4 = {
    key: pd.read_csv(path)
    for key, path in P4_FILES.items()
}

for key, df in P4.items():
    print("\n", key, df["protocol"].iloc[0])
    print(df["split"].value_counts().to_dict())

    active = df[df["split"] != "EXCLUDED"]
    if active["label"].nunique() != 10:
        raise RuntimeError(f"❌ {key} no contiene las 10 clases activas.")

# Verificar que P4A/P4B comparten train + validation ADT45.
for split_name in ["train", "validation"]:
    a_ids = set(P4["P4A"].loc[P4["P4A"]["split"] == split_name, "image_id"])
    b_ids = set(P4["P4B"].loc[P4["P4B"]["split"] == split_name, "image_id"])
    if a_ids != b_ids:
        raise RuntimeError(
            f"❌ P4A y P4B no comparten exactamente {split_name}."
        )

# Verificar que P4C/P4D comparten train + validation OTHER.
for split_name in ["train", "validation"]:
    c_ids = set(P4["P4C"].loc[P4["P4C"]["split"] == split_name, "image_id"])
    d_ids = set(P4["P4D"].loc[P4["P4D"]["split"] == split_name, "image_id"])
    if c_ids != d_ids:
        raise RuntimeError(
            f"❌ P4C y P4D no comparten exactamente {split_name}."
        )

DOMAIN_DATA = {
    "ADT45": {
        "train": P4["P4A"].loc[P4["P4A"]["split"] == "train"].copy(),
        "validation": P4["P4A"].loc[P4["P4A"]["split"] == "validation"].copy(),
        "tests": {
            "ADT45": P4["P4B"].loc[P4["P4B"]["split"] == "test"].copy(),
            "OTHER": P4["P4A"].loc[P4["P4A"]["split"] == "test"].copy(),
        }
    },
    "OTHER": {
        "train": P4["P4C"].loc[P4["P4C"]["split"] == "train"].copy(),
        "validation": P4["P4C"].loc[P4["P4C"]["split"] == "validation"].copy(),
        "tests": {
            "ADT45": P4["P4C"].loc[P4["P4C"]["split"] == "test"].copy(),
            "OTHER": P4["P4D"].loc[P4["P4D"]["split"] == "test"].copy(),
        }
    }
}

print("\n" + "=" * 80)
print("✅ P4 CARGADO Y VERIFICADO")
print("=" * 80)

for train_domain in TRAIN_DOMAINS:
    print(f"\nTRAIN DOMAIN = {train_domain}")
    print("train:", len(DOMAIN_DATA[train_domain]["train"]))
    print("validation:", len(DOMAIN_DATA[train_domain]["validation"]))
    for eval_domain in EVAL_DOMAINS:
        df = DOMAIN_DATA[train_domain]["tests"][eval_domain]
        protocol_name = PROTOCOL_MAP[(train_domain, eval_domain)]
        print(f"test {eval_domain}: {len(df)} -> {protocol_name}")

# Guardar auditoría compacta.
audit_rows = []
for train_domain in TRAIN_DOMAINS:
    audit_rows.append({
        "train_domain": train_domain,
        "partition": "train",
        "n_images": len(DOMAIN_DATA[train_domain]["train"]),
        "n_classes": DOMAIN_DATA[train_domain]["train"]["label"].nunique()
    })
    audit_rows.append({
        "train_domain": train_domain,
        "partition": "validation",
        "n_images": len(DOMAIN_DATA[train_domain]["validation"]),
        "n_classes": DOMAIN_DATA[train_domain]["validation"]["label"].nunique()
    })
    for eval_domain in EVAL_DOMAINS:
        test_df = DOMAIN_DATA[train_domain]["tests"][eval_domain]
        audit_rows.append({
            "train_domain": train_domain,
            "partition": f"test_{eval_domain}",
            "n_images": len(test_df),
            "n_classes": test_df["label"].nunique()
        })

pd.DataFrame(audit_rows).to_csv(
    RUN_ROOT / "06_P4_dataset_audit.csv",
    index=False
)

# %% [markdown] Cell 14
# ## 6. Índice físico de imágenes y mapa de clases

# %% Cell 15
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

image_lookup = {
    p.name: p
    for p in TRAIN_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
}

all_active_frames = []
for train_domain in TRAIN_DOMAINS:
    all_active_frames.append(DOMAIN_DATA[train_domain]["train"])
    all_active_frames.append(DOMAIN_DATA[train_domain]["validation"])
    for eval_domain in EVAL_DOMAINS:
        all_active_frames.append(DOMAIN_DATA[train_domain]["tests"][eval_domain])

all_active = pd.concat(all_active_frames, ignore_index=True)
active_ids = sorted(all_active["image_id"].unique())

missing_images = [
    image_id for image_id in active_ids
    if image_id not in image_lookup
]

print("Imágenes físicas indexadas:", len(image_lookup))
print("Imágenes P4 únicas activas:", len(active_ids))
print("Imágenes P4 no encontradas:", len(missing_images))

if missing_images:
    raise RuntimeError(
        f"❌ Faltan imágenes físicas de P4. Ejemplos: {missing_images[:10]}"
    )

CLASS_NAMES = sorted(all_active["label"].unique())

if len(CLASS_NAMES) != 10:
    raise RuntimeError(
        f"❌ Se esperaban 10 clases; se encontraron {len(CLASS_NAMES)}."
    )

CLASS_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IDX_TO_CLASS = {
    idx: name
    for name, idx in CLASS_TO_IDX.items()
}

NUM_CLASSES = len(CLASS_NAMES)

with open(RUN_ROOT / "06_label_map.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS
        },
        f,
        indent=2
    )

print("\nClases:")
for name, idx in CLASS_TO_IDX.items():
    print(f"{idx:2d} -> {name}")

# %% [markdown] Cell 16
# ## 7. Transformaciones comunes

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
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.RandomHorizontalFlip(p=HFLIP_P),
    transforms.RandomRotation(
        ROTATION_DEGREES,
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
    transforms.CenterCrop(IMAGE_SIZE),
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

class PaddyDataset(Dataset):
    def __init__(
        self,
        dataframe,
        image_lookup,
        class_to_idx,
        transform
    ):
        self.df = dataframe.reset_index(drop=True).copy()
        self.image_lookup = image_lookup
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = row["image_id"]
        label_name = row["label"]
        path = self.image_lookup[image_id]

        with Image.open(path) as img:
            image = img.convert("RGB")

        image = self.transform(image)
        target = self.class_to_idx[label_name]

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "image_id": image_id
        }

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_loader(
    dataframe,
    transform,
    seed,
    shuffle
):
    ds = PaddyDataset(
        dataframe,
        image_lookup,
        CLASS_TO_IDX,
        transform
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        worker_init_fn=seed_worker
    )

def make_loaders(train_domain, seed):
    train_df = DOMAIN_DATA[train_domain]["train"]
    val_df = DOMAIN_DATA[train_domain]["validation"]

    train_loader = build_loader(
        train_df,
        train_transform,
        seed,
        shuffle=True
    )

    val_loader = build_loader(
        val_df,
        eval_transform,
        seed,
        shuffle=False
    )

    test_loaders = {
        eval_domain: build_loader(
            DOMAIN_DATA[train_domain]["tests"][eval_domain],
            eval_transform,
            seed,
            shuffle=False
        )
        for eval_domain in EVAL_DOMAINS
    }

    return train_loader, val_loader, test_loaders

print("✅ Dataset y loader factory definidos.")

# %% [markdown] Cell 20
# ## 9. Reproducibilidad y métricas

# %% Cell 21
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
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def multiclass_brier(y_true, probabilities, n_classes):
    one_hot = np.eye(n_classes, dtype=np.float64)[y_true]
    return float(
        np.mean(
            np.sum((probabilities - one_hot) ** 2, axis=1)
        )
    )

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=15
):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]

        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)

        if not np.any(mask):
            continue

        bin_accuracy = correctness[mask].mean()
        bin_confidence = confidences[mask].mean()

        ece += (
            mask.mean()
            * abs(bin_accuracy - bin_confidence)
        )

    return float(ece)

def compute_metrics(y_true, probabilities):
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "weighted_f1": f1_score(y_true, predictions, average="weighted"),
        "nll": log_loss(
            y_true,
            probabilities,
            labels=list(range(NUM_CLASSES))
        ),
        "brier": multiclass_brier(
            y_true,
            probabilities,
            NUM_CLASSES
        ),
        "ece_15": expected_calibration_error(
            y_true,
            probabilities,
            n_bins=15
        )
    }

    try:
        metrics["macro_auroc_ovr"] = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
            labels=list(range(NUM_CLASSES))
        )
    except Exception:
        metrics["macro_auroc_ovr"] = np.nan

    return metrics

print("✅ Métricas definidas.")

# %% [markdown] Cell 22
# ## 10. Modelo, optimizador y scheduler

# %% Cell 23
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR
)

def build_model(timm_name, num_classes):
    return timm.create_model(
        timm_name,
        pretrained=True,
        num_classes=num_classes
    )

def count_trainable_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

def build_optimizer_scheduler(model):
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
        MAX_EPOCHS - WARMUP_EPOCHS
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=LEARNING_RATE * 0.01
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[WARMUP_EPOCHS]
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
        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )
        targets = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(1, n_samples)

@torch.no_grad()
def predict_loader(model, loader):
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
        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )
        targets = batch["target"].to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        probabilities = torch.softmax(
            logits.float(),
            dim=1
        )

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        all_targets.append(targets.cpu().numpy())
        all_probs.append(probabilities.cpu().numpy())
        all_ids.extend(batch["image_id"])

    y_true = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probs)
    mean_loss = running_loss / max(1, n_samples)

    return mean_loss, y_true, probabilities, all_ids

print("✅ Funciones de train/eval definidas.")

# %% [markdown] Cell 26
# ## 12. Función de evaluación de un dominio de test
# 
# Cada checkpoint entrenado se evalúa en:
# 
# - su propio dominio (**control in-domain**);
# - el dominio opuesto (**cross-cultivar**).
# 
# Se guardan predicciones por imagen, métricas por clase y matrices de confusión por separado.

# %% Cell 27
import matplotlib.pyplot as plt

def save_test_evaluation(
    model,
    loader,
    eval_df,
    train_domain,
    eval_domain,
    model_key,
    seed,
    run_dir,
    figure_dir
):
    protocol_name = PROTOCOL_MAP[
        (train_domain, eval_domain)
    ]

    eval_type = (
        "control_in_domain"
        if train_domain == eval_domain
        else "cross_domain"
    )

    eval_dir = (
        run_dir /
        f"eval__{eval_domain}"
    )
    eval_figure_dir = (
        figure_dir /
        f"eval__{eval_domain}"
    )

    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_figure_dir.mkdir(parents=True, exist_ok=True)

    (
        test_loss,
        y_test,
        p_test,
        test_ids
    ) = predict_loader(
        model,
        loader
    )

    test_metrics = compute_metrics(
        y_test,
        p_test
    )

    test_predictions = p_test.argmax(axis=1)

    metadata_cols = [
        c for c in [
            "image_id",
            "label",
            "variety",
            "age",
            "domain",
            "visual_group_id"
        ]
        if c in eval_df.columns
    ]

    metadata = (
        eval_df[metadata_cols]
        .drop_duplicates("image_id")
        .set_index("image_id")
    )

    prediction_data = {
        "image_id": test_ids,
        "y_true_idx": y_test,
        "y_true": [
            IDX_TO_CLASS[int(x)]
            for x in y_test
        ],
        "y_pred_idx": test_predictions,
        "y_pred": [
            IDX_TO_CLASS[int(x)]
            for x in test_predictions
        ],
        "confidence": p_test.max(axis=1)
    }

    for class_idx, class_name in IDX_TO_CLASS.items():
        prediction_data[
            f"prob__{class_name}"
        ] = p_test[:, class_idx]

    predictions_df = pd.DataFrame(prediction_data)

    # Añadir metadatos sin cambiar el orden de predicción.
    for col in metadata.columns:
        predictions_df[col] = [
            metadata.loc[image_id, col]
            if image_id in metadata.index
            else np.nan
            for image_id in predictions_df["image_id"]
        ]

    predictions_df.to_csv(
        eval_dir / "test_predictions.csv",
        index=False
    )

    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    per_class_rows = []

    for class_name in CLASS_NAMES:
        values = report_dict[class_name]

        per_class_rows.append({
            "class_name": class_name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": values["support"]
        })

    per_class_df = pd.DataFrame(per_class_rows)

    per_class_df.to_csv(
        eval_dir / "test_per_class_metrics.csv",
        index=False
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(range(NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        eval_dir / "test_confusion_matrix_counts.csv"
    )

    cm_norm = (
        cm.astype(np.float64)
        / np.maximum(
            cm.sum(axis=1, keepdims=True),
            1
        )
    )

    cm_norm_df = pd.DataFrame(
        cm_norm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_norm_df.to_csv(
        eval_dir / "test_confusion_matrix_normalized.csv"
    )

    fig = plt.figure(figsize=(11, 9))

    plt.imshow(
        cm_norm,
        aspect="auto"
    )
    plt.colorbar(label="Normalized recall")

    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=90
    )
    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(
        f"{model_key} seed {seed} | "
        f"{train_domain} → {eval_domain}"
    )
    plt.tight_layout()

    plt.savefig(
        eval_figure_dir /
        "confusion_matrix_normalized.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    summary = {
        "protocol": protocol_name,
        "eval_type": eval_type,
        "train_domain": train_domain,
        "eval_domain": eval_domain,
        "model_key": model_key,
        "seed": int(seed),
        "test_loss": float(test_loss),
    }

    for key, value in test_metrics.items():
        summary[f"test_{key}"] = float(value)

    pd.DataFrame([summary]).to_csv(
        eval_dir / "evaluation_summary.csv",
        index=False
    )

    return summary

# %% [markdown] Cell 28
# ## 13. Una ejecución de entrenamiento completa
# 
# Una ejecución corresponde a:
# 
# \[
# arquitectura + semilla + dominio\ de\ entrenamiento
# \]
# 
# El entrenamiento se realiza **una sola vez** y después el checkpoint se evalúa en ADT45 y OTHER.
# 
# Si el archivo `COMPLETE.json` ya existe, la ejecución se omite automáticamente al reanudar Colab.

# %% Cell 29
import gc

def run_training_experiment(
    model_key,
    seed,
    train_domain
):
    spec = MODEL_SPECS[model_key]
    timm_name = spec["timm_name"]
    family = spec["family"]

    run_name = (
        f"train_{train_domain}__"
        f"{model_key}__seed_{seed}"
    )

    run_dir = RUN_ROOT / run_name
    model_dir = MODEL_ROOT / run_name
    figure_dir = FIGURE_ROOT / run_name

    for folder in [
        run_dir,
        model_dir,
        figure_dir
    ]:
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

    complete_file = run_dir / "COMPLETE.json"

    if complete_file.exists():
        print(f"⏭️ Ya completado: {run_name}")

        with open(
            complete_file,
            "r",
            encoding="utf-8"
        ) as f:
            payload = json.load(f)

        return payload["training_summary"], payload["evaluations"]

    print("\n" + "=" * 80)
    print("🚀", run_name)
    print("=" * 80)

    set_global_seed(seed)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    train_loader, val_loader, test_loaders = (
        make_loaders(
            train_domain,
            seed
        )
    )

    model = build_model(
        timm_name,
        NUM_CLASSES
    ).to(DEVICE)

    n_parameters = count_trainable_parameters(
        model
    )

    optimizer, scheduler = (
        build_optimizer_scheduler(model)
    )

    scaler = torch.amp.GradScaler("cuda")

    best_val_macro_f1 = -np.inf
    best_epoch = None

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

        val_metrics = compute_metrics(
            y_val,
            p_val
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_seconds = (
            time.time() -
            epoch_start
        )

        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": current_lr,
            "epoch_seconds": epoch_seconds
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_macroF1={val_metrics['macro_f1']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if (
            val_metrics["macro_f1"]
            > best_val_macro_f1 + 1e-6
        ):
            best_val_macro_f1 = (
                val_metrics["macro_f1"]
            )
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_key": model_key,
                    "timm_name": timm_name,
                    "family": family,
                    "seed": seed,
                    "train_domain": train_domain,
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "class_to_idx": CLASS_TO_IDX,
                    "config": config
                },
                checkpoint_path
            )

        scheduler.step()

    training_seconds = (
        time.time() -
        start_time
    )

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        run_dir / "history.csv",
        index=False
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"❌ No se generó checkpoint para {run_name}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Evaluación final del mejor checkpoint sobre validation.
    (
        final_val_loss,
        y_val,
        p_val,
        _
    ) = predict_loader(
        model,
        val_loader
    )

    final_val_metrics = compute_metrics(
        y_val,
        p_val
    )

    # Curvas de entrenamiento.
    fig = plt.figure(figsize=(8, 5))

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
        figure_dir / "loss_curve.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        history_df["epoch"],
        history_df["val_macro_f1"],
        label="validation macro-F1"
    )

    plt.axvline(
        best_epoch,
        linestyle="--",
        label=f"best epoch = {best_epoch}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title(run_name)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        figure_dir / "macro_f1_curve.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)

    max_gpu_memory_gb = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    training_summary = {
        "protocol": PROTOCOL,
        "train_domain": train_domain,
        "model_key": model_key,
        "timm_name": timm_name,
        "family": family,
        "seed": int(seed),
        "n_parameters": int(n_parameters),
        "n_train": int(
            len(DOMAIN_DATA[train_domain]["train"])
        ),
        "n_validation": int(
            len(DOMAIN_DATA[train_domain]["validation"])
        ),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(
            best_val_macro_f1
        ),
        "final_val_loss": float(
            final_val_loss
        ),
        "training_seconds": float(
            training_seconds
        ),
        "max_gpu_memory_gb": float(
            max_gpu_memory_gb
        )
    }

    for key, value in final_val_metrics.items():
        training_summary[
            f"val_{key}"
        ] = float(value)

    # Evaluar el mismo checkpoint en ambos dominios de test.
    evaluations = []

    for eval_domain in EVAL_DOMAINS:
        evaluation_summary = save_test_evaluation(
            model=model,
            loader=test_loaders[eval_domain],
            eval_df=DOMAIN_DATA[train_domain]["tests"][eval_domain],
            train_domain=train_domain,
            eval_domain=eval_domain,
            model_key=model_key,
            seed=seed,
            run_dir=run_dir,
            figure_dir=figure_dir
        )

        evaluation_summary.update({
            "family": family,
            "timm_name": timm_name,
            "n_parameters": int(n_parameters),
            "best_epoch": int(best_epoch),
            "best_val_macro_f1": float(best_val_macro_f1),
            "training_seconds": float(training_seconds),
            "max_gpu_memory_gb": float(max_gpu_memory_gb)
        })

        evaluations.append(
            evaluation_summary
        )

    pd.DataFrame(
        [training_summary]
    ).to_csv(
        run_dir / "training_summary.csv",
        index=False
    )

    pd.DataFrame(
        evaluations
    ).to_csv(
        run_dir / "all_test_evaluations.csv",
        index=False
    )

    payload = {
        "training_summary": training_summary,
        "evaluations": evaluations
    }

    with open(
        complete_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            indent=2
        )

    print("\n✅ ENTRENAMIENTO COMPLETADO")
    print(
        f"Model: {model_key} | "
        f"Seed: {seed} | "
        f"Train domain: {train_domain}"
    )
    print(
        f"Best epoch: {best_epoch} | "
        f"Val macro-F1: {best_val_macro_f1:.4f}"
    )

    for ev in evaluations:
        print(
            f"  {ev['protocol']}: "
            f"Macro-F1={ev['test_macro_f1']:.4f} | "
            f"BalAcc={ev['test_balanced_accuracy']:.4f}"
        )

    del model
    del optimizer
    del scheduler
    del scaler
    del train_loader
    del val_loader
    del test_loaders

    gc.collect()
    torch.cuda.empty_cache()

    return training_summary, evaluations

# %% [markdown] Cell 30
# ## 14. Ejecutar P4 completo
# 
# ### Total
# - **24 entrenamientos**
# - **48 evaluaciones finales**
# 
# Al ser P4 mucho más pequeño que P1 (1.848 imágenes de entrenamiento por dominio), el tiempo por corrida debe ser considerablemente menor.
# 
# > Puedes ejecutar todas las celdas y dejar Colab trabajando.  
# > Cada entrenamiento finalizado queda marcado con `COMPLETE.json`.

# %% Cell 31
all_training_summaries = []
all_evaluations = []

total_training_runs = (
    len(MODEL_SPECS)
    * len(SEEDS)
    * len(TRAIN_DOMAINS)
)

counter = 0

for train_domain in TRAIN_DOMAINS:
    for model_key in MODEL_SPECS:
        for seed in SEEDS:
            counter += 1

            print("\n" + "#" * 80)
            print(
                f"TRAINING RUN {counter}/{total_training_runs} | "
                f"DOMAIN={train_domain}"
            )
            print("#" * 80)

            (
                training_summary,
                evaluations
            ) = run_training_experiment(
                model_key=model_key,
                seed=seed,
                train_domain=train_domain
            )

            all_training_summaries.append(
                training_summary
            )

            all_evaluations.extend(
                evaluations
            )

training_df = pd.DataFrame(
    all_training_summaries
)

evaluations_df = pd.DataFrame(
    all_evaluations
)

training_df.to_csv(
    RUN_ROOT /
    "06_all_training_runs_summary.csv",
    index=False
)

evaluations_df.to_csv(
    RUN_ROOT /
    "06_all_evaluations_summary.csv",
    index=False
)

display(
    training_df[
        [
            "train_domain",
            "model_key",
            "seed",
            "best_epoch",
            "best_val_macro_f1",
            "training_seconds",
            "max_gpu_memory_gb"
        ]
    ]
)

display(
    evaluations_df[
        [
            "protocol",
            "eval_type",
            "train_domain",
            "eval_domain",
            "model_key",
            "seed",
            "test_macro_f1",
            "test_balanced_accuracy",
            "test_ece_15"
        ]
    ]
)

# %% [markdown] Cell 32
# ## 15. Resumen agregado por arquitectura, dirección y tipo de evaluación

# %% Cell 33
aggregate = (
    evaluations_df
    .groupby(
        [
            "model_key",
            "family",
            "train_domain",
            "eval_domain",
            "protocol",
            "eval_type"
        ]
    )
    .agg(
        n_seeds=("seed", "nunique"),
        test_accuracy_mean=("test_accuracy", "mean"),
        test_accuracy_std=("test_accuracy", "std"),
        test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
        test_balanced_accuracy_std=("test_balanced_accuracy", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        test_macro_auroc_mean=("test_macro_auroc_ovr", "mean"),
        test_macro_auroc_std=("test_macro_auroc_ovr", "std"),
        test_nll_mean=("test_nll", "mean"),
        test_nll_std=("test_nll", "std"),
        test_brier_mean=("test_brier", "mean"),
        test_brier_std=("test_brier", "std"),
        test_ece_mean=("test_ece_15", "mean"),
        test_ece_std=("test_ece_15", "std")
    )
    .reset_index()
    .round(5)
)

aggregate = aggregate.sort_values(
    [
        "train_domain",
        "eval_type",
        "test_macro_f1_mean"
    ],
    ascending=[
        True,
        True,
        False
    ]
)

aggregate.to_csv(
    RUN_ROOT /
    "06_P4_aggregate_summary.csv",
    index=False
)

display(aggregate)

# %% [markdown] Cell 34
# ## 16. Degradación cross-cultivar frente al control in-domain
# 
# Para cada combinación modelo–semilla–dominio de entrenamiento:
# 
# \[
# \Delta F1 = F1_{cross} - F1_{control}
# \]
# 
# También se reporta:
# 
# \[
# Drop = F1_{control} - F1_{cross}
# \]
# 
# Un `Drop > 0` indica pérdida de rendimiento al cambiar de cultivar.

# %% Cell 35
gap_rows = []

for train_domain in TRAIN_DOMAINS:
    cross_domain = (
        "OTHER"
        if train_domain == "ADT45"
        else "ADT45"
    )

    for model_key in MODEL_SPECS:
        for seed in SEEDS:
            temp = evaluations_df[
                (evaluations_df["train_domain"] == train_domain)
                & (evaluations_df["model_key"] == model_key)
                & (evaluations_df["seed"] == seed)
            ].copy()

            control = temp[
                temp["eval_domain"] == train_domain
            ]

            cross = temp[
                temp["eval_domain"] == cross_domain
            ]

            if len(control) != 1 or len(cross) != 1:
                raise RuntimeError(
                    "❌ No se pudo construir un par control/cross único para "
                    f"{train_domain}, {model_key}, seed {seed}."
                )

            control = control.iloc[0]
            cross = cross.iloc[0]

            gap_rows.append({
                "train_domain": train_domain,
                "cross_eval_domain": cross_domain,
                "model_key": model_key,
                "family": MODEL_SPECS[model_key]["family"],
                "seed": seed,

                "control_macro_f1": control["test_macro_f1"],
                "cross_macro_f1": cross["test_macro_f1"],
                "delta_macro_f1_cross_minus_control": (
                    cross["test_macro_f1"]
                    - control["test_macro_f1"]
                ),
                "drop_macro_f1_control_minus_cross": (
                    control["test_macro_f1"]
                    - cross["test_macro_f1"]
                ),

                "control_balanced_accuracy": control["test_balanced_accuracy"],
                "cross_balanced_accuracy": cross["test_balanced_accuracy"],
                "delta_balanced_accuracy_cross_minus_control": (
                    cross["test_balanced_accuracy"]
                    - control["test_balanced_accuracy"]
                ),
                "drop_balanced_accuracy_control_minus_cross": (
                    control["test_balanced_accuracy"]
                    - cross["test_balanced_accuracy"]
                ),

                "control_ece_15": control["test_ece_15"],
                "cross_ece_15": cross["test_ece_15"],
                "delta_ece_cross_minus_control": (
                    cross["test_ece_15"]
                    - control["test_ece_15"]
                ),
            })

gap_df = pd.DataFrame(gap_rows)

gap_df.to_csv(
    RUN_ROOT /
    "06_P4_robustness_gaps_by_seed.csv",
    index=False
)

gap_summary = (
    gap_df
    .groupby(
        [
            "model_key",
            "family",
            "train_domain",
            "cross_eval_domain"
        ]
    )
    .agg(
        n_seeds=("seed", "nunique"),
        control_macro_f1_mean=("control_macro_f1", "mean"),
        cross_macro_f1_mean=("cross_macro_f1", "mean"),
        drop_macro_f1_mean=("drop_macro_f1_control_minus_cross", "mean"),
        drop_macro_f1_std=("drop_macro_f1_control_minus_cross", "std"),
        control_balanced_accuracy_mean=("control_balanced_accuracy", "mean"),
        cross_balanced_accuracy_mean=("cross_balanced_accuracy", "mean"),
        drop_balanced_accuracy_mean=("drop_balanced_accuracy_control_minus_cross", "mean"),
        delta_ece_mean=("delta_ece_cross_minus_control", "mean")
    )
    .reset_index()
    .round(5)
)

gap_summary.to_csv(
    RUN_ROOT /
    "06_P4_robustness_gap_summary.csv",
    index=False
)

display(
    gap_summary.sort_values(
        [
            "train_domain",
            "drop_macro_f1_mean"
        ]
    )
)

# %% [markdown] Cell 36
# ## 17. Resumen bidireccional de robustez por arquitectura

# %% Cell 37
bidirectional = (
    gap_df
    .groupby(
        [
            "model_key",
            "family"
        ]
    )
    .agg(
        n_direction_seed_pairs=("seed", "count"),
        control_macro_f1_mean=("control_macro_f1", "mean"),
        cross_macro_f1_mean=("cross_macro_f1", "mean"),
        drop_macro_f1_mean=("drop_macro_f1_control_minus_cross", "mean"),
        drop_macro_f1_std=("drop_macro_f1_control_minus_cross", "std"),
        drop_balanced_accuracy_mean=("drop_balanced_accuracy_control_minus_cross", "mean"),
        delta_ece_mean=("delta_ece_cross_minus_control", "mean")
    )
    .reset_index()
    .round(5)
)

# Menor drop = mayor robustez, pero no declarar "ganador"
# hasta completar inferencia estadística posterior.
bidirectional = bidirectional.sort_values(
    "drop_macro_f1_mean",
    ascending=True
)

bidirectional.to_csv(
    RUN_ROOT /
    "06_P4_bidirectional_robustness_summary.csv",
    index=False
)

display(bidirectional)

# %% [markdown] Cell 38
# ## 18. Métricas por clase para las 48 evaluaciones

# %% Cell 39
per_class_frames = []

for train_domain in TRAIN_DOMAINS:
    for model_key in MODEL_SPECS:
        for seed in SEEDS:
            run_dir = (
                RUN_ROOT /
                f"train_{train_domain}__"
                f"{model_key}__seed_{seed}"
            )

            for eval_domain in EVAL_DOMAINS:
                path = (
                    run_dir /
                    f"eval__{eval_domain}" /
                    "test_per_class_metrics.csv"
                )

                if not path.exists():
                    continue

                temp = pd.read_csv(path)

                temp["model_key"] = model_key
                temp["family"] = MODEL_SPECS[model_key]["family"]
                temp["seed"] = seed
                temp["train_domain"] = train_domain
                temp["eval_domain"] = eval_domain
                temp["protocol"] = PROTOCOL_MAP[
                    (train_domain, eval_domain)
                ]
                temp["eval_type"] = (
                    "control_in_domain"
                    if train_domain == eval_domain
                    else "cross_domain"
                )

                per_class_frames.append(temp)

if not per_class_frames:
    raise RuntimeError(
        "❌ No se encontraron métricas por clase."
    )

per_class_all = pd.concat(
    per_class_frames,
    ignore_index=True
)

per_class_all.to_csv(
    RUN_ROOT /
    "06_P4_per_class_all_evaluations.csv",
    index=False
)

per_class_summary = (
    per_class_all
    .groupby(
        [
            "model_key",
            "train_domain",
            "eval_domain",
            "eval_type",
            "class_name"
        ]
    )
    .agg(
        f1_mean=("f1_score", "mean"),
        f1_std=("f1_score", "std"),
        recall_mean=("recall", "mean"),
        precision_mean=("precision", "mean"),
        support_mean=("support", "mean")
    )
    .reset_index()
    .round(5)
)

per_class_summary.to_csv(
    RUN_ROOT /
    "06_P4_per_class_aggregate_summary.csv",
    index=False
)

display(per_class_summary)

# %% [markdown] Cell 40
# ## 19. Figuras comparativas P4

# %% Cell 41
# Figura 1: control vs cross para ADT45-trained.
for train_domain in TRAIN_DOMAINS:
    fig = plt.figure(figsize=(10, 6))

    models = list(MODEL_SPECS.keys())
    x = np.arange(len(models))
    offsets = [-0.12, 0.12]

    for offset, eval_type in zip(
        offsets,
        ["control_in_domain", "cross_domain"]
    ):
        subset = aggregate[
            (aggregate["train_domain"] == train_domain)
            & (aggregate["eval_type"] == eval_type)
        ].set_index("model_key").reindex(models)

        plt.errorbar(
            x + offset,
            subset["test_macro_f1_mean"],
            yerr=subset["test_macro_f1_std"].fillna(0),
            fmt="o",
            capsize=5,
            label=eval_type
        )

    plt.xticks(
        x,
        models,
        rotation=20,
        ha="right"
    )
    plt.ylabel("Test Macro-F1 (mean ± SD)")
    plt.xlabel("Architecture")
    plt.title(
        f"P4 — train {train_domain}: "
        "in-domain vs cross-cultivar"
    )
    plt.legend()
    plt.tight_layout()

    path = (
        FIGURE_ROOT /
        f"06_P4_train_{train_domain}_control_vs_cross_macroF1.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )
    plt.show()

# Figura 2: drop bidireccional.
plot_df = bidirectional.sort_values(
    "drop_macro_f1_mean",
    ascending=True
)

fig = plt.figure(figsize=(9, 5))

plt.errorbar(
    x=plot_df["drop_macro_f1_mean"],
    y=np.arange(len(plot_df)),
    xerr=plot_df["drop_macro_f1_std"].fillna(0),
    fmt="o",
    capsize=5
)

plt.yticks(
    np.arange(len(plot_df)),
    plot_df["model_key"]
)

plt.axvline(0.0, linestyle="--")
plt.xlabel("Macro-F1 drop: in-domain − cross-domain")
plt.ylabel("Architecture")
plt.title("P4 — Bidirectional cross-cultivar degradation")
plt.tight_layout()

path = (
    FIGURE_ROOT /
    "06_P4_bidirectional_macroF1_drop.png"
)

plt.savefig(
    path,
    dpi=250,
    bbox_inches="tight"
)
plt.show()

print("✅ Figuras P4 guardadas en:", FIGURE_ROOT)

# %% [markdown] Cell 42
# ## 20. Verificación final

# %% Cell 43
expected_training_runs = (
    len(MODEL_SPECS)
    * len(SEEDS)
    * len(TRAIN_DOMAINS)
)

expected_evaluations = (
    expected_training_runs
    * len(EVAL_DOMAINS)
)

completed_training_runs = len(training_df)
completed_evaluations = len(evaluations_df)

print("=" * 80)
print("RESUMEN NOTEBOOK 06 — P4 CROSS-CULTIVAR")
print("=" * 80)
print(
    f"Entrenamientos esperados: {expected_training_runs}"
)
print(
    f"Entrenamientos completados: {completed_training_runs}"
)
print(
    f"Evaluaciones esperadas: {expected_evaluations}"
)
print(
    f"Evaluaciones completadas: {completed_evaluations}"
)

if completed_training_runs != expected_training_runs:
    raise RuntimeError(
        "❌ P4 no tiene los 24 entrenamientos esperados."
    )

if completed_evaluations != expected_evaluations:
    raise RuntimeError(
        "❌ P4 no tiene las 48 evaluaciones esperadas."
    )

for protocol_name in PROTOCOL_MAP.values():
    n_rows = int(
        (evaluations_df["protocol"] == protocol_name).sum()
    )

    expected_protocol_rows = (
        len(MODEL_SPECS)
        * len(SEEDS)
    )

    print(
        f"{protocol_name}: "
        f"{n_rows}/{expected_protocol_rows}"
    )

    if n_rows != expected_protocol_rows:
        raise RuntimeError(
            f"❌ Protocolo incompleto: {protocol_name}"
        )

print("\n✅ P4 completo.")
print("✅ 24 entrenamientos / 48 evaluaciones.")
print("✅ Predicciones por imagen guardadas.")
print("✅ Checkpoints guardados en Drive.")
print(
    "⚠️ Interpretar P4 como cultivar shift potencialmente "
    "confundido por fenología; P5 realizará control exacto de edad."
)

# %% [markdown] Cell 44
# ## 21. Exportar resultados a ZIP

# %% Cell 45
import shutil

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

PACKAGE_NAME = (
    "06_PaddyDoctor_P4_CrossCultivar_30epochs_noEarlyStop_"
    f"RESULTS_{timestamp}"
)

TEMP_PACKAGE = (
    Path("/content") /
    PACKAGE_NAME
)

if TEMP_PACKAGE.exists():
    shutil.rmtree(TEMP_PACKAGE)

TEMP_PACKAGE.mkdir(
    parents=True,
    exist_ok=True
)

dest_results = (
    TEMP_PACKAGE /
    "results" /
    "06_P4_cross_cultivar_30ep_noES"
)

shutil.copytree(
    RUN_ROOT,
    dest_results,
    dirs_exist_ok=True
)

dest_figures = (
    TEMP_PACKAGE /
    "figures" /
    "06_P4_cross_cultivar_30ep_noES"
)

shutil.copytree(
    FIGURE_ROOT,
    dest_figures,
    dirs_exist_ok=True
)

# Incluir solamente los CSV de protocolo necesarios.
dest_splits = (
    TEMP_PACKAGE /
    "data" /
    "splits"
)

dest_splits.mkdir(
    parents=True,
    exist_ok=True
)

for path in P4_FILES.values():
    shutil.copy2(
        path,
        dest_splits / path.name
    )

with open(
    TEMP_PACKAGE /
    "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 06\n"
        "P4 Cross-Cultivar Generalization Benchmark\n"
        "30 full epochs, no early stopping\n"
        f"Timestamp: {timestamp}\n\n"
        "Training design:\n"
        "- 4 architectures x 3 seeds x 2 train domains = 24 trainings.\n"
        "- Every checkpoint evaluated on ADT45 and OTHER = 48 evaluations.\n\n"
        "Protocol mapping:\n"
        "- P4A: ADT45 -> OTHER (cross-domain)\n"
        "- P4B: ADT45 -> ADT45 (control)\n"
        "- P4C: OTHER -> ADT45 (cross-domain)\n"
        "- P4D: OTHER -> OTHER (control)\n\n"
        "Important interpretation:\n"
        "P4 is class-matched but not exact-age-matched. "
        "Phenological confounding is addressed in P5.\n\n"
        "Checkpoints are NOT inside this ZIP.\n"
        "They remain in Google Drive under "
        "models/06_P4_cross_cultivar_30ep_noES/.\n"
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
    Path(zip_path).stat().st_size
    / (1024 ** 2)
)

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print(zip_path)
print(f"Tamaño: {size_mb:.2f} MB")

files.download(zip_path)

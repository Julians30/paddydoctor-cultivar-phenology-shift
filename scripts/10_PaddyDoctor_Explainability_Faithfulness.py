"""Auto-exported from 10_PaddyDoctor_Explainability_Faithfulness.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 10
# ## Explainability cuantitativa, faithfulness y estabilidad
# 
# ### Objetivo científico
# Analizar **qué regiones visuales sostienen las predicciones** y si esas explicaciones cambian bajo *cultivar/domain shift*, sin convertir el trabajo en una colección puramente cualitativa de mapas de calor.
# 
# Se integran P1, P4 y P5.
# 
# ### Métodos de atribución
# - **ResNet-50:** Grad-CAM.
# - **ConvNeXt-Tiny:** Grad-CAM.
# - **ViT-S/16:** Attention Rollout.
# - **Swin-Tiny:** Grad-CAM adaptado a activaciones jerárquicas/channel-last.
# 
# ### Métricas cuantitativas
# - **Deletion AUC** ↓.
# - **Insertion AUC** ↑.
# - **Confidence drop @20% deletion** ↑.
# - **Cross-seed saliency consistency** ↑.
# - **Control-vs-cross saliency similarity** sobre las mismas imágenes objetivo.
# 
# ### Muestreo
# Determinista, estratificado por clase e independiente de la predicción:
# - P1: 2 imágenes/clase = 20.
# - P4: 2 imágenes/clase y target = 40 únicas.
# - P5: 3 imágenes/clase y target = 30 únicas.
# 
# Control y cross-domain usan exactamente las mismas imágenes objetivo.
# 
# ### Alcance
# El dataset no contiene máscaras de lesión. Por ello no se reportan IoU/Dice ni se afirma localización patológica validada.
# 
# > **No entrena redes.** Reutiliza los 60 checkpoints ya guardados.

# %% [markdown] Cell 2
# ## 0. Montaje de Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path
import os
import time

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

RUN_ROOT = RESULTS_DIR / "10_explainability_faithfulness"
FIGURE_ROOT = FIGURES_DIR / "10_explainability_faithfulness"
MAP_ROOT = RUN_ROOT / "saliency_maps"
RUN_CACHE = RUN_ROOT / "run_cache"

for folder in [RUN_ROOT, FIGURE_ROOT, MAP_ROOT, RUN_CACHE, EXPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("✅ Proyecto:", PROJECT)
print("✅ Salida N10:", RUN_ROOT)

# %% [markdown] Cell 4
# ## 1. Dependencias y dispositivo

# %% Cell 5
import sys
import subprocess
import shutil
import json
import gc
import math
import hashlib
import itertools
import warnings

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.28"],
    check=True
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from PIL import Image

import timm
import kagglehub

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from scipy.stats import spearmanr

DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"
LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"

if DRIVE_KAGGLE_JSON.exists():
    shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
    os.chmod(LOCAL_KAGGLE_JSON, 0o600)
    print("✅ kaggle.json restaurado desde Drive.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Dispositivo:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Recomiendo L4; faithfulness requiere múltiples forwards/backwards.")

print("timm:", timm.__version__)
print("torch:", torch.__version__)

# %% [markdown] Cell 6
# ## 2. Configuración congelada

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

METHOD_BY_MODEL = {
    "resnet50": "Grad-CAM",
    "convnext_tiny": "Grad-CAM",
    "vit_small_patch16_224": "Attention Rollout",
    "swin_tiny_patch4_window7_224": "Grad-CAM adapted",
}

SEEDS = [2026, 2027, 2028]

IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

P1_PER_CLASS = 2
P4_PER_CLASS = 2
P5_PER_CLASS = 3

PATCH_GRID = 14
PERTURB_FRACTIONS = np.linspace(0.0, 1.0, 11)
CONFIDENCE_DROP_FRACTION = 0.20
MAP_SAVE_SIZE = 56

N_BOOT = 2000
BOOT_SEED = 20260822
QUALITATIVE_SEED = 2026
QUALITATIVE_IMAGES_PER_FIGURE = 4

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
    "notebook": "10_explainability_faithfulness",
    "models": MODELS,
    "seeds": SEEDS,
    "methods": METHOD_BY_MODEL,
    "p1_per_class": P1_PER_CLASS,
    "p4_per_class": P4_PER_CLASS,
    "p5_per_class": P5_PER_CLASS,
    "patch_grid": PATCH_GRID,
    "perturb_fractions": PERTURB_FRACTIONS.tolist(),
    "map_save_size": MAP_SAVE_SIZE,
    "n_boot": N_BOOT,
    "qualitative_seed": QUALITATIVE_SEED,
    "device": str(DEVICE),
}

with open(RUN_ROOT / "10_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))

# %% [markdown] Cell 8
# ## 3. Verificar los 60 checkpoints

# %% Cell 9
expected = {
    "P1": (P1_RESULTS, P1_MODELS, 12),
    "P4": (P4_RESULTS, P4_MODELS, 24),
    "P5": (P5_RESULTS, P5_MODELS, 24),
}

for experiment, (result_root, model_root, n_expected) in expected.items():
    complete = len(list(result_root.rglob("COMPLETE.json")))
    checkpoints = len(list(model_root.rglob("best_model.pt")))

    print(
        experiment,
        "| COMPLETE:", f"{complete}/{n_expected}",
        "| checkpoints:", f"{checkpoints}/{n_expected}",
    )

    if complete != n_expected:
        raise RuntimeError(f"❌ {experiment}: resultados incompletos.")
    if checkpoints != n_expected:
        raise RuntimeError(f"❌ {experiment}: checkpoints incompletos.")

print("✅ 60 checkpoints verificados.")

# %% [markdown] Cell 10
# ## 4. Recuperar las imágenes

# %% Cell 11
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

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

image_lookup = {
    p.name: p
    for p in TRAIN_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
}

print("✅ Imágenes indexadas:", len(image_lookup))

# %% [markdown] Cell 12
# ## 5. Muestreo estratificado determinista e independiente de la predicción

# %% Cell 13
def stable_hash(text):
    return int(
        hashlib.sha256(
            str(text).encode("utf-8")
        ).hexdigest()[:16],
        16
    )

def deterministic_stratified_sample(df, n_per_class):
    rows = []

    for label, g in df.groupby("y_true", sort=True):
        g = g.copy()
        g["_hash"] = g["image_id"].astype(str).map(stable_hash)
        g = g.sort_values(["_hash", "image_id"])

        if len(g) < n_per_class:
            raise RuntimeError(
                f"❌ Clase {label}: {len(g)} imágenes; se requieren {n_per_class}."
            )

        rows.append(g.head(n_per_class))

    return pd.concat(rows, ignore_index=True).drop(columns=["_hash"])

def prediction_path(
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

    root = P4_RESULTS if experiment == "P4" else P5_RESULTS

    return (
        root /
        f"train_{train_domain}__{model_key}__seed_{seed}" /
        f"eval__{eval_domain}" /
        "test_predictions.csv"
    )

reference_model = "resnet50"
reference_seed = 2026
sample_specs = []

p1_ref = pd.read_csv(
    prediction_path("P1", reference_model, reference_seed)
)
p1_sample = deterministic_stratified_sample(
    p1_ref[["image_id", "y_true"]],
    P1_PER_CLASS
)
p1_sample["experiment"] = "P1"
p1_sample["target_domain"] = "MIXED"
sample_specs.append(p1_sample)

for target_domain in ["ADT45", "OTHER"]:
    ref = pd.read_csv(
        prediction_path(
            "P4",
            reference_model,
            reference_seed,
            train_domain=target_domain,
            eval_domain=target_domain,
        )
    )
    sample = deterministic_stratified_sample(
        ref[["image_id", "y_true"]],
        P4_PER_CLASS
    )
    sample["experiment"] = "P4"
    sample["target_domain"] = target_domain
    sample_specs.append(sample)

for target_domain in ["ADT45", "OTHER"]:
    ref = pd.read_csv(
        prediction_path(
            "P5",
            reference_model,
            reference_seed,
            train_domain=target_domain,
            eval_domain=target_domain,
        )
    )
    sample = deterministic_stratified_sample(
        ref[["image_id", "y_true"]],
        P5_PER_CLASS
    )
    sample["experiment"] = "P5"
    sample["target_domain"] = target_domain
    sample_specs.append(sample)

SAMPLE_REGISTRY = pd.concat(sample_specs, ignore_index=True)

SAMPLE_REGISTRY.to_csv(
    RUN_ROOT / "10_explanation_sample_registry.csv",
    index=False
)

print(SAMPLE_REGISTRY.groupby(["experiment", "target_domain"]).size())
display(SAMPLE_REGISTRY.head(20))

# %% [markdown] Cell 14
# ## 6. Transformación de evaluación y helpers de imagen

# %% Cell 15
eval_transform = transforms.Compose([
    transforms.Resize(
        RESIZE_SIZE,
        interpolation=InterpolationMode.BICUBIC
    ),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

MEAN_T = torch.tensor(
    IMAGENET_MEAN,
    dtype=torch.float32
).view(3, 1, 1)

STD_T = torch.tensor(
    IMAGENET_STD,
    dtype=torch.float32
).view(3, 1, 1)

def load_tensor(image_id):
    path = image_lookup.get(str(image_id))

    if path is None:
        raise FileNotFoundError(f"❌ No encontré {image_id}")

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        tensor = eval_transform(rgb)

    return tensor

def denormalize_tensor(x):
    return (
        x.detach().cpu() * STD_T + MEAN_T
    ).clamp(0.0, 1.0)

def normalize_map(saliency):
    saliency = np.asarray(saliency, dtype=np.float32)
    saliency = np.nan_to_num(
        saliency, nan=0.0, posinf=0.0, neginf=0.0
    )
    saliency = np.maximum(saliency, 0.0)

    mn = float(saliency.min())
    mx = float(saliency.max())

    if mx <= mn + 1e-12:
        return np.zeros_like(saliency, dtype=np.float32)

    return ((saliency - mn) / (mx - mn)).astype(np.float32)

def resize_saliency(saliency, size):
    t = torch.from_numpy(
        np.asarray(saliency, dtype=np.float32)
    )[None, None]

    out = F.interpolate(
        t,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    return normalize_map(out[0, 0].numpy())

print("✅ Helpers de imagen listos.")

# %% [markdown] Cell 16
# ## 7. Implementar Grad-CAM y Attention Rollout

# %% Cell 17
class GradCAMExplainer:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.handle = target_layer.register_forward_hook(
            self._forward_hook
        )

    def _forward_hook(self, module, inputs, output):
        if not torch.is_tensor(output):
            raise RuntimeError(
                "❌ Grad-CAM recibió una activación no tensor."
            )

        self.activations = output

        # During faithfulness forwards the model runs under torch.no_grad().
        # Register gradient hooks only when gradients are actually enabled.
        if output.requires_grad:
            output.register_hook(
                self._save_gradient
            )

    def _save_gradient(self, gradient):
        self.gradients = gradient

    @staticmethod
    def _to_bchw(tensor):
        if tensor.ndim == 4:
            # BCHW: canales suelen estar en dimensión 1.
            # BHWC: canales suelen estar al final (Swin/timm).
            if (
                tensor.shape[-1] > tensor.shape[1]
                and tensor.shape[-1] > tensor.shape[2]
            ):
                return tensor.permute(0, 3, 1, 2)

            return tensor

        if tensor.ndim == 3:
            # B x tokens x channels.
            b, n, c = tensor.shape

            side = int(round(math.sqrt(n)))

            if side * side == n:
                return (
                    tensor
                    .reshape(b, side, side, c)
                    .permute(0, 3, 1, 2)
                )

            side = int(round(math.sqrt(n - 1)))

            if side * side == (n - 1):
                patches = tensor[:, 1:, :]

                return (
                    patches
                    .reshape(b, side, side, c)
                    .permute(0, 3, 1, 2)
                )

        raise RuntimeError(
            f"❌ Forma de activación no soportada: {tuple(tensor.shape)}"
        )

    def explain(self, image_tensor, target_class=None):
        self.activations = None
        self.gradients = None

        x = image_tensor.unsqueeze(0).to(DEVICE)

        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)

        pred_idx = int(probabilities.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_idx

        confidence = float(
            probabilities[0, target_class].detach().cpu().item()
        )

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "❌ Grad-CAM no capturó activaciones/gradientes."
            )

        activations = self._to_bchw(
            self.activations.detach()
        )

        gradients = self._to_bchw(
            self.gradients.detach()
        )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].detach().cpu().numpy()
        cam = normalize_map(cam)

        return {
            "saliency": cam,
            "pred_idx": pred_idx,
            "target_idx": int(target_class),
            "target_probability": confidence,
        }

    def close(self):
        self.handle.remove()


class AttentionRolloutExplainer:
    def __init__(self, model):
        self.model = model
        self.attentions = []
        self.handles = []
        self.capture = False

        if not hasattr(model, "blocks"):
            raise RuntimeError(
                "❌ Este Attention Rollout espera un ViT con model.blocks."
            )

        for block in model.blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

            handle = block.attn.attn_drop.register_forward_hook(
                self._attention_hook
            )

            self.handles.append(handle)

    def _attention_hook(self, module, inputs, output):
        if self.capture and torch.is_tensor(output):
            self.attentions.append(
                output.detach().float().cpu()
            )

    def explain(self, image_tensor, target_class=None):
        self.attentions = []
        self.capture = True

        x = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(
                logits,
                dim=1
            )

        self.capture = False

        pred_idx = int(
            probabilities.argmax(dim=1).item()
        )

        if target_class is None:
            target_class = pred_idx

        confidence = float(
            probabilities[
                0,
                target_class
            ].cpu().item()
        )

        if len(self.attentions) == 0:
            raise RuntimeError(
                "❌ No se capturaron matrices de atención. "
                "Verifique fused_attn=False."
            )

        rollout = None

        for attention in self.attentions:
            # B x heads x tokens x tokens.
            a = attention.mean(
                dim=1
            )

            n_tokens = a.shape[-1]

            eye = torch.eye(
                n_tokens,
                dtype=a.dtype
            ).unsqueeze(0)

            a = a + eye

            a = a / (
                a.sum(
                    dim=-1,
                    keepdim=True
                ) + 1e-12
            )

            if rollout is None:
                rollout = a
            else:
                rollout = torch.bmm(
                    a,
                    rollout
                )

        cls_to_patches = rollout[
            0,
            0,
            1:
        ]

        n_patches = int(
            cls_to_patches.numel()
        )

        side = int(
            round(
                math.sqrt(
                    n_patches
                )
            )
        )

        if side * side != n_patches:
            raise RuntimeError(
                f"❌ Attention Rollout no puede reconstruir "
                f"{n_patches} patches."
            )

        saliency = (
            cls_to_patches
            .reshape(
                side,
                side
            )
            .numpy()
        )

        saliency = resize_saliency(
            saliency,
            IMAGE_SIZE
        )

        return {
            "saliency": saliency,
            "pred_idx": pred_idx,
            "target_idx": int(target_class),
            "target_probability": confidence,
        }

    def close(self):
        for handle in self.handles:
            handle.remove()


def choose_gradcam_layer(model, model_key):
    if model_key == "resnet50":
        return model.layer4[-1]

    if model_key == "convnext_tiny":
        return model.stages[-1].blocks[-1]

    if model_key == "swin_tiny_patch4_window7_224":
        return model.layers[-1].blocks[-1].norm1

    raise ValueError(
        f"No hay capa Grad-CAM definida para {model_key}"
    )


def build_explainer(model, model_key):
    if model_key == "vit_small_patch16_224":
        return AttentionRolloutExplainer(
            model
        )

    return GradCAMExplainer(
        model,
        choose_gradcam_layer(
            model,
            model_key
        )
    )


print("✅ Grad-CAM + Attention Rollout implementados.")

# %% [markdown] Cell 18
# ## 8. Faithfulness: deletion, insertion y confidence drop

# %% Cell 19
def patch_ranking_from_saliency(
    saliency,
    grid=PATCH_GRID,
):
    small = resize_saliency(
        saliency,
        grid
    )

    scores = small.reshape(-1)

    # Mayor atribución primero.
    ranking = np.argsort(
        -scores,
        kind="stable"
    )

    return ranking, scores


def apply_selected_patches(
    source,
    destination,
    selected_patch_indices,
    grid=PATCH_GRID,
):
    patch_h = IMAGE_SIZE // grid
    patch_w = IMAGE_SIZE // grid

    for patch_idx in selected_patch_indices:
        row = int(patch_idx) // grid
        col = int(patch_idx) % grid

        y0 = row * patch_h
        y1 = (
            IMAGE_SIZE
            if row == grid - 1
            else (row + 1) * patch_h
        )

        x0 = col * patch_w
        x1 = (
            IMAGE_SIZE
            if col == grid - 1
            else (col + 1) * patch_w
        )

        destination[
            :,
            y0:y1,
            x0:x1
        ] = source[
            :,
            y0:y1,
            x0:x1
        ]


@torch.no_grad()
def faithfulness_curves(
    model,
    image_tensor,
    saliency,
    target_class,
):
    ranking, _ = patch_ranking_from_saliency(
        saliency,
        grid=PATCH_GRID
    )

    n_patches = len(ranking)

    # Zero en espacio normalizado = color medio ImageNet.
    baseline = torch.zeros_like(
        image_tensor
    )

    deletion_batch = []
    insertion_batch = []

    for fraction in PERTURB_FRACTIONS:
        k = int(
            round(
                float(fraction)
                * n_patches
            )
        )

        selected = ranking[:k]

        # Deletion: original -> baseline en regiones relevantes.
        deletion = image_tensor.clone()

        apply_selected_patches(
            source=baseline,
            destination=deletion,
            selected_patch_indices=selected,
            grid=PATCH_GRID
        )

        deletion_batch.append(
            deletion
        )

        # Insertion: baseline -> original en regiones relevantes.
        insertion = baseline.clone()

        apply_selected_patches(
            source=image_tensor,
            destination=insertion,
            selected_patch_indices=selected,
            grid=PATCH_GRID
        )

        insertion_batch.append(
            insertion
        )

    combined = torch.stack(
        deletion_batch + insertion_batch,
        dim=0
    ).to(DEVICE)

    probabilities = []

    # Chunk conservador para evitar picos de VRAM.
    for start in range(
        0,
        len(combined),
        22
    ):
        batch = combined[
            start:start + 22
        ]

        logits = model(batch)

        probs = torch.softmax(
            logits,
            dim=1
        )[
            :,
            target_class
        ]

        probabilities.extend(
            probs.detach().cpu().numpy().tolist()
        )

    n_steps = len(PERTURB_FRACTIONS)

    deletion_prob = np.asarray(
        probabilities[:n_steps],
        dtype=float
    )

    insertion_prob = np.asarray(
        probabilities[n_steps:],
        dtype=float
    )

    fractions = np.asarray(
        PERTURB_FRACTIONS,
        dtype=float
    )

    deletion_auc = float(
        np.trapz(
            deletion_prob,
            fractions
        )
    )

    insertion_auc = float(
        np.trapz(
            insertion_prob,
            fractions
        )
    )

    idx20 = int(
        np.argmin(
            np.abs(
                fractions
                - CONFIDENCE_DROP_FRACTION
            )
        )
    )

    confidence_drop_20 = float(
        deletion_prob[0]
        - deletion_prob[idx20]
    )

    return {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "confidence_drop_20": confidence_drop_20,
        "deletion_curve": deletion_prob,
        "insertion_curve": insertion_prob,
    }


print("✅ Faithfulness curves implementadas.")

# %% [markdown] Cell 20
# ## 9. Carga de checkpoints y definición de los 108 bloques de explicación

# %% Cell 21
def checkpoint_path(
    experiment,
    model_key,
    seed,
    train_domain=None,
):
    if experiment == "P1":
        return (
            P1_MODELS /
            f"{model_key}__seed_{seed}" /
            "best_model.pt"
        )

    root = (
        P4_MODELS
        if experiment == "P4"
        else P5_MODELS
    )

    return (
        root /
        f"train_{train_domain}__{model_key}__seed_{seed}" /
        "best_model.pt"
    )


def load_model_checkpoint(
    experiment,
    model_key,
    seed,
    train_domain=None,
):
    path = checkpoint_path(
        experiment,
        model_key,
        seed,
        train_domain=train_domain
    )

    if not path.exists():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    class_to_idx = {
        str(k): int(v)
        for k, v in checkpoint[
            "class_to_idx"
        ].items()
    }

    idx_to_class = {
        int(v): str(k)
        for k, v in class_to_idx.items()
    }

    timm_name = checkpoint.get(
        "timm_name",
        model_key
    )

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=len(class_to_idx)
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        class_to_idx,
        idx_to_class,
        path
    )


RUN_SPECS = []

# P1 = 12 bloques.
for model_key in MODELS:
    for seed in SEEDS:
        RUN_SPECS.append({
            "experiment": "P1",
            "protocol": "P1_IID",
            "train_domain": "MIXED",
            "eval_domain": "MIXED",
            "eval_type": "IID",
            "model_key": model_key,
            "seed": seed,
        })

# P4 = 48 bloques.
for train_domain in ["ADT45", "OTHER"]:
    for eval_domain in ["ADT45", "OTHER"]:
        for model_key in MODELS:
            for seed in SEEDS:
                RUN_SPECS.append({
                    "experiment": "P4",
                    "protocol": P4_PROTOCOL_MAP[
                        (
                            train_domain,
                            eval_domain
                        )
                    ],
                    "train_domain": train_domain,
                    "eval_domain": eval_domain,
                    "eval_type": (
                        "control"
                        if train_domain == eval_domain
                        else "cross"
                    ),
                    "model_key": model_key,
                    "seed": seed,
                })

# P5 = 48 bloques.
for train_domain in ["ADT45", "OTHER"]:
    for eval_domain in ["ADT45", "OTHER"]:
        for model_key in MODELS:
            for seed in SEEDS:
                RUN_SPECS.append({
                    "experiment": "P5",
                    "protocol": P5_PROTOCOL_MAP[
                        (
                            train_domain,
                            eval_domain
                        )
                    ],
                    "train_domain": train_domain,
                    "eval_domain": eval_domain,
                    "eval_type": (
                        "control"
                        if train_domain == eval_domain
                        else "cross"
                    ),
                    "model_key": model_key,
                    "seed": seed,
                })

if len(RUN_SPECS) != 108:
    raise RuntimeError(
        f"❌ Se esperaban 108 bloques; hay {len(RUN_SPECS)}."
    )

print("✅ Bloques de explicación:", len(RUN_SPECS))

# %% [markdown] Cell 22
# ## 10. Ejecutar explicaciones y métricas de faithfulness
# 
# Este bloque es el más costoso del Notebook 10.  
# Tiene **reanudación por bloque**: cada combinación experimento/protocolo/modelo/seed guarda `DONE.json`, CSV y mapas compactos.
# 
# Si Colab se desconecta, vuelva a ejecutar el notebook: los bloques completos se saltan.

# %% Cell 23
def run_id_from_spec(spec):
    return (
        f"{spec['experiment']}__"
        f"{spec['protocol']}__"
        f"{spec['model_key']}__"
        f"seed_{spec['seed']}"
    )

def map_file_from_spec(spec):
    return (
        MAP_ROOT /
        f"{run_id_from_spec(spec)}.npz"
    )

def metrics_file_from_spec(spec):
    return (
        RUN_CACHE /
        f"{run_id_from_spec(spec)}__metrics.csv"
    )

def done_file_from_spec(spec):
    return (
        RUN_CACHE /
        f"{run_id_from_spec(spec)}__DONE.json"
    )

def sample_for_spec(spec):
    target = (
        "MIXED"
        if spec["experiment"] == "P1"
        else spec["eval_domain"]
    )

    sample = SAMPLE_REGISTRY[
        (SAMPLE_REGISTRY["experiment"] == spec["experiment"])
        & (SAMPLE_REGISTRY["target_domain"] == target)
    ].copy()

    if len(sample) == 0:
        raise RuntimeError(
            f"❌ Sin muestra para {spec['experiment']} target={target}"
        )

    return sample.sort_values(
        ["y_true", "image_id"]
    ).reset_index(drop=True)


all_metric_frames = []

start_all = time.time()

for run_number, spec in enumerate(
    RUN_SPECS,
    start=1
):
    run_id = run_id_from_spec(spec)
    metrics_path = metrics_file_from_spec(spec)
    maps_path = map_file_from_spec(spec)
    done_path = done_file_from_spec(spec)

    print(
        f"\n[{run_number:03d}/{len(RUN_SPECS)}] "
        f"{run_id}"
    )

    if (
        metrics_path.exists()
        and maps_path.exists()
        and done_path.exists()
    ):
        print("   ↪️ ya completo; se reutiliza cache.")
        all_metric_frames.append(
            pd.read_csv(metrics_path)
        )
        continue

    sample = sample_for_spec(spec)

    train_domain_arg = (
        None
        if spec["experiment"] == "P1"
        else spec["train_domain"]
    )

    model, class_to_idx, idx_to_class, ckpt_path = (
        load_model_checkpoint(
            experiment=spec["experiment"],
            model_key=spec["model_key"],
            seed=spec["seed"],
            train_domain=train_domain_arg,
        )
    )

    explainer = build_explainer(
        model,
        spec["model_key"]
    )

    # Auditoría contra predicciones guardadas.
    pred_path = prediction_path(
        experiment=spec["experiment"],
        model_key=spec["model_key"],
        seed=spec["seed"],
        train_domain=(
            None
            if spec["experiment"] == "P1"
            else spec["train_domain"]
        ),
        eval_domain=(
            None
            if spec["experiment"] == "P1"
            else spec["eval_domain"]
        ),
    )

    stored_predictions = pd.read_csv(
        pred_path
    ).set_index("image_id")

    metric_rows = []
    map_stack = []
    map_image_ids = []
    direct_prediction_mismatches = 0

    for image_number, row in sample.iterrows():
        image_id = str(
            row["image_id"]
        )

        true_label = str(
            row["y_true"]
        )

        if true_label not in class_to_idx:
            raise RuntimeError(
                f"❌ Label {true_label} no existe en checkpoint."
            )

        image_tensor = load_tensor(
            image_id
        )

        explanation = explainer.explain(
            image_tensor=image_tensor,
            target_class=None,
        )

        saliency = explanation[
            "saliency"
        ]

        pred_idx = int(
            explanation[
                "pred_idx"
            ]
        )

        pred_label = idx_to_class[
            pred_idx
        ]

        original_confidence = float(
            explanation[
                "target_probability"
            ]
        )

        stored_label = str(
            stored_predictions.loc[
                image_id,
                "y_pred"
            ]
        )

        if pred_label != stored_label:
            direct_prediction_mismatches += 1

        faith = faithfulness_curves(
            model=model,
            image_tensor=image_tensor,
            saliency=saliency,
            target_class=pred_idx,
        )

        saved_map = resize_saliency(
            saliency,
            MAP_SAVE_SIZE
        ).astype(
            np.float16
        )

        map_stack.append(
            saved_map
        )

        map_image_ids.append(
            image_id
        )

        metric_rows.append({
            "experiment": spec["experiment"],
            "protocol": spec["protocol"],
            "train_domain": spec["train_domain"],
            "eval_domain": spec["eval_domain"],
            "eval_type": spec["eval_type"],
            "model_key": spec["model_key"],
            "model": MODEL_SHORT[
                spec["model_key"]
            ],
            "attribution_method": METHOD_BY_MODEL[
                spec["model_key"]
            ],
            "seed": int(
                spec["seed"]
            ),
            "image_id": image_id,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": int(
                pred_label == true_label
            ),
            "original_confidence": original_confidence,
            "deletion_auc": faith[
                "deletion_auc"
            ],
            "insertion_auc": faith[
                "insertion_auc"
            ],
            "confidence_drop_20": faith[
                "confidence_drop_20"
            ],
        })

    explainer.close()

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    np.savez_compressed(
        maps_path,
        image_ids=np.asarray(
            map_image_ids,
            dtype=str
        ),
        saliency_maps=np.stack(
            map_stack,
            axis=0
        ),
    )

    with open(
        done_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                "run_id": run_id,
                "checkpoint": str(
                    ckpt_path
                ),
                "n_images": int(
                    len(metrics_df)
                ),
                "prediction_mismatches_vs_saved_test": int(
                    direct_prediction_mismatches
                ),
                "completed_at": time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            },
            f,
            indent=2
        )

    if direct_prediction_mismatches > 0:
        warnings.warn(
            f"{run_id}: {direct_prediction_mismatches} "
            "predicciones directas difieren del CSV guardado. "
            "Revise transform/preprocesamiento."
        )

    all_metric_frames.append(
        metrics_df
    )

    del explainer
    del model

    gc.collect()

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

elapsed_minutes = (
    time.time() - start_all
) / 60.0

FAITHFULNESS = pd.concat(
    all_metric_frames,
    ignore_index=True
)

FAITHFULNESS.to_csv(
    RUN_ROOT /
    "10_faithfulness_per_image_all_runs.csv",
    index=False
)

print("=" * 80)
print("✅ 108 BLOQUES PROCESADOS")
print("=" * 80)
print("Filas faithfulness:", len(FAITHFULNESS))
print(f"Tiempo total de este bloque: {elapsed_minutes:.1f} min")

# %% [markdown] Cell 24
# ## 11. Auditoría de consistencia con predicciones guardadas

# %% Cell 25
done_files = list(
    RUN_CACHE.glob(
        "*__DONE.json"
    )
)

if len(done_files) != 108:
    raise RuntimeError(
        f"❌ Se esperaban 108 DONE.json; hay {len(done_files)}."
    )

audit_rows = []

for path in done_files:
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        audit_rows.append(
            json.load(f)
        )

prediction_audit = pd.DataFrame(
    audit_rows
)

prediction_audit.to_csv(
    RUN_ROOT /
    "10_prediction_reproduction_audit.csv",
    index=False
)

total_mismatches = int(
    prediction_audit[
        "prediction_mismatches_vs_saved_test"
    ].sum()
)

print(
    "Predicciones que no reprodujeron el CSV original:",
    total_mismatches
)

if total_mismatches == 0:
    print("✅ Preprocesamiento y checkpoints reproducen las predicciones guardadas.")
else:
    print(
        "⚠️ Hay discrepancias. No invalida automáticamente saliency, "
        "pero debe revisarse antes de usar las figuras en el paper."
    )

# %% [markdown] Cell 26
# ## 12. Resumen de faithfulness

# %% Cell 27
faithfulness_summary = (
    FAITHFULNESS
    .groupby(
        [
            "experiment",
            "protocol",
            "train_domain",
            "eval_domain",
            "eval_type",
            "model_key",
            "model",
            "attribution_method",
        ]
    )
    .agg(
        n_image_seed=("image_id", "count"),
        accuracy_sample=("correct", "mean"),
        original_confidence_mean=(
            "original_confidence",
            "mean"
        ),
        deletion_auc_mean=(
            "deletion_auc",
            "mean"
        ),
        deletion_auc_sd=(
            "deletion_auc",
            "std"
        ),
        insertion_auc_mean=(
            "insertion_auc",
            "mean"
        ),
        insertion_auc_sd=(
            "insertion_auc",
            "std"
        ),
        confidence_drop_20_mean=(
            "confidence_drop_20",
            "mean"
        ),
        confidence_drop_20_sd=(
            "confidence_drop_20",
            "std"
        ),
    )
    .reset_index()
)

faithfulness_summary.to_csv(
    RUN_ROOT /
    "10_faithfulness_summary.csv",
    index=False
)

display(
    faithfulness_summary.round(5)
)

# %% [markdown] Cell 28
# ## 13. Estabilidad de saliency entre seeds

# %% Cell 29
def load_maps_for_spec(spec):
    path = map_file_from_spec(
        spec
    )

    arrays = np.load(
        path,
        allow_pickle=False
    )

    image_ids = arrays[
        "image_ids"
    ].astype(str)

    maps = arrays[
        "saliency_maps"
    ].astype(np.float32)

    return image_ids, maps


seed_stability_rows = []

group_keys = {}

for spec in RUN_SPECS:
    key = (
        spec["experiment"],
        spec["protocol"],
        spec["model_key"],
    )

    group_keys.setdefault(
        key,
        []
    ).append(spec)

for (
    experiment,
    protocol,
    model_key
), specs in group_keys.items():

    specs = sorted(
        specs,
        key=lambda x: x["seed"]
    )

    if len(specs) != 3:
        raise RuntimeError(
            f"❌ {experiment}/{protocol}/{model_key}: "
            f"se esperaban 3 seeds."
        )

    loaded = {}

    for spec in specs:
        ids, maps = load_maps_for_spec(
            spec
        )

        loaded[
            spec["seed"]
        ] = (
            ids,
            maps
        )

    reference_ids = loaded[
        SEEDS[0]
    ][0]

    for seed in SEEDS[1:]:
        if not np.array_equal(
            reference_ids,
            loaded[seed][0]
        ):
            raise RuntimeError(
                f"❌ Orden de imágenes cambia entre seeds: "
                f"{experiment}/{protocol}/{model_key}"
            )

    for image_index, image_id in enumerate(
        reference_ids
    ):
        correlations = []

        for seed_a, seed_b in itertools.combinations(
            SEEDS,
            2
        ):
            map_a = loaded[
                seed_a
            ][1][image_index].reshape(-1)

            map_b = loaded[
                seed_b
            ][1][image_index].reshape(-1)

            rho, _ = spearmanr(
                map_a,
                map_b
            )

            if np.isnan(rho):
                rho = 0.0

            correlations.append(
                float(rho)
            )

        seed_stability_rows.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_key": model_key,
            "model": MODEL_SHORT[
                model_key
            ],
            "image_id": image_id,
            "mean_pairwise_spearman": float(
                np.mean(
                    correlations
                )
            ),
            "min_pairwise_spearman": float(
                np.min(
                    correlations
                )
            ),
        })


SEED_STABILITY = pd.DataFrame(
    seed_stability_rows
)

SEED_STABILITY.to_csv(
    RUN_ROOT /
    "10_cross_seed_saliency_stability_per_image.csv",
    index=False
)

seed_stability_summary = (
    SEED_STABILITY
    .groupby(
        [
            "experiment",
            "protocol",
            "model_key",
            "model",
        ]
    )
    .agg(
        n_images=("image_id", "nunique"),
        mean_spearman=(
            "mean_pairwise_spearman",
            "mean"
        ),
        sd_spearman=(
            "mean_pairwise_spearman",
            "std"
        ),
        mean_min_spearman=(
            "min_pairwise_spearman",
            "mean"
        ),
    )
    .reset_index()
)

seed_stability_summary.to_csv(
    RUN_ROOT /
    "10_cross_seed_saliency_stability_summary.csv",
    index=False
)

display(
    seed_stability_summary.round(5)
)

# %% [markdown] Cell 30
# ## 14. Cambio de saliency entre control y cross-domain sobre las mismas imágenes

# %% Cell 31
def find_spec(
    experiment,
    train_domain,
    eval_domain,
    model_key,
    seed,
):
    matches = [
        spec
        for spec in RUN_SPECS
        if (
            spec["experiment"] == experiment
            and spec["train_domain"] == train_domain
            and spec["eval_domain"] == eval_domain
            and spec["model_key"] == model_key
            and spec["seed"] == seed
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"❌ Spec no único: {experiment}, {train_domain}, "
            f"{eval_domain}, {model_key}, {seed}"
        )

    return matches[0]


control_cross_saliency_rows = []

for experiment in ["P4", "P5"]:
    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        for model_key in MODELS:
            for seed in SEEDS:
                control_spec = find_spec(
                    experiment,
                    control_source,
                    target_domain,
                    model_key,
                    seed,
                )

                cross_spec = find_spec(
                    experiment,
                    cross_source,
                    target_domain,
                    model_key,
                    seed,
                )

                control_ids, control_maps = load_maps_for_spec(
                    control_spec
                )

                cross_ids, cross_maps = load_maps_for_spec(
                    cross_spec
                )

                if not np.array_equal(
                    control_ids,
                    cross_ids
                ):
                    raise RuntimeError(
                        f"❌ Control/cross no usa las mismas imágenes: "
                        f"{experiment}, target={target_domain}, "
                        f"{model_key}, seed={seed}"
                    )

                for idx, image_id in enumerate(
                    control_ids
                ):
                    rho, _ = spearmanr(
                        control_maps[idx].reshape(-1),
                        cross_maps[idx].reshape(-1),
                    )

                    if np.isnan(rho):
                        rho = 0.0

                    control_cross_saliency_rows.append({
                        "experiment": experiment,
                        "target_domain": target_domain,
                        "model_key": model_key,
                        "model": MODEL_SHORT[
                            model_key
                        ],
                        "seed": int(
                            seed
                        ),
                        "image_id": str(
                            image_id
                        ),
                        "control_train_domain": control_source,
                        "cross_train_domain": cross_source,
                        "saliency_spearman_control_vs_cross": float(
                            rho
                        ),
                    })


CONTROL_CROSS_SALIENCY = pd.DataFrame(
    control_cross_saliency_rows
)

CONTROL_CROSS_SALIENCY.to_csv(
    RUN_ROOT /
    "10_control_vs_cross_saliency_similarity_per_image.csv",
    index=False
)

control_cross_saliency_summary = (
    CONTROL_CROSS_SALIENCY
    .groupby(
        [
            "experiment",
            "target_domain",
            "model_key",
            "model",
        ]
    )
    .agg(
        n_image_seed=("image_id", "count"),
        mean_spearman=(
            "saliency_spearman_control_vs_cross",
            "mean"
        ),
        sd_spearman=(
            "saliency_spearman_control_vs_cross",
            "std"
        ),
        median_spearman=(
            "saliency_spearman_control_vs_cross",
            "median"
        ),
    )
    .reset_index()
)

control_cross_saliency_summary.to_csv(
    RUN_ROOT /
    "10_control_vs_cross_saliency_similarity_summary.csv",
    index=False
)

display(
    control_cross_saliency_summary.round(5)
)

# %% [markdown] Cell 32
# ## 15. Bootstrap pareado de faithfulness: control vs cross-domain
# 
# El objetivo aquí no es comparar Grad-CAM contra Attention Rollout como si fueran el mismo algoritmo.  
# La comparación principal se hace **dentro de cada arquitectura**, usando las mismas imágenes target.
# 
# Para cada imagen se promedian primero las tres seeds y luego se hace bootstrap pareado por `image_id`.
# 
# Interpretación de la diferencia `cross − control`:
# 
# - **Deletion AUC:** positivo = peor faithfulness bajo shift.
# - **Insertion AUC:** negativo = peor faithfulness bajo shift.
# - **Confidence drop @20%:** negativo = menor sensibilidad a regiones atribuidas bajo shift.

# %% Cell 33
def paired_bootstrap_mean_delta(
    deltas,
    n_boot=N_BOOT,
    seed=BOOT_SEED,
):
    deltas = np.asarray(
        deltas,
        dtype=float
    )

    rng = np.random.default_rng(
        seed
    )

    boot = np.empty(
        n_boot,
        dtype=float
    )

    n = len(deltas)

    for b in range(n_boot):
        idx = rng.integers(
            0,
            n,
            size=n
        )

        boot[b] = float(
            np.mean(
                deltas[idx]
            )
        )

    lo, hi = np.quantile(
        boot,
        [0.025, 0.975]
    )

    return {
        "estimate": float(
            np.mean(
                deltas
            )
        ),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "probability_delta_gt_0": float(
            np.mean(
                boot > 0
            )
        ),
    }


faith_boot_rows = []

metrics_to_compare = [
    "deletion_auc",
    "insertion_auc",
    "confidence_drop_20",
]

for experiment in ["P4", "P5"]:
    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        for model_idx, model_key in enumerate(
            MODELS
        ):
            control = (
                FAITHFULNESS[
                    (FAITHFULNESS["experiment"] == experiment)
                    & (FAITHFULNESS["train_domain"] == control_source)
                    & (FAITHFULNESS["eval_domain"] == target_domain)
                    & (FAITHFULNESS["model_key"] == model_key)
                ]
                .groupby(
                    "image_id"
                )[metrics_to_compare]
                .mean()
                .reset_index()
            )

            cross = (
                FAITHFULNESS[
                    (FAITHFULNESS["experiment"] == experiment)
                    & (FAITHFULNESS["train_domain"] == cross_source)
                    & (FAITHFULNESS["eval_domain"] == target_domain)
                    & (FAITHFULNESS["model_key"] == model_key)
                ]
                .groupby(
                    "image_id"
                )[metrics_to_compare]
                .mean()
                .reset_index()
            )

            paired = control.merge(
                cross,
                on="image_id",
                suffixes=(
                    "_control",
                    "_cross"
                ),
                validate="one_to_one"
            )

            for metric_idx, metric in enumerate(
                metrics_to_compare
            ):
                deltas = (
                    paired[
                        f"{metric}_cross"
                    ]
                    - paired[
                        f"{metric}_control"
                    ]
                ).to_numpy()

                result = paired_bootstrap_mean_delta(
                    deltas,
                    n_boot=N_BOOT,
                    seed=(
                        BOOT_SEED
                        + model_idx * 100
                        + metric_idx * 10
                        + (
                            1000
                            if experiment == "P5"
                            else 0
                        )
                        + (
                            500
                            if target_domain == "OTHER"
                            else 0
                        )
                    )
                )

                expected_worse_direction = {
                    "deletion_auc": "positive",
                    "insertion_auc": "negative",
                    "confidence_drop_20": "negative",
                }[metric]

                faith_boot_rows.append({
                    "experiment": experiment,
                    "target_domain": target_domain,
                    "model_key": model_key,
                    "model": MODEL_SHORT[
                        model_key
                    ],
                    "attribution_method": METHOD_BY_MODEL[
                        model_key
                    ],
                    "metric": metric,
                    "n_images": int(
                        len(paired)
                    ),
                    "delta_cross_minus_control": result[
                        "estimate"
                    ],
                    "ci95_low": result[
                        "ci95_low"
                    ],
                    "ci95_high": result[
                        "ci95_high"
                    ],
                    "bootstrap_probability_delta_gt_0": result[
                        "probability_delta_gt_0"
                    ],
                    "ci_excludes_zero": bool(
                        (
                            result["ci95_low"] > 0
                        )
                        or (
                            result["ci95_high"] < 0
                        )
                    ),
                    "direction_consistent_with_worse_faithfulness": (
                        (
                            expected_worse_direction == "positive"
                            and result["estimate"] > 0
                        )
                        or (
                            expected_worse_direction == "negative"
                            and result["estimate"] < 0
                        )
                    ),
                    "expected_worse_direction": expected_worse_direction,
                    "n_boot": int(
                        N_BOOT
                    ),
                })


FAITHFULNESS_BOOTSTRAP = pd.DataFrame(
    faith_boot_rows
)

FAITHFULNESS_BOOTSTRAP.to_csv(
    RUN_ROOT /
    "10_control_vs_cross_faithfulness_paired_bootstrap.csv",
    index=False
)

display(
    FAITHFULNESS_BOOTSTRAP.round(5)
)

# %% [markdown] Cell 34
# ## 16. Figuras cualitativas pareadas: misma imagen, control vs cross

# %% Cell 35
def overlay_saliency(
    axis,
    image_tensor,
    saliency,
    title,
):
    image = denormalize_tensor(
        image_tensor
    ).permute(
        1,
        2,
        0
    ).numpy()

    axis.imshow(
        image
    )

    axis.imshow(
        saliency,
        alpha=0.45
    )

    axis.set_title(
        title,
        fontsize=9
    )

    axis.axis(
        "off"
    )


qualitative_files = []

for experiment in ["P4", "P5"]:
    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        for model_key in MODELS:
            control_spec = find_spec(
                experiment,
                control_source,
                target_domain,
                model_key,
                QUALITATIVE_SEED,
            )

            cross_spec = find_spec(
                experiment,
                cross_source,
                target_domain,
                model_key,
                QUALITATIVE_SEED,
            )

            control_ids, control_maps = load_maps_for_spec(
                control_spec
            )

            cross_ids, cross_maps = load_maps_for_spec(
                cross_spec
            )

            if not np.array_equal(
                control_ids,
                cross_ids
            ):
                raise RuntimeError(
                    "❌ IDs no apareados para figura."
                )

            sample_registry = SAMPLE_REGISTRY[
                (SAMPLE_REGISTRY["experiment"] == experiment)
                & (
                    SAMPLE_REGISTRY["target_domain"]
                    == target_domain
                )
            ].copy()

            label_lookup = (
                sample_registry
                .set_index(
                    "image_id"
                )["y_true"]
                .astype(str)
                .to_dict()
            )

            chosen_ids = list(
                control_ids[
                    :QUALITATIVE_IMAGES_PER_FIGURE
                ]
            )

            fig, axes = plt.subplots(
                len(chosen_ids),
                3,
                figsize=(
                    10,
                    3 * len(chosen_ids)
                ),
                squeeze=False
            )

            for row_idx, image_id in enumerate(
                chosen_ids
            ):
                image_tensor = load_tensor(
                    image_id
                )

                map_idx = int(
                    np.where(
                        control_ids == image_id
                    )[0][0]
                )

                original = denormalize_tensor(
                    image_tensor
                ).permute(
                    1,
                    2,
                    0
                ).numpy()

                axes[
                    row_idx,
                    0
                ].imshow(
                    original
                )

                axes[
                    row_idx,
                    0
                ].set_title(
                    f"{image_id}\ntrue={label_lookup[str(image_id)]}",
                    fontsize=9
                )

                axes[
                    row_idx,
                    0
                ].axis(
                    "off"
                )

                overlay_saliency(
                    axes[
                        row_idx,
                        1
                    ],
                    image_tensor,
                    control_maps[
                        map_idx
                    ],
                    f"Control train={control_source}",
                )

                overlay_saliency(
                    axes[
                        row_idx,
                        2
                    ],
                    image_tensor,
                    cross_maps[
                        map_idx
                    ],
                    f"Cross train={cross_source}",
                )

            fig.suptitle(
                f"{experiment} | target={target_domain} | "
                f"{MODEL_SHORT[model_key]} | "
                f"{METHOD_BY_MODEL[model_key]} | seed {QUALITATIVE_SEED}",
                fontsize=11
            )

            plt.tight_layout()

            figure_path = (
                FIGURE_ROOT /
                f"10_pair__{experiment}__target_{target_domain}__"
                f"{model_key}.png"
            )

            plt.savefig(
                figure_path,
                dpi=250,
                bbox_inches="tight"
            )

            plt.close(
                fig
            )

            qualitative_files.append(
                str(
                    figure_path
                )
            )

print(
    "✅ Figuras control-vs-cross:",
    len(qualitative_files)
)

# %% [markdown] Cell 36
# ## 17. Figuras IID de referencia

# %% Cell 37
p1_figures = []

for model_key in MODELS:
    spec = find_spec(
        "P1",
        "MIXED",
        "MIXED",
        model_key,
        QUALITATIVE_SEED,
    )

    image_ids, maps = load_maps_for_spec(
        spec
    )

    chosen_ids = list(
        image_ids[
            :QUALITATIVE_IMAGES_PER_FIGURE
        ]
    )

    fig, axes = plt.subplots(
        len(chosen_ids),
        2,
        figsize=(
            7,
            3 * len(chosen_ids)
        ),
        squeeze=False
    )

    for row_idx, image_id in enumerate(
        chosen_ids
    ):
        image_tensor = load_tensor(
            image_id
        )

        original = denormalize_tensor(
            image_tensor
        ).permute(
            1,
            2,
            0
        ).numpy()

        axes[
            row_idx,
            0
        ].imshow(
            original
        )

        axes[
            row_idx,
            0
        ].set_title(
            str(
                image_id
            ),
            fontsize=9
        )

        axes[
            row_idx,
            0
        ].axis(
            "off"
        )

        overlay_saliency(
            axes[
                row_idx,
                1
            ],
            image_tensor,
            maps[
                row_idx
            ],
            METHOD_BY_MODEL[
                model_key
            ],
        )

    fig.suptitle(
        f"P1 IID | {MODEL_SHORT[model_key]} | "
        f"seed {QUALITATIVE_SEED}",
        fontsize=11
    )

    plt.tight_layout()

    path = (
        FIGURE_ROOT /
        f"10_P1_IID__{model_key}.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )

    p1_figures.append(
        str(
            path
        )
    )

print(
    "✅ Figuras IID:",
    len(p1_figures)
)

# %% [markdown] Cell 38
# ## 18. Figuras cuantitativas de faithfulness

# %% Cell 39
for experiment in ["P4", "P5"]:
    for metric in [
        "deletion_auc",
        "insertion_auc",
        "confidence_drop_20",
    ]:
        sub = FAITHFULNESS_BOOTSTRAP[
            (FAITHFULNESS_BOOTSTRAP["experiment"] == experiment)
            & (FAITHFULNESS_BOOTSTRAP["metric"] == metric)
        ].copy()

        sub["label"] = (
            sub["model"]
            + " | target "
            + sub["target_domain"]
        )

        y = np.arange(
            len(sub)
        )

        x = sub[
            "delta_cross_minus_control"
        ].to_numpy()

        xerr = np.vstack([
            x
            - sub[
                "ci95_low"
            ].to_numpy(),
            sub[
                "ci95_high"
            ].to_numpy()
            - x,
        ])

        fig = plt.figure(
            figsize=(
                10,
                6
            )
        )

        plt.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            capsize=4
        )

        plt.axvline(
            0.0,
            linestyle="--"
        )

        plt.yticks(
            y,
            sub[
                "label"
            ]
        )

        plt.xlabel(
            f"Cross − control: {metric}"
        )

        plt.ylabel(
            "Architecture / target"
        )

        plt.title(
            f"{experiment} — paired faithfulness shift"
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_ROOT /
            f"10_{experiment}__{metric}__cross_minus_control.png",
            dpi=250,
            bbox_inches="tight"
        )

        plt.close(
            fig
        )

print("✅ Figuras cuantitativas guardadas.")

# %% [markdown] Cell 40
# ## 19. Tabla integrada de explainability

# %% Cell 41
integrated_rows = []

for _, row in faithfulness_summary.iterrows():
    stability = seed_stability_summary[
        (seed_stability_summary["experiment"] == row["experiment"])
        & (seed_stability_summary["protocol"] == row["protocol"])
        & (seed_stability_summary["model_key"] == row["model_key"])
    ]

    integrated = row.to_dict()

    if len(stability) == 1:
        integrated[
            "cross_seed_saliency_spearman"
        ] = float(
            stability.iloc[0][
                "mean_spearman"
            ]
        )
    else:
        integrated[
            "cross_seed_saliency_spearman"
        ] = np.nan

    integrated_rows.append(
        integrated
    )

INTEGRATED_EXPLAINABILITY = pd.DataFrame(
    integrated_rows
)

INTEGRATED_EXPLAINABILITY.to_csv(
    RUN_ROOT /
    "10_integrated_explainability_summary.csv",
    index=False
)

display(
    INTEGRATED_EXPLAINABILITY[
        [
            "experiment",
            "protocol",
            "model",
            "attribution_method",
            "deletion_auc_mean",
            "insertion_auc_mean",
            "confidence_drop_20_mean",
            "cross_seed_saliency_spearman",
        ]
    ].round(5)
)

# %% [markdown] Cell 42
# ## 20. Nota metodológica reproducible

# %% Cell 43
methodology_note = f'''
PADDY DOCTOR Q1 — NOTEBOOK 10
EXPLAINABILITY, FAITHFULNESS AND STABILITY

Experiments:
- P1 clean IID benchmark.
- P4 cross-cultivar benchmark, 10 classes.
- P5 exact disease-age matched sensitivity benchmark, 5 classes.

Architectures and attribution:
- ResNet-50: Grad-CAM.
- ConvNeXt-Tiny: Grad-CAM.
- ViT-S/16: Attention Rollout.
- Swin-Tiny: Grad-CAM adapted to hierarchical/channel-last activations.

Sampling:
- Deterministic SHA-256 ordering within each true class.
- Sampling does not use predictions, confidence, correctness or test performance.
- P1: {P1_PER_CLASS} images per class.
- P4: {P4_PER_CLASS} images per class and target domain.
- P5: {P5_PER_CLASS} images per class and target domain.
- Matched-domain and cross-domain explanations use the same target images.

Faithfulness:
- Attribution maps are ranked on a {PATCH_GRID}x{PATCH_GRID} patch grid.
- Deletion uses the ImageNet-mean image (zero in normalized space) as baseline.
- Insertion starts from the same baseline.
- Fractions: {PERTURB_FRACTIONS.tolist()}.
- Target is the class predicted by the unperturbed model.
- Deletion AUC: lower is more faithful.
- Insertion AUC: higher is more faithful.
- Confidence drop at {int(CONFIDENCE_DROP_FRACTION*100)}% deletion: higher indicates
  stronger dependence on regions ranked as important.

Stability:
- Cross-seed stability uses pairwise Spearman correlation of compact saliency maps
  for seeds {SEEDS}.
- Control-vs-cross saliency similarity compares explanations from differently
  trained models on exactly the same target image.

Paired faithfulness sensitivity:
- For P4/P5, metrics are first averaged across the three seeds for each image.
- Control-vs-cross differences are then bootstrapped at image level.
- Bootstrap replicates: {N_BOOT}.
- These explanation analyses are complementary to the primary predictive
  inference in Notebook 08.

Important limitations:
- The dataset contains no lesion masks or expert localization annotations.
- No IoU, Dice or clinical localization claim is made.
- Grad-CAM and Attention Rollout are different attribution families, so absolute
  faithfulness values should not be overinterpreted as direct method rankings
  across architectures.
- The strongest comparisons are within architecture across IID/control/cross
  conditions.
- Attention Rollout is not class-specific in the same sense as Grad-CAM; its
  faithfulness is evaluated against the originally predicted class.
'''

with open(
    RUN_ROOT /
    "10_explainability_methodology_notes.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        methodology_note
    )

print(
    methodology_note
)

# %% [markdown] Cell 44
# ## 21. Verificación final

# %% Cell 45
required_outputs = [
    "10_config.json",
    "10_explanation_sample_registry.csv",
    "10_faithfulness_per_image_all_runs.csv",
    "10_prediction_reproduction_audit.csv",
    "10_faithfulness_summary.csv",
    "10_cross_seed_saliency_stability_per_image.csv",
    "10_cross_seed_saliency_stability_summary.csv",
    "10_control_vs_cross_saliency_similarity_per_image.csv",
    "10_control_vs_cross_saliency_similarity_summary.csv",
    "10_control_vs_cross_faithfulness_paired_bootstrap.csv",
    "10_integrated_explainability_summary.csv",
    "10_explainability_methodology_notes.txt",
]

missing = [
    name
    for name in required_outputs
    if not (
        RUN_ROOT /
        name
    ).exists()
]

if missing:
    raise RuntimeError(
        f"❌ Faltan outputs N10: {missing}"
    )

expected_faithfulness_rows = 1920

if len(FAITHFULNESS) != expected_faithfulness_rows:
    raise RuntimeError(
        f"❌ Faithfulness: esperado {expected_faithfulness_rows}, "
        f"observado {len(FAITHFULNESS)}."
    )

if len(list(MAP_ROOT.glob("*.npz"))) != 108:
    raise RuntimeError(
        "❌ Se esperaban 108 archivos de mapas."
    )

if len(list(RUN_CACHE.glob("*__DONE.json"))) != 108:
    raise RuntimeError(
        "❌ Se esperaban 108 DONE.json."
    )

if len(SEED_STABILITY) != 640:
    raise RuntimeError(
        f"❌ Seed stability: esperado 640, observado {len(SEED_STABILITY)}."
    )

if len(CONTROL_CROSS_SALIENCY) != 840:
    raise RuntimeError(
        f"❌ Control-cross saliency: esperado 840, "
        f"observado {len(CONTROL_CROSS_SALIENCY)}."
    )

if len(FAITHFULNESS_BOOTSTRAP) != 48:
    raise RuntimeError(
        f"❌ Bootstrap: esperado 48, observado {len(FAITHFULNESS_BOOTSTRAP)}."
    )

figure_count = len(
    list(
        FIGURE_ROOT.glob(
            "*.png"
        )
    )
)

print("=" * 80)
print("✅ NOTEBOOK 10 COMPLETADO")
print("=" * 80)
print("Faithfulness image-seed rows:", len(FAITHFULNESS))
print("Saliency maps:", len(list(MAP_ROOT.glob('*.npz'))))
print("Cross-seed stability rows:", len(SEED_STABILITY))
print("Control-vs-cross saliency rows:", len(CONTROL_CROSS_SALIENCY))
print("Paired bootstrap contrasts:", len(FAITHFULNESS_BOOTSTRAP))
print("Figuras:", figure_count)
print("Predicción mismatches:", total_mismatches)

# %% [markdown] Cell 46
# ## 22. Exportar resultados del Notebook 10

# %% Cell 47
from google.colab import files

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

package_name = (
    "10_PaddyDoctor_Explainability_Faithfulness_"
    f"RESULTS_{timestamp}"
)

temp_package = (
    Path("/content") /
    package_name
)

if temp_package.exists():
    shutil.rmtree(
        temp_package
    )

(temp_package / "results").mkdir(
    parents=True,
    exist_ok=True
)

(temp_package / "figures").mkdir(
    parents=True,
    exist_ok=True
)

# Exportar tablas finales, pero no los 108 mapas/caches internos.
for path in RUN_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(
            path,
            temp_package /
            "results" /
            path.name
        )

for path in FIGURE_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(
            path,
            temp_package /
            "figures" /
            path.name
        )

with open(
    temp_package /
    "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 10\n"
        "Explainability, Faithfulness and Stability\n\n"
        "No model training is performed.\n"
        "ResNet/ConvNeXt: Grad-CAM.\n"
        "ViT: Attention Rollout.\n"
        "Swin: adapted Grad-CAM.\n"
        "Faithfulness: deletion AUC, insertion AUC, confidence drop @20%.\n"
        "Stability: cross-seed saliency Spearman correlation.\n"
        "Control-vs-cross saliency similarity included.\n"
        "No lesion masks are available; no localization IoU/Dice claim is made.\n"
    )

zip_base = (
    EXPORTS_DIR /
    package_name
)

zip_path = shutil.make_archive(
    str(
        zip_base
    ),
    "zip",
    root_dir=temp_package
)

size_mb = (
    Path(
        zip_path
    ).stat().st_size
    / (
        1024 ** 2
    )
)

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print(zip_path)
print(f"Tamaño: {size_mb:.2f} MB")

files.download(
    zip_path
)

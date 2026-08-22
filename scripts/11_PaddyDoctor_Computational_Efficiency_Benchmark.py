"""Auto-exported from 11_PaddyDoctor_Computational_Efficiency_Benchmark.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 11
# ## Eficiencia computacional: parámetros, FLOPs, latencia, throughput y VRAM
# 
# Este notebook **no entrena modelos**. Reutiliza un checkpoint representativo de P1 (seed 2026) por arquitectura para medir:
# 
# - parámetros totales y entrenables;
# - tamaño del checkpoint;
# - GFLOPs por imagen con `fvcore`;
# - latencia y p95;
# - throughput;
# - VRAM pico;
# - comparación FP32 vs AMP/FP16;
# - batch sizes 1, 8 y 32.
# 
# ### Protocolo
# - Entrada sintética `3×224×224` para aislar el costo del modelo de I/O/JPEG/DataLoader.
# - 20 iteraciones de warm-up.
# - 60 iteraciones cronometradas.
# - `model.eval()` + `torch.inference_mode()`.
# - Sin `torch.compile`.
# - CUDA sincronizada en cada medición.
# 
# > Para la tabla del artículo, ejecútalo en la misma **GPU L4** usada en los experimentos.

# %% [markdown] Cell 2
# ## 0. Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path
import os, time

try:
    drive.mount("/content/drive", force_remount=False)
except Exception:
    drive.mount("/content/drive", force_remount=True)

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
RESULTS_DIR = PROJECT / "results"
MODELS_DIR = PROJECT / "models"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"

P1_MODELS = MODELS_DIR / "05_P1_30ep_noES"
RUN_ROOT = RESULTS_DIR / "11_computational_efficiency"
FIGURE_ROOT = FIGURES_DIR / "11_computational_efficiency"

for p in [RUN_ROOT, FIGURE_ROOT, EXPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("✅ Proyecto:", PROJECT)

# %% [markdown] Cell 4
# ## 1. Dependencias y hardware

# %% Cell 5
import sys, subprocess, json, gc, platform, shutil
from contextlib import nullcontext

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "timm==1.0.28", "fvcore"],
    check=True
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import timm

from fvcore.nn import FlopCountAnalysis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type != "cuda":
    raise RuntimeError(
        "❌ Notebook 11 requiere GPU CUDA para medir latencia y VRAM. "
        "En Colab selecciona GPU L4."
    )

GPU_NAME = torch.cuda.get_device_name(0)
torch.backends.cudnn.benchmark = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("GPU:", GPU_NAME)
print("CUDA:", torch.version.cuda)
print("PyTorch:", torch.__version__)
print("timm:", timm.__version__)
print("Python:", platform.python_version())

if "L4" not in GPU_NAME.upper():
    print("⚠️ La GPU no parece ser L4. No mezcles estos tiempos con otra GPU.")

# %% [markdown] Cell 6
# ## 2. Configuración

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

REPRESENTATIVE_SEED = 2026
IMAGE_SIZE = 224
BATCH_SIZES = [1, 8, 32]
PRECISIONS = ["fp32", "amp_fp16"]
WARMUP_ITERS = 20
TIMED_ITERS = 60
RANDOM_SEED = 20260822

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CONFIG = {
    "notebook": "11_computational_efficiency",
    "models": MODELS,
    "representative_seed": REPRESENTATIVE_SEED,
    "image_size": IMAGE_SIZE,
    "batch_sizes": BATCH_SIZES,
    "precisions": PRECISIONS,
    "warmup_iters": WARMUP_ITERS,
    "timed_iters": TIMED_ITERS,
    "gpu": GPU_NAME,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "timm": timm.__version__,
}

with open(RUN_ROOT / "11_benchmark_config.json", "w", encoding="utf-8") as f:
    json.dump(CONFIG, f, indent=2)

print(json.dumps(CONFIG, indent=2))

# %% [markdown] Cell 8
# ## 3. Checkpoints representativos

# %% Cell 9
CHECKPOINTS = {}

for model_key in MODELS:
    path = (
        P1_MODELS /
        f"{model_key}__seed_{REPRESENTATIVE_SEED}" /
        "best_model.pt"
    )

    if not path.exists():
        raise FileNotFoundError(f"❌ Falta: {path}")

    CHECKPOINTS[model_key] = path
    print(
        MODEL_SHORT[model_key],
        f"| checkpoint={path.stat().st_size / (1024**2):.2f} MB"
    )

print("✅ 4 checkpoints encontrados.")

# %% [markdown] Cell 10
# ## 4. Cargar modelo entrenado

# %% Cell 11
def load_model(model_key):
    path = CHECKPOINTS[model_key]

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    class_to_idx = {
        str(k): int(v)
        for k, v in ckpt["class_to_idx"].items()
    }

    timm_name = ckpt.get("timm_name", model_key)

    model = timm.create_model(
        timm_name,
        pretrained=False,
        num_classes=len(class_to_idx)
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt, len(class_to_idx), timm_name, path

# %% [markdown] Cell 12
# ## 5. Parámetros y FLOPs

# %% Cell 13
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)

def estimate_flops(model):
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    try:
        analysis = FlopCountAnalysis(model, x)

        try:
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)
            analysis.tracer_warnings("none")
        except Exception:
            pass

        total = float(analysis.total())

        try:
            unsupported = {
                str(k): int(v)
                for k, v in analysis.unsupported_ops().items()
            }
        except Exception:
            unsupported = {}

        return total, unsupported, ""

    except Exception as e:
        return np.nan, {}, repr(e)

rows = []

for model_key in MODELS:
    print("Analizando:", MODEL_SHORT[model_key])

    model, ckpt, n_classes, timm_name, path = load_model(model_key)
    total_params, trainable_params = count_parameters(model)
    flops, unsupported, flop_error = estimate_flops(model)

    rows.append({
        "model_key": model_key,
        "model": MODEL_SHORT[model_key],
        "timm_name": timm_name,
        "n_classes": n_classes,
        "parameters_total": total_params,
        "parameters_trainable": trainable_params,
        "parameters_millions": total_params / 1e6,
        "checkpoint_size_mb": path.stat().st_size / (1024**2),
        "flops_total": flops,
        "gflops_fvcore": flops / 1e9 if np.isfinite(flops) else np.nan,
        "unsupported_ops_json": json.dumps(unsupported, sort_keys=True),
        "flop_error": flop_error,
    })

    del model, ckpt
    gc.collect()

COMPLEXITY = pd.DataFrame(rows)
COMPLEXITY.to_csv(RUN_ROOT / "11_model_complexity.csv", index=False)

display(
    COMPLEXITY[
        ["model", "parameters_millions", "checkpoint_size_mb",
         "gflops_fvcore", "unsupported_ops_json"]
    ].round(4)
)

# %% [markdown] Cell 14
# ## 6. Benchmark CUDA

# %% Cell 15
def precision_context(precision):
    if precision == "fp32":
        return nullcontext()

    if precision == "amp_fp16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )

    raise ValueError(precision)

@torch.inference_mode()
def forward_once(model, x, precision):
    with precision_context(precision):
        return model(x)

def benchmark_setting(model, batch_size, precision):
    x = torch.randn(
        batch_size, 3, IMAGE_SIZE, IMAGE_SIZE,
        device=DEVICE, dtype=torch.float32
    )

    # Warm-up.
    for _ in range(WARMUP_ITERS):
        _ = forward_once(model, x, precision)

    torch.cuda.synchronize()

    # Se resetea el pico manteniendo el modelo y el input ya asignados.
    torch.cuda.reset_peak_memory_stats()
    baseline_mb = torch.cuda.memory_allocated() / (1024**2)

    times = []

    for _ in range(TIMED_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = forward_once(model, x, precision)
        end.record()

        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))

    t = np.asarray(times, dtype=float)

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    mean_batch_ms = float(t.mean())

    result = {
        "batch_size": int(batch_size),
        "precision": precision,
        "warmup_iters": int(WARMUP_ITERS),
        "timed_iters": int(TIMED_ITERS),
        "batch_latency_mean_ms": mean_batch_ms,
        "batch_latency_sd_ms": float(t.std(ddof=1)),
        "batch_latency_median_ms": float(np.median(t)),
        "batch_latency_p95_ms": float(np.quantile(t, 0.95)),
        "latency_mean_ms_per_image": mean_batch_ms / batch_size,
        "throughput_images_per_s": batch_size * 1000.0 / mean_batch_ms,
        "vram_baseline_model_plus_input_mb": baseline_mb,
        "vram_peak_total_mb": peak_mb,
        "vram_incremental_peak_mb": peak_mb - baseline_mb,
    }

    del x
    gc.collect()
    torch.cuda.empty_cache()

    return result

# %% [markdown] Cell 16
# ## 7. Ejecutar 24 configuraciones

# %% Cell 17
benchmark_rows = []
t0 = time.time()

for idx, model_key in enumerate(MODELS, start=1):
    print("\n" + "="*80)
    print(f"MODELO {idx}/4 | {MODEL_SHORT[model_key]}")
    print("="*80)

    model, ckpt, n_classes, timm_name, path = load_model(model_key)
    model = model.to(DEVICE)
    model.eval()

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    for precision in PRECISIONS:
        for batch_size in BATCH_SIZES:
            print(f"  {precision} | batch={batch_size}")

            r = benchmark_setting(
                model=model,
                batch_size=batch_size,
                precision=precision
            )

            benchmark_rows.append({
                "model_key": model_key,
                "model": MODEL_SHORT[model_key],
                "timm_name": timm_name,
                "gpu": GPU_NAME,
                **r,
            })

            print(
                f"    {r['latency_mean_ms_per_image']:.3f} ms/img | "
                f"{r['throughput_images_per_s']:.1f} img/s | "
                f"VRAM peak {r['vram_peak_total_mb']:.1f} MB"
            )

    del model, ckpt
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

BENCHMARK = pd.DataFrame(benchmark_rows)

if len(BENCHMARK) != 24:
    raise RuntimeError(f"❌ Se esperaban 24 filas; hay {len(BENCHMARK)}.")

BENCHMARK.to_csv(
    RUN_ROOT / "11_latency_throughput_vram_all_settings.csv",
    index=False
)

print(f"✅ Benchmark completo en {(time.time()-t0)/60:.1f} min")
display(BENCHMARK.round(4))

# %% [markdown] Cell 18
# ## 8. Tabla principal del artículo
# 
# Se fija un escenario claro:
# 
# - **latencia individual:** batch 1, AMP/FP16;
# - **throughput:** batch 32, AMP/FP16;
# - **VRAM:** batch 32, AMP/FP16.
# 
# El CSV completo conserva FP32 y batches 1/8/32.

# %% Cell 19
latency = (
    BENCHMARK[
        (BENCHMARK["precision"] == "amp_fp16") &
        (BENCHMARK["batch_size"] == 1)
    ][
        ["model_key", "latency_mean_ms_per_image", "batch_latency_p95_ms"]
    ]
    .rename(columns={
        "latency_mean_ms_per_image": "latency_amp_batch1_ms_per_image",
        "batch_latency_p95_ms": "latency_amp_batch1_p95_ms",
    })
)

throughput = (
    BENCHMARK[
        (BENCHMARK["precision"] == "amp_fp16") &
        (BENCHMARK["batch_size"] == 32)
    ][
        ["model_key", "throughput_images_per_s",
         "vram_peak_total_mb", "vram_incremental_peak_mb"]
    ]
    .rename(columns={
        "throughput_images_per_s": "throughput_amp_batch32_images_per_s",
        "vram_peak_total_mb": "vram_amp_batch32_peak_total_mb",
        "vram_incremental_peak_mb": "vram_amp_batch32_incremental_peak_mb",
    })
)

SUMMARY = (
    COMPLEXITY
    .merge(latency, on="model_key", validate="one_to_one")
    .merge(throughput, on="model_key", validate="one_to_one")
)

SUMMARY["images_per_second_per_gflop"] = (
    SUMMARY["throughput_amp_batch32_images_per_s"] /
    SUMMARY["gflops_fvcore"]
)

SUMMARY.to_csv(
    RUN_ROOT / "11_efficiency_primary_summary.csv",
    index=False
)

display(
    SUMMARY[
        ["model", "parameters_millions", "gflops_fvcore",
         "checkpoint_size_mb", "latency_amp_batch1_ms_per_image",
         "latency_amp_batch1_p95_ms",
         "throughput_amp_batch32_images_per_s",
         "vram_amp_batch32_peak_total_mb"]
    ].round(3)
)

# %% [markdown] Cell 20
# ## 9. AMP vs FP32

# %% Cell 21
amp_rows = []

for model_key in MODELS:
    for batch_size in BATCH_SIZES:
        f = BENCHMARK[
            (BENCHMARK["model_key"] == model_key) &
            (BENCHMARK["batch_size"] == batch_size) &
            (BENCHMARK["precision"] == "fp32")
        ].iloc[0]

        a = BENCHMARK[
            (BENCHMARK["model_key"] == model_key) &
            (BENCHMARK["batch_size"] == batch_size) &
            (BENCHMARK["precision"] == "amp_fp16")
        ].iloc[0]

        amp_rows.append({
            "model_key": model_key,
            "model": MODEL_SHORT[model_key],
            "batch_size": batch_size,
            "fp32_latency_ms_per_image": f["latency_mean_ms_per_image"],
            "amp_latency_ms_per_image": a["latency_mean_ms_per_image"],
            "amp_speedup_latency": (
                f["latency_mean_ms_per_image"] /
                a["latency_mean_ms_per_image"]
            ),
            "fp32_throughput_images_per_s": f["throughput_images_per_s"],
            "amp_throughput_images_per_s": a["throughput_images_per_s"],
            "amp_speedup_throughput": (
                a["throughput_images_per_s"] /
                f["throughput_images_per_s"]
            ),
            "fp32_peak_vram_mb": f["vram_peak_total_mb"],
            "amp_peak_vram_mb": a["vram_peak_total_mb"],
        })

AMP_COMPARISON = pd.DataFrame(amp_rows)
AMP_COMPARISON.to_csv(RUN_ROOT / "11_amp_vs_fp32.csv", index=False)
display(AMP_COMPARISON.round(3))

# %% [markdown] Cell 22
# ## 10. Figuras

# %% Cell 23
# Parámetros
fig = plt.figure(figsize=(8,5))
plt.bar(SUMMARY["model"], SUMMARY["parameters_millions"])
plt.ylabel("Parameters (millions)")
plt.xlabel("Architecture")
plt.title("Model parameter count")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_ROOT / "11_parameters_millions.png", dpi=250, bbox_inches="tight")
plt.show()

# GFLOPs
fig = plt.figure(figsize=(8,5))
plt.bar(SUMMARY["model"], SUMMARY["gflops_fvcore"])
plt.ylabel("GFLOPs per image (fvcore)")
plt.xlabel("Architecture")
plt.title("Computational complexity at 224×224")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_ROOT / "11_gflops.png", dpi=250, bbox_inches="tight")
plt.show()

# Latencia
fig = plt.figure(figsize=(8,5))
plt.bar(SUMMARY["model"], SUMMARY["latency_amp_batch1_ms_per_image"])
plt.ylabel("Latency (ms/image)")
plt.xlabel("Architecture")
plt.title("Single-image inference latency — AMP/FP16")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_ROOT / "11_latency_amp_batch1.png", dpi=250, bbox_inches="tight")
plt.show()

# Throughput
fig = plt.figure(figsize=(8,5))
plt.bar(SUMMARY["model"], SUMMARY["throughput_amp_batch32_images_per_s"])
plt.ylabel("Images per second")
plt.xlabel("Architecture")
plt.title("Inference throughput — batch 32, AMP/FP16")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_ROOT / "11_throughput_amp_batch32.png", dpi=250, bbox_inches="tight")
plt.show()

# VRAM
fig = plt.figure(figsize=(8,5))
plt.bar(SUMMARY["model"], SUMMARY["vram_amp_batch32_peak_total_mb"])
plt.ylabel("Peak allocated VRAM (MB)")
plt.xlabel("Architecture")
plt.title("Inference VRAM — batch 32, AMP/FP16")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(FIGURE_ROOT / "11_vram_amp_batch32.png", dpi=250, bbox_inches="tight")
plt.show()

print("✅ Figuras:", FIGURE_ROOT)

# %% [markdown] Cell 24
# ## 11. Nota metodológica y entorno

# %% Cell 25
unsupported_any = COMPLEXITY["unsupported_ops_json"].astype(str).ne("{}").any()

NOTE = f'''
PADDY DOCTOR Q1 — NOTEBOOK 11
COMPUTATIONAL EFFICIENCY BENCHMARK

Hardware:
- GPU: {GPU_NAME}
- CUDA: {torch.version.cuda}
- PyTorch: {torch.__version__}
- timm: {timm.__version__}
- Python: {platform.python_version()}

Representative checkpoint:
- P1 seed {REPRESENTATIVE_SEED}.
- Efficiency is architectural; learned weights do not alter operation count.

Input:
- Synthetic tensor 3x{IMAGE_SIZE}x{IMAGE_SIZE}.
- I/O, JPEG decoding and DataLoader cost are excluded.

Timing:
- model.eval()
- torch.inference_mode()
- no torch.compile
- cuDNN benchmark enabled
- {WARMUP_ITERS} warm-up iterations
- {TIMED_ITERS} timed iterations
- CUDA synchronized per timing
- batch sizes {BATCH_SIZES}
- FP32 and AMP/FP16

Primary manuscript table:
- latency: AMP/FP16 batch 1
- throughput: AMP/FP16 batch 32
- VRAM: AMP/FP16 batch 32

Complexity:
- Parameters from PyTorch.
- FLOPs from fvcore FlopCountAnalysis at 224x224.
- FLOP estimates are analyzer/convention dependent.
- Unsupported fvcore operators present in at least one model: {unsupported_any}

Limitations:
- Hardware/software specific.
- Do not compare directly with another GPU, precision, batch size,
  input resolution, compilation strategy or timing protocol.
- Inference only; training time and energy consumption are not estimated.
'''

with open(RUN_ROOT / "11_efficiency_methodology_notes.txt", "w", encoding="utf-8") as f:
    f.write(NOTE)

ENV = {
    "gpu": GPU_NAME,
    "cuda": torch.version.cuda,
    "torch": torch.__version__,
    "timm": timm.__version__,
    "python": platform.python_version(),
    "warmup_iters": WARMUP_ITERS,
    "timed_iters": TIMED_ITERS,
    "batch_sizes": BATCH_SIZES,
    "precisions": PRECISIONS,
    "image_size": IMAGE_SIZE,
}

with open(RUN_ROOT / "11_environment.json", "w", encoding="utf-8") as f:
    json.dump(ENV, f, indent=2)

print(NOTE)

# %% [markdown] Cell 26
# ## 12. Verificación final

# %% Cell 27
required = [
    "11_benchmark_config.json",
    "11_model_complexity.csv",
    "11_latency_throughput_vram_all_settings.csv",
    "11_efficiency_primary_summary.csv",
    "11_amp_vs_fp32.csv",
    "11_efficiency_methodology_notes.txt",
    "11_environment.json",
]

missing = [name for name in required if not (RUN_ROOT / name).exists()]

if missing:
    raise RuntimeError(f"❌ Faltan outputs: {missing}")

if len(COMPLEXITY) != 4:
    raise RuntimeError("❌ COMPLEXITY debe tener 4 modelos.")

if len(BENCHMARK) != 24:
    raise RuntimeError("❌ BENCHMARK debe tener 24 configuraciones.")

if len(SUMMARY) != 4:
    raise RuntimeError("❌ SUMMARY debe tener 4 modelos.")

if len(AMP_COMPARISON) != 12:
    raise RuntimeError("❌ AMP_COMPARISON debe tener 12 filas.")

print("="*80)
print("✅ NOTEBOOK 11 COMPLETADO")
print("="*80)
print("GPU:", GPU_NAME)
print("Modelos:", len(COMPLEXITY))
print("Benchmark settings:", len(BENCHMARK))
print("Figuras:", len(list(FIGURE_ROOT.glob("*.png"))))

# %% [markdown] Cell 28
# ## 13. Exportar resultados a ZIP

# %% Cell 29
from google.colab import files

timestamp = time.strftime("%Y%m%d_%H%M%S")
package_name = f"11_PaddyDoctor_Computational_Efficiency_RESULTS_{timestamp}"
temp_package = Path("/content") / package_name

if temp_package.exists():
    shutil.rmtree(temp_package)

(temp_package / "results").mkdir(parents=True, exist_ok=True)
(temp_package / "figures").mkdir(parents=True, exist_ok=True)

for path in RUN_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(path, temp_package / "results" / path.name)

for path in FIGURE_ROOT.iterdir():
    if path.is_file():
        shutil.copy2(path, temp_package / "figures" / path.name)

with open(temp_package / "README_RESULTS.txt", "w", encoding="utf-8") as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 11\n"
        "Computational Efficiency Benchmark\n\n"
        f"GPU: {GPU_NAME}\n"
        f"Input: 3x{IMAGE_SIZE}x{IMAGE_SIZE}\n"
        f"Warm-up: {WARMUP_ITERS}\n"
        f"Timed iterations: {TIMED_ITERS}\n"
        f"Batch sizes: {BATCH_SIZES}\n"
        "Precision: FP32 and AMP/FP16\n"
        "Metrics: parameters, checkpoint size, GFLOPs, latency, throughput, VRAM.\n"
        "No model training was performed.\n"
    )

zip_path = shutil.make_archive(
    str(EXPORTS_DIR / package_name),
    "zip",
    root_dir=temp_package
)

print("✅ ZIP:", zip_path)
print(f"Tamaño: {Path(zip_path).stat().st_size/(1024**2):.2f} MB")
files.download(zip_path)

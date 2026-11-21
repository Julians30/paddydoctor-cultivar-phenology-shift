"""Auto-exported mirror of Notebook 13.
Canonical executable record: notebooks/13_PaddyDoctor_Advanced_Reliability_Equivalence_Selective_Conformal.ipynb
Runs in Google Colab and writes outputs to /content/drive/MyDrive/PADDY_DOCTOR_Q1.
"""

# # Paddy Doctor Q1 — Notebook 13
# ## Advanced Reliability, Practical Equivalence, Selective Prediction and Prediction-Set Diagnostics
# 
# ### Objetivo
# Añadir una capa estadística y de confiabilidad avanzada **sin reentrenar redes**. El notebook reutiliza los resultados ya producidos por P1, P4, P5, Notebook 08 y Notebook 09.
# 
# ### Bloques
# 1. **Practical equivalence** entre arquitecturas con margen preespecificado \(\delta=\pm 0.01\) en Macro-F1.
# 2. **Bootstrap pareado estratificado por clase** con 5,000 réplicas.
# 3. **Selective classification / risk–coverage** y AURC.
# 4. **Prediction-set diagnostics** inspirados en split conformal usando LAC (Least Ambiguous Classifier).
# 5. Figuras y tablas listas para revisión editorial.
# 6. Empaquetado automático en ZIP, guardado en Drive y descarga al equipo.
# 
# ### Regla anti-leakage
# No se ajusta ningún modelo con test. Todas las comparaciones de test usan predicciones previamente congeladas.
# 
# ### Advertencia metodológica sobre conformal
# La validación disponible fue usada previamente para selección de checkpoint y, en Notebook 09, para temperature scaling. Por tanto, los prediction sets de este notebook se interpretan **como diagnóstico empírico de cobertura y eficiencia bajo domain shift**, no como una garantía conformal finito-muestral estricta. Una afirmación formal requeriría un conjunto de calibración independiente no usado en selección del modelo.
# 
# > **GPU no requerida.** Ejecutar en Colab CPU.

# %% Cell 1
from google.colab import drive
from pathlib import Path
import os, json, math, shutil, time, hashlib, warnings

drive.mount("/content/drive", force_remount=False)

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"
NOTEBOOKS_DIR = PROJECT / "notebooks"
SCRIPTS_DIR = PROJECT / "scripts"

P1_ROOT = RESULTS_DIR / "05_P1_30ep_noES"
P4_ROOT = RESULTS_DIR / "06_P4_cross_cultivar_30ep_noES"
P5_ROOT = RESULTS_DIR / "07_P5_exact_age_matched_30ep_noES"
N08_ROOT = RESULTS_DIR / "08_joint_statistical_inference"
N09_ROOT = RESULTS_DIR / "09_calibration_reliability"
VAL_CACHE = N09_ROOT / "validation_cache"

RUN_ROOT = RESULTS_DIR / "13_advanced_reliability_equivalence"
FIGURE_ROOT = FIGURES_DIR / "13_advanced_reliability_equivalence"

for p in [PROJECT, RESULTS_DIR, FIGURES_DIR, EXPORTS_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR,
          RUN_ROOT, FIGURE_ROOT]:
    p.mkdir(parents=True, exist_ok=True)

print("✅ Proyecto:", PROJECT)
print("✅ Resultados N13:", RUN_ROOT)
print("✅ Figuras N13:", FIGURE_ROOT)
print("✅ GPU no requerida")

# %% Cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score
from scipy.special import softmax

N_BOOT = 5000
BOOT_SEED = 20260828
EQUIV_DELTA = 0.01
ALPHA = 0.05
SELECTIVE_COVERAGES = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
CURVE_COVERAGES = np.linspace(0.10, 1.00, 91)
CONFORMAL_ALPHAS = [0.10, 0.05]  # nominal coverage 90% and 95%
EPS = 1e-12

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

P4_PROTOCOL_MAP = {
    ("ADT45", "OTHER"): "P4A_ADT45_to_OTHER",
    ("ADT45", "ADT45"): "P4B_ADT45_to_ADT45",
    ("OTHER", "ADT45"): "P4C_OTHER_to_ADT45",
    ("OTHER", "OTHER"): "P4D_OTHER_to_OTHER",
}

P5_PROTOCOL_MAP = {
    ("ADT45", "OTHER"): "P5A_ADT45_to_OTHER_age_matched",
    ("ADT45", "ADT45"): "P5B_ADT45_to_ADT45_age_control",
    ("OTHER", "ADT45"): "P5C_OTHER_to_ADT45_age_matched",
    ("OTHER", "OTHER"): "P5D_OTHER_to_OTHER_age_control",
}

config = {
    "notebook": "13_PaddyDoctor_Advanced_Reliability_Equivalence_Selective_Conformal",
    "n_bootstrap": N_BOOT,
    "bootstrap_seed": BOOT_SEED,
    "primary_metric": "macro_f1",
    "equivalence_margin_absolute_macro_f1": EQUIV_DELTA,
    "equivalence_ci": "90% percentile bootstrap CI (TOST-compatible CI rule)",
    "estimation_ci": "95% percentile bootstrap CI",
    "selective_coverages": SELECTIVE_COVERAGES,
    "conformal_alphas": CONFORMAL_ALPHAS,
    "conformal_method": "LAC nonconformity score = 1 - p_true",
    "conformal_interpretation": (
        "Empirical prediction-set diagnostic only; validation was previously used "
        "for checkpoint selection and temperature scaling, so strict split-conformal "
        "finite-sample validity is not claimed."
    ),
    "models": MODELS,
    "seeds": SEEDS,
}

with open(RUN_ROOT / "13_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))

# ## 1. Auditoría de insumos
# 
# El notebook exige que los resultados previos estén completos antes de ejecutar cualquier inferencia estadística nueva.

# %% Cell 3
MASTER_PRED = N08_ROOT / "08_all_predictions_index.csv"
TEMP_PARAMS = N09_ROOT / "09_temperature_parameters_validation.csv"

required_files = [
    MASTER_PRED,
    TEMP_PARAMS,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"❌ Falta insumo obligatorio: {p}")

expected_complete = {
    "P1": (P1_ROOT, 12),
    "P4": (P4_ROOT, 24),
    "P5": (P5_ROOT, 24),
}

audit_rows = []

for exp, (root, expected_n) in expected_complete.items():
    n_complete = len(list(root.rglob("COMPLETE.json")))
    audit_rows.append({
        "component": exp,
        "path": str(root),
        "expected_complete": expected_n,
        "observed_complete": n_complete,
        "status": "OK" if n_complete == expected_n else "FAIL",
    })
    if n_complete != expected_n:
        raise RuntimeError(f"❌ {exp} incompleto: {n_complete}/{expected_n}")

n_cache_npz = len(list(VAL_CACHE.glob("*.npz"))) if VAL_CACHE.exists() else 0
n_cache_json = len(list(VAL_CACHE.glob("*.json"))) if VAL_CACHE.exists() else 0

audit_rows.append({
    "component": "N09_validation_cache_npz",
    "path": str(VAL_CACHE),
    "expected_complete": 60,
    "observed_complete": n_cache_npz,
    "status": "OK" if n_cache_npz == 60 else "WARN",
})
audit_rows.append({
    "component": "N09_validation_cache_json",
    "path": str(VAL_CACHE),
    "expected_complete": 60,
    "observed_complete": n_cache_json,
    "status": "OK" if n_cache_json == 60 else "WARN",
})

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(RUN_ROOT / "13_input_audit.csv", index=False)
display(audit_df)

if n_cache_npz != 60:
    print("⚠️ Prediction-set diagnostics requerirán los 60 .npz de validation_cache.")

# %% Cell 4
all_pred = pd.read_csv(MASTER_PRED)

required_cols = {
    "image_id", "y_true", "y_pred", "confidence",
    "experiment", "protocol", "model_key", "seed",
    "train_domain", "eval_domain", "correct"
}
missing = required_cols - set(all_pred.columns)
if missing:
    raise RuntimeError(f"❌ Faltan columnas en 08_all_predictions_index.csv: {missing}")

all_pred["seed"] = all_pred["seed"].astype(int)
all_pred["image_id"] = all_pred["image_id"].astype(str)

protocol_audit = (
    all_pred.groupby(["experiment", "protocol", "model_key", "seed"])
    .agg(
        n=("image_id", "size"),
        n_images=("image_id", "nunique"),
        accuracy=("correct", "mean"),
    )
    .reset_index()
)

protocol_audit.to_csv(RUN_ROOT / "13_prediction_index_audit.csv", index=False)

print("Filas totales:", len(all_pred))
print("Protocolos:", all_pred["protocol"].nunique())
print("Modelos:", all_pred["model_key"].nunique())
print("Seeds:", sorted(all_pred["seed"].unique().tolist()))
display(protocol_audit.head(12))

# ## 2. Practical equivalence con bootstrap pareado de 5,000 réplicas
# 
# Se utiliza un margen preespecificado:
# 
# \[
# \delta = \pm 0.01
# \]
# 
# en la escala absoluta de Macro-F1.
# 
# Para cada protocolo:
# - se mantiene exactamente el mismo remuestreo de imágenes para todas las arquitecturas;
# - el bootstrap es estratificado por clase;
# - Macro-F1 se calcula por seed y luego se promedian los tres seeds dentro de cada réplica;
# - se reporta IC 95% para estimación;
# - se reporta IC 90% para la regla de equivalencia práctica;
# - equivalencia se declara solo si el IC 90% completo queda dentro de \([-0.01, +0.01]\).

# %% Cell 5
def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(np.mean(f1))

def fast_macro_f1(y_true_code, y_pred_code, n_classes, idx=None):
    if idx is None:
        yt = y_true_code
        yp = y_pred_code
    else:
        yt = y_true_code[idx]
        yp = y_pred_code[idx]
    cm = np.bincount(
        yt * n_classes + yp,
        minlength=n_classes * n_classes
    ).reshape(n_classes, n_classes)
    return macro_f1_from_confusion(cm)

def prepare_protocol_arrays(df, experiment, protocol):
    sub = df[(df["experiment"] == experiment) & (df["protocol"] == protocol)].copy()
    if sub.empty:
        raise RuntimeError(f"❌ Sin datos: {experiment}/{protocol}")

    ref = (
        sub[(sub["model_key"] == MODELS[0]) & (sub["seed"] == SEEDS[0])]
        [["image_id", "y_true"]]
        .sort_values("image_id")
        .reset_index(drop=True)
    )
    if ref.empty:
        raise RuntimeError(f"❌ Referencia vacía: {experiment}/{protocol}")

    class_names = sorted(ref["y_true"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true_code = ref["y_true"].astype(str).map(class_to_idx).to_numpy(dtype=np.int16)
    image_ids = ref["image_id"].astype(str).to_numpy()

    pred_map = {}
    observed_seed_f1 = {}

    for model in MODELS:
        observed_seed_f1[model] = []
        for seed in SEEDS:
            m = (
                sub[(sub["model_key"] == model) & (sub["seed"] == seed)]
                [["image_id", "y_true", "y_pred"]]
                .sort_values("image_id")
                .reset_index(drop=True)
            )
            if len(m) != len(ref):
                raise RuntimeError(
                    f"❌ N diferente en {experiment}/{protocol}/{model}/{seed}: "
                    f"{len(m)} vs {len(ref)}"
                )
            if not np.array_equal(m["image_id"].astype(str).to_numpy(), image_ids):
                raise RuntimeError(f"❌ image_id no apareado: {experiment}/{protocol}/{model}/{seed}")
            if not np.array_equal(
                m["y_true"].astype(str).to_numpy(),
                ref["y_true"].astype(str).to_numpy()
            ):
                raise RuntimeError(f"❌ y_true no apareado: {experiment}/{protocol}/{model}/{seed}")

            yp = m["y_pred"].astype(str).map(class_to_idx)
            if yp.isna().any():
                bad = m.loc[yp.isna(), "y_pred"].unique().tolist()
                raise RuntimeError(f"❌ Predicciones fuera del mapa de clases: {bad}")

            yp_code = yp.to_numpy(dtype=np.int16)
            pred_map[(model, seed)] = yp_code
            observed_seed_f1[model].append(
                fast_macro_f1(y_true_code, yp_code, len(class_names))
            )

    return {
        "image_ids": image_ids,
        "y_true_code": y_true_code,
        "class_names": class_names,
        "pred_map": pred_map,
        "observed_seed_f1": observed_seed_f1,
    }

def stratified_indices(y_true_code, rng):
    pieces = []
    for c in np.unique(y_true_code):
        idx = np.flatnonzero(y_true_code == c)
        pieces.append(rng.choice(idx, size=len(idx), replace=True))
    return np.concatenate(pieces)

def bootstrap_architectures(prepared, n_boot=N_BOOT, seed=BOOT_SEED):
    y = prepared["y_true_code"]
    pred_map = prepared["pred_map"]
    n_classes = len(prepared["class_names"])
    rng = np.random.default_rng(seed)

    boot = np.empty((n_boot, len(MODELS)), dtype=np.float32)

    for b in range(n_boot):
        idx = stratified_indices(y, rng)
        for mi, model in enumerate(MODELS):
            vals = [
                fast_macro_f1(y, pred_map[(model, s)], n_classes, idx)
                for s in SEEDS
            ]
            boot[b, mi] = np.mean(vals)

    return boot

protocols = (
    all_pred[["experiment", "protocol"]]
    .drop_duplicates()
    .sort_values(["experiment", "protocol"])
    .itertuples(index=False, name=None)
)

equiv_rows = []
bootstrap_protocol_summaries = []

start = time.time()

for p_idx, (experiment, protocol) in enumerate(protocols, start=1):
    print(f"[{p_idx}] {experiment} / {protocol}")
    prepared = prepare_protocol_arrays(all_pred, experiment, protocol)
    boot = bootstrap_architectures(
        prepared,
        n_boot=N_BOOT,
        seed=BOOT_SEED + p_idx
    )

    observed_means = {
        model: float(np.mean(prepared["observed_seed_f1"][model]))
        for model in MODELS
    }
    observed_sds = {
        model: float(np.std(prepared["observed_seed_f1"][model], ddof=1))
        for model in MODELS
    }

    for model in MODELS:
        bootstrap_protocol_summaries.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_key": model,
            "model": MODEL_SHORT[model],
            "macro_f1_mean_3seeds": observed_means[model],
            "macro_f1_sd_3seeds": observed_sds[model],
            "bootstrap_ci95_low": float(np.quantile(boot[:, MODELS.index(model)], 0.025)),
            "bootstrap_ci95_high": float(np.quantile(boot[:, MODELS.index(model)], 0.975)),
        })

    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            d = boot[:, i] - boot[:, j]
            observed_delta = observed_means[a] - observed_means[b]

            ci95_lo, ci95_hi = np.quantile(d, [0.025, 0.975])
            ci90_lo, ci90_hi = np.quantile(d, [0.05, 0.95])

            equivalent = bool(
                (ci90_lo > -EQUIV_DELTA) and
                (ci90_hi < EQUIV_DELTA)
            )
            statistically_directional = (
                "A>B" if ci95_lo > 0
                else "A<B" if ci95_hi < 0
                else "CI95_includes_0"
            )

            if equivalent and statistically_directional == "CI95_includes_0":
                interpretation = "practically_equivalent"
            elif equivalent:
                interpretation = "statistically_different_but_practically_equivalent"
            elif statistically_directional == "A>B":
                interpretation = "A_superior"
            elif statistically_directional == "A<B":
                interpretation = "B_superior"
            else:
                interpretation = "inconclusive"

            equiv_rows.append({
                "experiment": experiment,
                "protocol": protocol,
                "model_A_key": a,
                "model_A": MODEL_SHORT[a],
                "model_B_key": b,
                "model_B": MODEL_SHORT[b],
                "delta_macro_f1_A_minus_B": observed_delta,
                "ci95_low": float(ci95_lo),
                "ci95_high": float(ci95_hi),
                "ci90_low": float(ci90_lo),
                "ci90_high": float(ci90_hi),
                "equivalence_margin": EQUIV_DELTA,
                "equivalent_90ci_within_margin": equivalent,
                "statistical_direction_95ci": statistically_directional,
                "bootstrap_prob_A_gt_B": float(np.mean(d > 0)),
                "bootstrap_prob_abs_delta_lt_margin": float(np.mean(np.abs(d) < EQUIV_DELTA)),
                "interpretation": interpretation,
                "n_boot": N_BOOT,
            })

equivalence_df = pd.DataFrame(equiv_rows)
protocol_summary_df = pd.DataFrame(bootstrap_protocol_summaries)

equivalence_df.to_csv(RUN_ROOT / "13_equivalence_pairwise_5000boot.csv", index=False)
protocol_summary_df.to_csv(RUN_ROOT / "13_macro_f1_bootstrap_summary_5000boot.csv", index=False)

print(f"✅ Equivalencia completada en {(time.time()-start)/60:.2f} min")
display(equivalence_df.head(12))

# ## 3. Selective classification y risk–coverage
# 
# Se ordenan las imágenes por confianza descendente. Para cada nivel de cobertura se calcula:
# - accuracy;
# - selective risk \(=1-\mathrm{accuracy}\);
# - Macro-F1;
# - número de imágenes retenidas.
# 
# También se calcula **AURC** (Area Under the Risk–Coverage Curve). Menor AURC implica mejor capacidad de abstención.

# %% Cell 6
def selective_metrics_group(g):
    g = g.copy()
    g["confidence"] = pd.to_numeric(g["confidence"], errors="raise")
    g["correct"] = pd.to_numeric(g["correct"], errors="raise").astype(int)
    g = g.sort_values("confidence", ascending=False).reset_index(drop=True)

    n = len(g)
    rows = []

    all_labels = sorted(g["y_true"].astype(str).unique().tolist())

    for cov in SELECTIVE_COVERAGES:
        k = max(1, min(n, int(math.ceil(cov * n))))
        kept = g.iloc[:k]
        acc = float((kept["y_true"].astype(str) == kept["y_pred"].astype(str)).mean())
        mf1 = float(f1_score(
            kept["y_true"].astype(str),
            kept["y_pred"].astype(str),
            labels=all_labels,
            average="macro",
            zero_division=0,
        ))
        rows.append({
            "coverage": k / n,
            "target_coverage": cov,
            "n_total": n,
            "n_retained": k,
            "accuracy": acc,
            "risk": 1.0 - acc,
            "macro_f1": mf1,
            "mean_confidence_retained": float(kept["confidence"].mean()),
        })

    curve_rows = []
    cum_errors = np.cumsum(1 - g["correct"].to_numpy())
    for cov in CURVE_COVERAGES:
        k = max(1, min(n, int(math.ceil(cov * n))))
        risk = float(cum_errors[k - 1] / k)
        curve_rows.append({
            "coverage": k / n,
            "target_coverage": float(cov),
            "risk": risk,
        })

    full_risk = cum_errors / np.arange(1, n + 1)
    aurc = float(np.mean(full_risk))

    return pd.DataFrame(rows), pd.DataFrame(curve_rows), aurc

selective_rows = []
curve_rows_all = []
aurc_rows = []

group_cols = ["experiment", "protocol", "model_key", "seed"]

for keys, g in all_pred.groupby(group_cols, sort=False):
    experiment, protocol, model, seed = keys
    sel, curve, aurc = selective_metrics_group(g)

    for _, r in sel.iterrows():
        selective_rows.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_key": model,
            "model": MODEL_SHORT.get(model, model),
            "seed": int(seed),
            **r.to_dict(),
        })

    for _, r in curve.iterrows():
        curve_rows_all.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_key": model,
            "model": MODEL_SHORT.get(model, model),
            "seed": int(seed),
            **r.to_dict(),
        })

    aurc_rows.append({
        "experiment": experiment,
        "protocol": protocol,
        "model_key": model,
        "model": MODEL_SHORT.get(model, model),
        "seed": int(seed),
        "aurc": aurc,
    })

selective_df = pd.DataFrame(selective_rows)
curve_df = pd.DataFrame(curve_rows_all)
aurc_df = pd.DataFrame(aurc_rows)

selective_aggregate = (
    selective_df
    .groupby(["experiment", "protocol", "model_key", "model", "target_coverage"], as_index=False)
    .agg(
        coverage_mean=("coverage", "mean"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        risk_mean=("risk", "mean"),
        risk_sd=("risk", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        n_retained_mean=("n_retained", "mean"),
    )
)

aurc_aggregate = (
    aurc_df
    .groupby(["experiment", "protocol", "model_key", "model"], as_index=False)
    .agg(
        aurc_mean=("aurc", "mean"),
        aurc_sd=("aurc", "std"),
    )
)

selective_df.to_csv(RUN_ROOT / "13_selective_by_run.csv", index=False)
selective_aggregate.to_csv(RUN_ROOT / "13_selective_aggregate.csv", index=False)
curve_df.to_csv(RUN_ROOT / "13_risk_coverage_points_by_run.csv", index=False)
aurc_df.to_csv(RUN_ROOT / "13_aurc_by_run.csv", index=False)
aurc_aggregate.to_csv(RUN_ROOT / "13_aurc_aggregate.csv", index=False)

print("✅ Selective classification completada")
display(aurc_aggregate.head(12))

# ## 4. Prediction-set diagnostics (LAC)
# 
# Se usa el score:
# 
# \[
# s(x,y)=1-p_y(x)
# \]
# 
# y el cuantil finito-muestral:
# 
# \[
# q=\frac{\lceil(n+1)(1-\alpha)\rceil}{n}
# \]
# 
# con método `higher`.
# 
# Se evalúan probabilidades:
# - **raw**;
# - **temperature-scaled**, reutilizando exclusivamente la temperatura ajustada en validation por Notebook 09.
# 
# ### Importante
# Estos resultados **no deben presentarse como cobertura conformal garantizada**, porque el mismo validation intervino antes en selección de checkpoint. Se reportan como diagnóstico de:
# - cobertura empírica;
# - desviación respecto a cobertura nominal;
# - tamaño del prediction set;
# - tasa de singleton;
# - tasa de set vacío;
# - degradación bajo domain shift.

# %% Cell 7
temp_params = pd.read_csv(TEMP_PARAMS)
required_temp_cols = {
    "experiment", "model_key", "seed", "train_domain",
    "run_id", "class_names_json", "temperature"
}
missing = required_temp_cols - set(temp_params.columns)
if missing:
    raise RuntimeError(f"❌ Faltan columnas en 09_temperature_parameters_validation.csv: {missing}")

temp_params["seed"] = temp_params["seed"].astype(int)

def temperature_probabilities_from_probs(probs, temperature):
    probs = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    pseudo_logits = np.log(probs)
    return softmax(pseudo_logits / float(temperature), axis=1)

def prediction_path(experiment, model, seed, train_domain, eval_domain):
    if experiment == "P1":
        return P1_ROOT / f"{model}__seed_{seed}" / "test_predictions.csv"
    if experiment == "P4":
        return (
            P4_ROOT /
            f"train_{train_domain}__{model}__seed_{seed}" /
            f"eval__{eval_domain}" /
            "test_predictions.csv"
        )
    if experiment == "P5":
        return (
            P5_ROOT /
            f"train_{train_domain}__{model}__seed_{seed}" /
            f"eval__{eval_domain}" /
            "test_predictions.csv"
        )
    raise ValueError(experiment)

def protocol_name(experiment, train_domain, eval_domain):
    if experiment == "P1":
        return "P1_IID"
    if experiment == "P4":
        return P4_PROTOCOL_MAP[(train_domain, eval_domain)]
    if experiment == "P5":
        return P5_PROTOCOL_MAP[(train_domain, eval_domain)]
    raise ValueError(experiment)

def finite_sample_qhat(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    qlevel = math.ceil((n + 1) * (1 - alpha)) / n
    qlevel = min(1.0, qlevel)
    try:
        qhat = np.quantile(scores, qlevel, method="higher")
    except TypeError:
        qhat = np.quantile(scores, qlevel, interpolation="higher")
    return float(qhat), float(qlevel)

def evaluate_lac_sets(probs_cal, y_cal, probs_test, y_test, alpha):
    scores = 1.0 - probs_cal[np.arange(len(y_cal)), y_cal]
    qhat, qlevel = finite_sample_qhat(scores, alpha)
    threshold = 1.0 - qhat

    sets = probs_test >= threshold
    sizes = sets.sum(axis=1)
    included = sets[np.arange(len(y_test)), y_test]

    return {
        "qhat": qhat,
        "qlevel": qlevel,
        "probability_threshold": threshold,
        "empirical_coverage": float(np.mean(included)),
        "mean_set_size": float(np.mean(sizes)),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "empty_set_rate": float(np.mean(sizes == 0)),
        "max_set_size": int(np.max(sizes)),
        "_sets": sets,
        "_sizes": sizes,
        "_included": included,
    }

prediction_set_rows = []
prediction_set_class_rows = []

for _, tp in temp_params.iterrows():
    experiment = str(tp["experiment"])
    model = str(tp["model_key"])
    seed = int(tp["seed"])
    train_domain = str(tp["train_domain"])
    run_id = str(tp["run_id"])
    temperature = float(tp["temperature"])
    class_names = json.loads(tp["class_names_json"])
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    cache_npz = VAL_CACHE / f"{run_id}.npz"
    if not cache_npz.exists():
        raise FileNotFoundError(
            f"❌ Falta validation cache para prediction-set diagnostics: {cache_npz}"
        )

    arrays = np.load(cache_npz, allow_pickle=False)
    y_val = arrays["y_true"].astype(int)
    logits_val = arrays["logits"].astype(float)

    val_raw = softmax(logits_val, axis=1)
    val_ts = softmax(logits_val / temperature, axis=1)

    eval_domains = ["MIXED"] if experiment == "P1" else ["ADT45", "OTHER"]

    for eval_domain in eval_domains:
        path = prediction_path(
            experiment, model, seed,
            train_domain if experiment != "P1" else "MIXED",
            eval_domain
        )
        if not path.exists():
            raise FileNotFoundError(path)

        df_test = pd.read_csv(path)
        prob_cols = [f"prob__{c}" for c in class_names]
        missing_prob = [c for c in prob_cols if c not in df_test.columns]
        if missing_prob:
            raise RuntimeError(
                f"❌ Faltan columnas de probabilidad en {path.name}: {missing_prob[:5]}"
            )

        test_raw = df_test[prob_cols].to_numpy(dtype=float)
        test_ts = temperature_probabilities_from_probs(test_raw, temperature)

        y_test = df_test["y_true"].astype(str).map(class_to_idx)
        if y_test.isna().any():
            raise RuntimeError(f"❌ y_true fuera del mapa de clases en {path}")
        y_test = y_test.to_numpy(dtype=int)

        protocol = protocol_name(
            experiment,
            "MIXED" if experiment == "P1" else train_domain,
            eval_domain
        )

        for method, pcal, ptest in [
            ("raw", val_raw, test_raw),
            ("temperature_scaled", val_ts, test_ts),
        ]:
            for alpha in CONFORMAL_ALPHAS:
                res = evaluate_lac_sets(pcal, y_val, ptest, y_test, alpha)
                nominal = 1.0 - alpha

                prediction_set_rows.append({
                    "experiment": experiment,
                    "protocol": protocol,
                    "model_key": model,
                    "model": MODEL_SHORT.get(model, model),
                    "seed": seed,
                    "train_domain": train_domain,
                    "eval_domain": eval_domain,
                    "method": method,
                    "alpha": alpha,
                    "nominal_coverage": nominal,
                    "n_calibration": len(y_val),
                    "n_test": len(y_test),
                    "temperature": temperature,
                    "qhat": res["qhat"],
                    "qlevel": res["qlevel"],
                    "probability_threshold": res["probability_threshold"],
                    "empirical_coverage": res["empirical_coverage"],
                    "coverage_minus_nominal": res["empirical_coverage"] - nominal,
                    "mean_set_size": res["mean_set_size"],
                    "median_set_size": res["median_set_size"],
                    "singleton_rate": res["singleton_rate"],
                    "empty_set_rate": res["empty_set_rate"],
                    "max_set_size": res["max_set_size"],
                    "formal_coverage_guarantee_claimed": False,
                })

                sets = res["_sets"]
                sizes = res["_sizes"]
                included = res["_included"]

                for class_idx, class_name in enumerate(class_names):
                    mask = y_test == class_idx
                    if mask.sum() == 0:
                        continue
                    prediction_set_class_rows.append({
                        "experiment": experiment,
                        "protocol": protocol,
                        "model_key": model,
                        "model": MODEL_SHORT.get(model, model),
                        "seed": seed,
                        "train_domain": train_domain,
                        "eval_domain": eval_domain,
                        "method": method,
                        "alpha": alpha,
                        "class_name": class_name,
                        "n_class": int(mask.sum()),
                        "empirical_coverage_class": float(np.mean(included[mask])),
                        "mean_set_size_class": float(np.mean(sizes[mask])),
                    })

prediction_sets_df = pd.DataFrame(prediction_set_rows)
prediction_sets_class_df = pd.DataFrame(prediction_set_class_rows)

prediction_sets_aggregate = (
    prediction_sets_df
    .groupby([
        "experiment", "protocol", "model_key", "model",
        "method", "alpha", "nominal_coverage"
    ], as_index=False)
    .agg(
        empirical_coverage_mean=("empirical_coverage", "mean"),
        empirical_coverage_sd=("empirical_coverage", "std"),
        coverage_minus_nominal_mean=("coverage_minus_nominal", "mean"),
        mean_set_size_mean=("mean_set_size", "mean"),
        mean_set_size_sd=("mean_set_size", "std"),
        singleton_rate_mean=("singleton_rate", "mean"),
        empty_set_rate_mean=("empty_set_rate", "mean"),
    )
)

prediction_sets_df.to_csv(
    RUN_ROOT / "13_prediction_set_diagnostics_by_run.csv", index=False
)
prediction_sets_aggregate.to_csv(
    RUN_ROOT / "13_prediction_set_diagnostics_aggregate.csv", index=False
)
prediction_sets_class_df.to_csv(
    RUN_ROOT / "13_prediction_set_diagnostics_classwise.csv", index=False
)

print("✅ Prediction-set diagnostics completados")
display(prediction_sets_aggregate.head(12))

# ## 5. Figuras de síntesis
# 
# **Plotting note.** Figure 13C uses the full empirical-coverage axis (0–1.02). This prevents cross-domain coverage failures from being visually truncated and preserves faithful interpretation of the results.

# %% Cell 8
# Figure 13A — Equivalence forest plot for P1
p1eq = equivalence_df[equivalence_df["protocol"] == "P1_IID"].copy()
if not p1eq.empty:
    p1eq["comparison"] = p1eq["model_A"] + " − " + p1eq["model_B"]
    p1eq = p1eq.sort_values("delta_macro_f1_A_minus_B").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(p1eq))
    x = p1eq["delta_macro_f1_A_minus_B"].to_numpy()
    lo = p1eq["ci90_low"].to_numpy()
    hi = p1eq["ci90_high"].to_numpy()

    ax.errorbar(
        x, y,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="o",
        capsize=4
    )
    ax.axvline(0, linewidth=1)
    ax.axvspan(-EQUIV_DELTA, EQUIV_DELTA, alpha=0.12)
    ax.set_yticks(y)
    ax.set_yticklabels(p1eq["comparison"])
    ax.set_xlabel("Δ Macro-F1 (A − B), 90% bootstrap CI")
    ax.set_title("P1 practical equivalence (margin ±0.01)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "Figure_13A_P1_Practical_Equivalence.png", dpi=300, bbox_inches="tight")
    plt.show()

# **Plotting note for Figure 13B.** Risk–coverage is displayed as three separate figures using identical axes (coverage 0.10–1.00; selective risk 0–0.65): P1 IID, P4 bidirectional cross-cultivar evaluation, and P5 bidirectional exact-age-matched cross-cultivar evaluation. P4 and P5 curves average the two cross-domain directions and the three seeds at each coverage. This is a descriptive visualization; inferential claims remain based on the paired analyses reported elsewhere.

# %% Cell 9
# Figure 13B series — full-scale selective risk–coverage
#
# Three separate figures are generated with the SAME axis limits.
# This avoids mixing P1, P4 and P5 in a single crowded plot and makes the
# magnitude of cross-domain degradation directly comparable.
#
# P4 visualization:
#   averages P4A (ADT45→OTHER) and P4C (OTHER→ADT45), across the three seeds.
# P5 visualization:
#   averages P5A (ADT45→OTHER, exact-age matched) and
#   P5C (OTHER→ADT45, exact-age matched), across the three seeds.
#
# These are descriptive risk–coverage summaries; inferential conclusions remain
# based on the paired analyses defined elsewhere in the notebook.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Standalone regeneration from saved Notebook 13 outputs.
if "PROJECT" not in globals():
    PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

if "RUN_ROOT" not in globals():
    RUN_ROOT = PROJECT / "results" / "13_advanced_reliability_equivalence"

if "FIGURE_ROOT" not in globals():
    FIGURE_ROOT = PROJECT / "figures" / "13_advanced_reliability_equivalence"

FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

curve_csv = RUN_ROOT / "13_risk_coverage_points_by_run.csv"

if "curve_df" not in globals():
    if not curve_csv.exists():
        raise FileNotFoundError(
            "Missing 13_risk_coverage_points_by_run.csv. "
            "Run the selective-prediction block once before regenerating Figure 13B."
        )
    curve_df = pd.read_csv(curve_csv)

COMMON_XLIM = (0.10, 1.00)
COMMON_YLIM = (0.00, 0.65)

def aggregate_risk_curve(data):
    return (
        data
        .groupby(["model_key", "model", "target_coverage"], as_index=False)
        .agg(
            risk_mean=("risk", "mean"),
            risk_sd=("risk", "std"),
        )
    )

def save_risk_coverage_figure(data, title, png_name, pdf_name):
    agg = aggregate_risk_curve(data)

    fig, ax = plt.subplots(figsize=(8.8, 6.0))

    for model in MODELS:
        s = agg[agg["model_key"] == model].sort_values("target_coverage")
        if s.empty:
            continue
        ax.plot(
            s["target_coverage"],
            s["risk_mean"],
            label=MODEL_SHORT[model],
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 − accuracy)")
    ax.set_title(title)
    ax.set_xlim(*COMMON_XLIM)
    ax.set_ylim(*COMMON_YLIM)
    ax.grid(alpha=0.22)
    ax.legend()

    fig.tight_layout()

    png_path = FIGURE_ROOT / png_name
    pdf_path = FIGURE_ROOT / pdf_name

    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.show()

    return png_path, pdf_path

# -------------------------------------------------------------------------
# Figure 13B1 — P1 IID
# -------------------------------------------------------------------------
p1_curve = curve_df[curve_df["protocol"] == "P1_IID"].copy()

if p1_curve.empty:
    raise RuntimeError("No P1_IID risk–coverage rows found.")

p1_png, p1_pdf = save_risk_coverage_figure(
    p1_curve,
    "Selective risk–coverage under IID evaluation (P1)",
    "Figure_13B1_P1_IID_Risk_Coverage.png",
    "Figure_13B1_P1_IID_Risk_Coverage.pdf",
)

# -------------------------------------------------------------------------
# Figure 13B2 — P4 bidirectional cross-cultivar shift
# -------------------------------------------------------------------------
p4_cross_protocols = [
    "P4A_ADT45_to_OTHER",
    "P4C_OTHER_to_ADT45",
]

p4_curve = curve_df[
    (curve_df["experiment"] == "P4")
    & (curve_df["protocol"].isin(p4_cross_protocols))
].copy()

if p4_curve.empty:
    raise RuntimeError("No P4 bidirectional cross-domain risk–coverage rows found.")

p4_png, p4_pdf = save_risk_coverage_figure(
    p4_curve,
    "Selective risk–coverage under bidirectional cultivar shift (P4)",
    "Figure_13B2_P4_CrossCultivar_Risk_Coverage.png",
    "Figure_13B2_P4_CrossCultivar_Risk_Coverage.pdf",
)

# -------------------------------------------------------------------------
# Figure 13B3 — P5 bidirectional exact-age-matched cultivar shift
# -------------------------------------------------------------------------
p5_cross_protocols = [
    "P5A_ADT45_to_OTHER_age_matched",
    "P5C_OTHER_to_ADT45_age_matched",
]

p5_curve = curve_df[
    (curve_df["experiment"] == "P5")
    & (curve_df["protocol"].isin(p5_cross_protocols))
].copy()

if p5_curve.empty:
    raise RuntimeError("No P5 bidirectional exact-age-matched risk–coverage rows found.")

p5_png, p5_pdf = save_risk_coverage_figure(
    p5_curve,
    "Selective risk–coverage after exact-age matching (P5)",
    "Figure_13B3_P5_ExactAgeMatched_Risk_Coverage.png",
    "Figure_13B3_P5_ExactAgeMatched_Risk_Coverage.pdf",
)

print("✅ Figure 13B series generated with common axes")
print("P1:", p1_png)
print("P4:", p4_png)
print("P5:", p5_png)

# **Plotting note for Figure 13C.** The empirical-coverage axis is fixed to 0–1.02 so that severe cross-domain coverage failures are never visually truncated. IID, in-domain control, and cross-domain conditions are displayed separately, with raw and temperature-scaled probabilities distinguished by marker.

# %% Cell 10
# Standalone regeneration support:
# This cell can regenerate Figure 13C from the saved aggregate CSV
# without rerunning bootstrap, selective prediction, or prediction-set calculations.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "PROJECT" not in globals():
    PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

if "RUN_ROOT" not in globals():
    RUN_ROOT = PROJECT / "results" / "13_advanced_reliability_equivalence"

if "FIGURE_ROOT" not in globals():
    FIGURE_ROOT = PROJECT / "figures" / "13_advanced_reliability_equivalence"

FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

aggregate_csv = RUN_ROOT / "13_prediction_set_diagnostics_aggregate.csv"

if "prediction_sets_aggregate" not in globals():
    if not aggregate_csv.exists():
        raise FileNotFoundError(
            "Missing saved aggregate CSV. Run Notebook 13 through the "
            "prediction-set diagnostics block once before regenerating Figure 13C."
        )
    prediction_sets_aggregate = pd.read_csv(aggregate_csv)

# Figure 13C — Prediction-set coverage under cultivar/domain shift
# Full-scale version: full empirical-coverage scale (0–1) so cross-domain
# failures are never visually truncated.

if not prediction_sets_aggregate.empty:
    plot_df = prediction_sets_aggregate.copy()

    def classify_condition(protocol):
        p = str(protocol)
        if p == "P1_IID":
            return "IID"
        if (
            "ADT45_to_ADT45" in p
            or "OTHER_to_OTHER" in p
            or "control" in p.lower()
        ):
            return "In-domain control"
        return "Cross-domain"

    plot_df["condition"] = plot_df["protocol"].map(classify_condition)

    marker_map = {
        ("IID", "raw"): "o",
        ("IID", "temperature_scaled"): "s",
        ("In-domain control", "raw"): "^",
        ("In-domain control", "temperature_scaled"): "v",
        ("Cross-domain", "raw"): "x",
        ("Cross-domain", "temperature_scaled"): "D",
    }

    label_map = {
        ("IID", "raw"): "IID — raw",
        ("IID", "temperature_scaled"): "IID — temperature scaled",
        ("In-domain control", "raw"): "Control — raw",
        ("In-domain control", "temperature_scaled"): "Control — temperature scaled",
        ("Cross-domain", "raw"): "Cross-domain — raw",
        ("Cross-domain", "temperature_scaled"): "Cross-domain — temperature scaled",
    }

    fig, ax = plt.subplots(figsize=(9.5, 7.0))

    for key, marker in marker_map.items():
        condition, method = key
        s = plot_df[
            (plot_df["condition"] == condition)
            & (plot_df["method"] == method)
        ]

        if s.empty:
            continue

        ax.scatter(
            s["nominal_coverage"],
            s["empirical_coverage_mean"],
            marker=marker,
            s=62,
            alpha=0.65,
            label=label_map[key],
        )

    # Perfect nominal-to-empirical coverage agreement.
    x_reference = np.array([0.875, 0.975])
    ax.plot(
        x_reference,
        x_reference,
        linestyle="--",
        linewidth=1.5,
        label="Nominal = empirical",
    )

    # Plotting rule: empirical coverage must not be truncated.
    ax.set_xlim(0.875, 0.975)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0.90, 0.95])
    ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Prediction-set coverage under cultivar/domain shift")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower left", fontsize=9, frameon=True)

    fig.tight_layout()

    figure_13c_png = (
        FIGURE_ROOT /
        "Figure_13C_Prediction_Set_Coverage_Diagnostic.png"
    )
    figure_13c_pdf = (
        FIGURE_ROOT /
        "Figure_13C_Prediction_Set_Coverage_Diagnostic.pdf"
    )

    fig.savefig(
        figure_13c_png,
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        figure_13c_pdf,
        bbox_inches="tight",
    )

    plt.show()

    print("✅ Figure 13C full-scale version generated")
    print("PNG:", figure_13c_png)
    print("PDF:", figure_13c_pdf)

# ## 6. Síntesis automática y notas metodológicas

# %% Cell 11
key_rows = []

# Equivalence counts
if not equivalence_df.empty:
    key_rows.append({
        "section": "equivalence",
        "finding": "pairwise_comparisons_total",
        "value": int(len(equivalence_df)),
    })
    key_rows.append({
        "section": "equivalence",
        "finding": "practically_equivalent_comparisons",
        "value": int(equivalence_df["equivalent_90ci_within_margin"].sum()),
    })

# Best AURC by protocol
if not aurc_aggregate.empty:
    for (exp, protocol), g in aurc_aggregate.groupby(["experiment", "protocol"]):
        best = g.sort_values("aurc_mean").iloc[0]
        key_rows.append({
            "section": "selective_prediction",
            "finding": f"lowest_AURC__{exp}__{protocol}",
            "value": f"{best['model']} | AURC={best['aurc_mean']:.6f}",
        })

# Prediction-set coverage summary
if not prediction_sets_aggregate.empty:
    for (exp, method, alpha), g in prediction_sets_aggregate.groupby(
        ["experiment", "method", "alpha"]
    ):
        key_rows.append({
            "section": "prediction_sets",
            "finding": f"mean_coverage__{exp}__{method}__alpha_{alpha}",
            "value": float(g["empirical_coverage_mean"].mean()),
        })

key_df = pd.DataFrame(key_rows)
key_df.to_csv(RUN_ROOT / "13_key_findings.csv", index=False)

notes = f'''
Paddy Doctor Q1 — Notebook 13 methodological notes

Primary outcome:
- Macro-F1.

Practical equivalence:
- Absolute equivalence margin: ±{EQUIV_DELTA:.3f} Macro-F1.
- {N_BOOT:,} paired, class-stratified bootstrap replicates.
- Same resampled test images are used for all architectures within each replicate.
- Seed-specific Macro-F1 is averaged across seeds inside each bootstrap replicate.
- 95% CI is reported for estimation.
- 90% CI is used for CI-based equivalence assessment.

Selective classification:
- Images are ranked by model confidence.
- Risk = 1 - accuracy.
- AURC summarizes the complete confidence-ranked error trajectory.
- Lower AURC indicates better selective behavior.

Prediction-set diagnostics:
- LAC score: 1 - p_true.
- Nominal coverages: 90% and 95%.
- Raw and temperature-scaled probabilities are evaluated.
- Temperature parameters are inherited from Notebook 09 and were fit only on validation.
- IMPORTANT: validation was previously used for checkpoint selection and temperature scaling.
  Therefore strict split-conformal finite-sample coverage is NOT claimed here.
  Results are interpreted as empirical prediction-set diagnostics and domain-shift sensitivity.
- A formal conformal claim would require an independent calibration set that was not used
  for model selection, hyperparameter tuning, checkpoint selection, or probability calibration.

No network retraining is performed by Notebook 13.
'''

with open(RUN_ROOT / "13_methodology_notes.txt", "w", encoding="utf-8") as f:
    f.write(notes.strip() + "\n")

print(notes)
display(key_df.head(20))

# ## 7. Verificación final de salidas

# %% Cell 12
required_outputs = [
    "13_config.json",
    "13_input_audit.csv",
    "13_prediction_index_audit.csv",
    "13_equivalence_pairwise_5000boot.csv",
    "13_macro_f1_bootstrap_summary_5000boot.csv",
    "13_selective_by_run.csv",
    "13_selective_aggregate.csv",
    "13_risk_coverage_points_by_run.csv",
    "13_aurc_by_run.csv",
    "13_aurc_aggregate.csv",
    "13_prediction_set_diagnostics_by_run.csv",
    "13_prediction_set_diagnostics_aggregate.csv",
    "13_prediction_set_diagnostics_classwise.csv",
    "13_key_findings.csv",
    "13_methodology_notes.txt",
]

status_rows = []
for name in required_outputs:
    p = RUN_ROOT / name
    status_rows.append({
        "file": name,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else None,
    })

status_df = pd.DataFrame(status_rows)
display(status_df)

if not status_df["exists"].all():
    missing = status_df.loc[~status_df["exists"], "file"].tolist()
    raise RuntimeError(f"❌ Faltan salidas: {missing}")

print("✅ Todas las salidas obligatorias existen.")

# ## 8. Crear ZIP para revisión
# 
# El ZIP:
# - queda guardado permanentemente en `PADDY_DOCTOR_Q1/exports/`;
# - incluye tablas, notas y figuras del Notebook 13;
# - intenta incluir también el notebook y el script si ya están presentes en Drive;
# - se descarga automáticamente al navegador para que puedas adjuntarlo luego en ChatGPT.

# %% Cell 13
from google.colab import files

PACKAGE_NAME = "PaddyDoctor_N13_Advanced_Reliability_Results"
TEMP_PACKAGE = Path("/content") / PACKAGE_NAME

if TEMP_PACKAGE.exists():
    shutil.rmtree(TEMP_PACKAGE)

(TEMP_PACKAGE / "results").mkdir(parents=True, exist_ok=True)
(TEMP_PACKAGE / "figures").mkdir(parents=True, exist_ok=True)
(TEMP_PACKAGE / "code").mkdir(parents=True, exist_ok=True)

for p in RUN_ROOT.iterdir():
    if p.is_file():
        shutil.copy2(p, TEMP_PACKAGE / "results" / p.name)

for p in FIGURE_ROOT.iterdir():
    if p.is_file():
        shutil.copy2(p, TEMP_PACKAGE / "figures" / p.name)

candidate_code = [
    NOTEBOOKS_DIR / "13_PaddyDoctor_Advanced_Reliability_Equivalence_Selective_Conformal.ipynb",
    SCRIPTS_DIR / "13_PaddyDoctor_Advanced_Reliability_Equivalence_Selective_Conformal.py",
]
for p in candidate_code:
    if p.exists():
        shutil.copy2(p, TEMP_PACKAGE / "code" / p.name)

manifest_rows = []
for p in sorted(TEMP_PACKAGE.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        manifest_rows.append({
            "relative_path": str(p.relative_to(TEMP_PACKAGE)),
            "size_bytes": p.stat().st_size,
            "sha256": h.hexdigest(),
        })

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(TEMP_PACKAGE / "MANIFEST_SHA256.csv", index=False)

zip_base = EXPORTS_DIR / PACKAGE_NAME
zip_path = Path(shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=TEMP_PACKAGE
))

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print("Ruta:", zip_path)
print("Tamaño MB:", round(zip_path.stat().st_size / (1024**2), 2))
print("Archivos:", len(manifest_rows))
print("Cuando termine la ejecución, selecciona el ZIP descargado y adjúntalo en ChatGPT.")

files.download(str(zip_path))

# ## 9. Fin
# 
# Al terminar, conserva y adjunta:
# 
# `PaddyDoctor_N13_Advanced_Reliability_Results.zip`
# 
# Ese ZIP será suficiente para auditar:
# - equivalencia;
# - bootstrap 5,000;
# - selective prediction;
# - AURC;
# - prediction-set diagnostics;
# - figuras;
# - integridad de archivos mediante SHA-256.

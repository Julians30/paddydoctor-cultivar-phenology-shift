"""Auto-exported from 08_PaddyDoctor_Joint_Statistical_Inference_P1_P4_P5.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 08
# ## Inferencia estadística conjunta: P1 + P4 + P5
# 
# ### Objetivo
# Convertir los resultados descriptivos de los tres experimentos ya congelados en evidencia inferencial reproducible:
# 
# - **P1:** benchmark IID limpio y deduplicado.
# - **P4:** generalización cross-cultivar, 10 clases.
# - **P5:** sensibilidad cross-cultivar con emparejamiento exacto por enfermedad y edad, 5 clases.
# 
# ### Principios estadísticos
# 1. **Macro-F1 es la métrica primaria.**
# 2. Los **tres seeds no se tratan como tres datasets independientes**.
# 3. La incertidumbre de Macro-F1 se estima mediante **bootstrap estratificado por clase**, usando el mismo remuestreo para todos los modelos y promediando los tres seeds dentro de cada réplica.
# 4. Las comparaciones binarias de corrección sobre las **mismas imágenes** usan **McNemar exacto**, por seed, con ajuste **Holm**.
# 5. El efecto global de arquitectura, cambio de dominio e interacción arquitectura×dominio se evalúa de forma secundaria mediante **GEE binomial** con covarianza robusta y agrupamiento por imagen o por par exacto.
# 6. En P4 y P5, la inferencia del efecto cross-domain se realiza manteniendo **fijo el conjunto de imágenes objetivo**:
#    - target ADT45: ADT45→ADT45 vs OTHER→ADT45.
#    - target OTHER: OTHER→OTHER vs ADT45→OTHER.
# 7. La comparación P4↔P5 restringida a las mismas cinco clases se interpreta **solo como análisis de sensibilidad descriptivo**, porque cambian el número de clases del entrenamiento, el subconjunto de imágenes y la cabeza de clasificación.
# 
# ### Carga computacional
# Este notebook **no entrena redes y no necesita GPU**.  
# Puede ejecutarse con runtime **CPU** para no consumir unidades de GPU.

# %% [markdown] Cell 2
# ## 0. Montar Google Drive

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

RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"

P1_ROOT = RESULTS_DIR / "05_P1_30ep_noES"
P4_ROOT = RESULTS_DIR / "06_P4_cross_cultivar_30ep_noES"
P5_ROOT = RESULTS_DIR / "07_P5_exact_age_matched_30ep_noES"

RUN_ROOT = RESULTS_DIR / "08_joint_statistical_inference"
FIGURE_ROOT = FIGURES_DIR / "08_joint_statistical_inference"

for folder in [RUN_ROOT, FIGURE_ROOT, EXPORTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("✅ Proyecto:", PROJECT)
print("P1:", P1_ROOT)
print("P4:", P4_ROOT)
print("P5:", P5_ROOT)
print("Salida:", RUN_ROOT)

# %% [markdown] Cell 4
# ## 1. Librerías y configuración

# %% Cell 5
import re
import json
import math
import itertools
import warnings
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binomtest, spearmanr
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from statsmodels.stats.multitest import multipletests

try:
    from statsmodels.formula.api import gee
    from statsmodels.genmod.families import Binomial
    from statsmodels.genmod.cov_struct import Exchangeable, Independence
except Exception as e:
    raise ImportError(
        "statsmodels no está disponible en este runtime. "
        "Instala statsmodels>=0.14 y vuelve a ejecutar."
    ) from e

ALPHA = 0.05

# 2.000 réplicas ofrecen una buena relación rigor/tiempo.
# Para una corrida editorial final puede subirse a 5.000 si se desea.
N_BOOT = 2000
BOOT_SEED = 20260822

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

P5_CLASSES = [
    "blast",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
]

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

PROTOCOL_ORDER = [
    "P1_IID",
    "P4B_ADT45_to_ADT45",
    "P4A_ADT45_to_OTHER",
    "P4D_OTHER_to_OTHER",
    "P4C_OTHER_to_ADT45",
    "P5B_ADT45_to_ADT45_age_control",
    "P5A_ADT45_to_OTHER_age_matched",
    "P5D_OTHER_to_OTHER_age_control",
    "P5C_OTHER_to_ADT45_age_matched",
]

print(f"✅ Bootstrap: {N_BOOT:,} réplicas")
print("✅ GPU no requerida.")

# %% [markdown] Cell 6
# ## 2. Verificar que P1, P4 y P5 estén completos

# %% Cell 7
expected_roots = {
    "P1": (P1_ROOT, 12),
    "P4": (P4_ROOT, 24),
    "P5": (P5_ROOT, 24),
}

for exp, (root, expected_complete) in expected_roots.items():
    if not root.exists():
        raise FileNotFoundError(
            f"❌ No existe {exp}: {root}. "
            "Asegúrate de haber ejecutado y guardado los notebooks anteriores."
        )

    complete_files = list(root.rglob("COMPLETE.json"))
    print(
        f"{exp}: COMPLETE.json = "
        f"{len(complete_files)}/{expected_complete}"
    )

    if len(complete_files) != expected_complete:
        raise RuntimeError(
            f"❌ {exp} incompleto: se esperaban "
            f"{expected_complete} ejecuciones completas."
        )

print("\n✅ P1, P4 y P5 están completos.")

# %% [markdown] Cell 8
# ## 3. Cargar todas las predicciones a nivel de imagen

# %% Cell 9
def load_prediction_file(path, extra_cols=None):
    extra_cols = extra_cols or []

    df = pd.read_csv(path)

    required = {
        "image_id",
        "y_true",
        "y_pred",
        "confidence",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"❌ Faltan columnas {missing} en {path}"
        )

    keep = [
        c for c in [
            "image_id",
            "y_true",
            "y_pred",
            "confidence",
            "age",
            "variety",
            "domain",
            "matched_pair_id",
        ] + extra_cols
        if c in df.columns
    ]

    return df[keep].copy()


def load_p1_predictions():
    frames = []

    for model in MODELS:
        for seed in SEEDS:
            run_dir = P1_ROOT / f"{model}__seed_{seed}"
            path = run_dir / "test_predictions.csv"

            if not path.exists():
                raise FileNotFoundError(path)

            df = load_prediction_file(path)

            if len(df) != 1476:
                raise RuntimeError(
                    f"❌ P1 {model} seed {seed}: "
                    f"se esperaban 1476 predicciones, hay {len(df)}."
                )

            df["experiment"] = "P1"
            df["protocol"] = "P1_IID"
            df["model_key"] = model
            df["seed"] = int(seed)
            df["train_domain"] = "MIXED"
            df["eval_domain"] = "MIXED"
            df["eval_type"] = "IID"
            df["correct"] = (
                df["y_true"] == df["y_pred"]
            ).astype(int)

            frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_domain_predictions(root, experiment, protocol_map):
    frames = []

    for train_domain in ["ADT45", "OTHER"]:
        for model in MODELS:
            for seed in SEEDS:
                run_dir = (
                    root /
                    f"train_{train_domain}__{model}__seed_{seed}"
                )

                for eval_domain in ["ADT45", "OTHER"]:
                    path = (
                        run_dir /
                        f"eval__{eval_domain}" /
                        "test_predictions.csv"
                    )

                    if not path.exists():
                        raise FileNotFoundError(path)

                    df = load_prediction_file(path)

                    expected_n = 616 if experiment == "P4" else 100

                    if len(df) != expected_n:
                        raise RuntimeError(
                            f"❌ {experiment} {train_domain}->{eval_domain} "
                            f"{model} seed {seed}: esperado {expected_n}, "
                            f"observado {len(df)}."
                        )

                    df["experiment"] = experiment
                    df["protocol"] = protocol_map[
                        (train_domain, eval_domain)
                    ]
                    df["model_key"] = model
                    df["seed"] = int(seed)
                    df["train_domain"] = train_domain
                    df["eval_domain"] = eval_domain
                    df["eval_type"] = (
                        "control"
                        if train_domain == eval_domain
                        else "cross"
                    )
                    df["correct"] = (
                        df["y_true"] == df["y_pred"]
                    ).astype(int)

                    frames.append(df)

    return pd.concat(frames, ignore_index=True)


p1 = load_p1_predictions()
p4 = load_domain_predictions(
    P4_ROOT,
    "P4",
    P4_PROTOCOL_MAP
)
p5 = load_domain_predictions(
    P5_ROOT,
    "P5",
    P5_PROTOCOL_MAP
)

all_predictions = pd.concat(
    [p1, p4, p5],
    ignore_index=True,
    sort=False
)

print("P1 rows:", len(p1))
print("P4 rows:", len(p4))
print("P5 rows:", len(p5))
print("Total rows:", len(all_predictions))

all_predictions.to_csv(
    RUN_ROOT / "08_all_predictions_index.csv",
    index=False
)

# %% [markdown] Cell 10
# ## 4. Auditoría estricta de apareamiento

# %% Cell 11
def assert_same_truth_across_models(df, exp, protocol):
    sub = df[
        (df["experiment"] == exp)
        & (df["protocol"] == protocol)
    ]

    for seed in SEEDS:
        s = sub[sub["seed"] == seed]

        reference = None

        for model in MODELS:
            m = (
                s[s["model_key"] == model]
                [["image_id", "y_true"]]
                .sort_values("image_id")
                .reset_index(drop=True)
            )

            if m["image_id"].duplicated().any():
                raise RuntimeError(
                    f"❌ Duplicados en {exp}/{protocol}/{model}/{seed}"
                )

            if reference is None:
                reference = m
            elif not m.equals(reference):
                raise RuntimeError(
                    f"❌ Test set no idéntico entre modelos: "
                    f"{exp}/{protocol}/seed {seed}"
                )


for protocol in ["P1_IID"]:
    assert_same_truth_across_models(
        all_predictions, "P1", protocol
    )

for protocol in P4_PROTOCOL_MAP.values():
    assert_same_truth_across_models(
        all_predictions, "P4", protocol
    )

for protocol in P5_PROTOCOL_MAP.values():
    assert_same_truth_across_models(
        all_predictions, "P5", protocol
    )


# Verificar que control y cross usan EXACTAMENTE las mismas imágenes
# cuando el dominio objetivo se mantiene fijo.
for exp, df in [("P4", p4), ("P5", p5)]:
    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        for model in MODELS:
            for seed in SEEDS:
                control = (
                    df[
                        (df["model_key"] == model)
                        & (df["seed"] == seed)
                        & (df["train_domain"] == control_source)
                        & (df["eval_domain"] == target_domain)
                    ]
                    [["image_id", "y_true"]]
                    .sort_values("image_id")
                    .reset_index(drop=True)
                )

                cross = (
                    df[
                        (df["model_key"] == model)
                        & (df["seed"] == seed)
                        & (df["train_domain"] == cross_source)
                        & (df["eval_domain"] == target_domain)
                    ]
                    [["image_id", "y_true"]]
                    .sort_values("image_id")
                    .reset_index(drop=True)
                )

                if not control.equals(cross):
                    raise RuntimeError(
                        f"❌ {exp}: target {target_domain} "
                        f"no está apareado para {model}, seed {seed}."
                    )


# Verificación adicional P5: mismo matched_pair_id, label y age
# entre los dos dominios objetivo.
p5_ref = p5[
    (p5["model_key"] == MODELS[0])
    & (p5["seed"] == SEEDS[0])
    & (p5["train_domain"] == "ADT45")
].copy()

adt = (
    p5_ref[p5_ref["eval_domain"] == "ADT45"]
    [["matched_pair_id", "y_true", "age"]]
    .sort_values("matched_pair_id")
    .reset_index(drop=True)
)

other = (
    p5_ref[p5_ref["eval_domain"] == "OTHER"]
    [["matched_pair_id", "y_true", "age"]]
    .sort_values("matched_pair_id")
    .reset_index(drop=True)
)

if not (
    adt["matched_pair_id"].equals(other["matched_pair_id"])
    and adt["y_true"].equals(other["y_true"])
    and adt["age"].equals(other["age"])
):
    raise RuntimeError(
        "❌ P5 no conserva matched_pair_id + label + age entre dominios."
    )

print("✅ Tests idénticos entre arquitecturas.")
print("✅ Control/cross apareados sobre las mismas imágenes objetivo.")
print("✅ P5 conserva mismo par, enfermedad y edad entre ADT45 y OTHER.")

# %% [markdown] Cell 12
# ## 5. Funciones de métricas

# %% Cell 13
def macro_f1_selected(y_true, y_pred, labels):
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)

    f1_values = []

    for label in labels:
        tp = np.sum(
            (y_true == label)
            & (y_pred == label)
        )
        fp = np.sum(
            (y_true != label)
            & (y_pred == label)
        )
        fn = np.sum(
            (y_true == label)
            & (y_pred != label)
        )

        denom = (2 * tp) + fp + fn

        f1 = (
            0.0
            if denom == 0
            else (2 * tp) / denom
        )

        f1_values.append(f1)

    return float(np.mean(f1_values))


def group_metrics(df, labels=None):
    if labels is None:
        labels = sorted(df["y_true"].unique())

    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": macro_f1_selected(
            y_true,
            y_pred,
            labels
        ),
    }


metric_rows = []

for keys, g in all_predictions.groupby(
    [
        "experiment",
        "protocol",
        "model_key",
        "seed",
    ],
    sort=False
):
    experiment, protocol, model, seed = keys

    labels = (
        P5_CLASSES
        if experiment == "P5"
        else sorted(g["y_true"].unique())
    )

    metrics = group_metrics(g, labels=labels)

    metric_rows.append({
        "experiment": experiment,
        "protocol": protocol,
        "model_key": model,
        "seed": int(seed),
        **metrics,
    })

metrics_by_seed = pd.DataFrame(metric_rows)

metrics_by_seed.to_csv(
    RUN_ROOT / "08_metrics_by_protocol_model_seed.csv",
    index=False
)

metrics_summary = (
    metrics_by_seed
    .groupby(
        ["experiment", "protocol", "model_key"]
    )
    .agg(
        n_seeds=("seed", "nunique"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_sd=("macro_f1", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
    )
    .reset_index()
)

metrics_summary["model"] = (
    metrics_summary["model_key"]
    .map(MODEL_SHORT)
)

metrics_summary.to_csv(
    RUN_ROOT / "08_descriptive_summary.csv",
    index=False
)

display(
    metrics_summary.sort_values(
        ["experiment", "protocol", "macro_f1_mean"],
        ascending=[True, True, False]
    ).round(5)
)

# %% [markdown] Cell 14
# ## 6. Bootstrap estratificado de Macro-F1 entre arquitecturas
# 
# La misma muestra bootstrap de imágenes se utiliza para todas las arquitecturas y seeds dentro de cada protocolo.
# 
# Por cada réplica:
# 
# 1. se remuestrean imágenes **dentro de cada clase**;
# 2. se calcula Macro-F1 para cada seed;
# 3. se promedian los tres seeds;
# 4. se calculan diferencias entre arquitecturas.
# 
# Así se preserva el carácter pareado del test set y se evita tratar las semillas como datasets independientes.

# %% Cell 15
def stratified_boot_indices(y_true, rng):
    y_true = np.asarray(y_true, dtype=object)
    sampled = []

    for label in sorted(np.unique(y_true)):
        idx = np.flatnonzero(y_true == label)

        sampled.append(
            rng.choice(
                idx,
                size=len(idx),
                replace=True
            )
        )

    return np.concatenate(sampled)


def prepare_protocol_arrays(df):
    first = (
        df[
            (df["model_key"] == MODELS[0])
            & (df["seed"] == SEEDS[0])
        ]
        [["image_id", "y_true"]]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    units = first["image_id"].to_numpy()
    y_true = first["y_true"].to_numpy()
    labels = sorted(np.unique(y_true))

    preds = {}

    for model in MODELS:
        preds[model] = {}

        for seed in SEEDS:
            temp = (
                df[
                    (df["model_key"] == model)
                    & (df["seed"] == seed)
                ]
                [["image_id", "y_pred"]]
                .set_index("image_id")
                .reindex(units)
            )

            if temp["y_pred"].isna().any():
                raise RuntimeError(
                    f"❌ Falta apareamiento en {model}, seed {seed}."
                )

            preds[model][seed] = (
                temp["y_pred"].to_numpy()
            )

    return units, y_true, labels, preds


def bootstrap_architectures(
    df,
    experiment,
    protocol,
    n_boot=N_BOOT,
    rng_seed=BOOT_SEED
):
    units, y_true, labels, preds = (
        prepare_protocol_arrays(df)
    )

    original = {}

    for model in MODELS:
        original[model] = np.mean([
            macro_f1_selected(
                y_true,
                preds[model][seed],
                labels
            )
            for seed in SEEDS
        ])

    boot_scores = np.zeros(
        (n_boot, len(MODELS)),
        dtype=np.float64
    )

    rng = np.random.default_rng(rng_seed)

    for b in range(n_boot):
        idx = stratified_boot_indices(
            y_true,
            rng
        )

        yt = y_true[idx]

        for m_idx, model in enumerate(MODELS):
            seed_scores = []

            for seed in SEEDS:
                yp = preds[model][seed][idx]

                seed_scores.append(
                    macro_f1_selected(
                        yt,
                        yp,
                        labels
                    )
                )

            boot_scores[b, m_idx] = (
                np.mean(seed_scores)
            )

    pair_rows = []

    for a_idx, b_idx in itertools.combinations(
        range(len(MODELS)), 2
    ):
        model_a = MODELS[a_idx]
        model_b = MODELS[b_idx]

        diffs = (
            boot_scores[:, a_idx]
            - boot_scores[:, b_idx]
        )

        lo, hi = np.quantile(
            diffs,
            [ALPHA / 2, 1 - ALPHA / 2]
        )

        estimate = (
            original[model_a]
            - original[model_b]
        )

        pair_rows.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_a": model_a,
            "model_b": model_b,
            "macro_f1_a": original[model_a],
            "macro_f1_b": original[model_b],
            "delta_macro_f1_a_minus_b": estimate,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "bootstrap_probability_a_gt_b": float(
                np.mean(diffs > 0)
            ),
            "ci_excludes_zero": bool(
                (lo > 0) or (hi < 0)
            ),
            "n_boot": int(n_boot),
        })

    # Probabilidad bootstrap de ocupar el primer lugar.
    winners = np.argmax(
        boot_scores,
        axis=1
    )

    top_rows = []

    for idx, model in enumerate(MODELS):
        top_rows.append({
            "experiment": experiment,
            "protocol": protocol,
            "model_key": model,
            "macro_f1_original_mean": original[model],
            "bootstrap_probability_rank1": float(
                np.mean(winners == idx)
            ),
            "n_boot": int(n_boot),
        })

    return (
        pd.DataFrame(pair_rows),
        pd.DataFrame(top_rows)
    )


architecture_boot_frames = []
rank_probability_frames = []

for experiment, df in [
    ("P1", p1),
    ("P4", p4),
    ("P5", p5),
]:
    protocols = (
        ["P1_IID"]
        if experiment == "P1"
        else (
            list(P4_PROTOCOL_MAP.values())
            if experiment == "P4"
            else list(P5_PROTOCOL_MAP.values())
        )
    )

    for i, protocol in enumerate(protocols):
        print(
            f"Bootstrap arquitecturas: "
            f"{experiment} / {protocol}"
        )

        sub = df[df["protocol"] == protocol]

        pair_df, top_df = bootstrap_architectures(
            sub,
            experiment=experiment,
            protocol=protocol,
            n_boot=N_BOOT,
            rng_seed=BOOT_SEED + i
        )

        architecture_boot_frames.append(pair_df)
        rank_probability_frames.append(top_df)


architecture_bootstrap = pd.concat(
    architecture_boot_frames,
    ignore_index=True
)

rank1_probabilities = pd.concat(
    rank_probability_frames,
    ignore_index=True
)

architecture_bootstrap.to_csv(
    RUN_ROOT /
    "08_pairwise_architecture_macroF1_bootstrap.csv",
    index=False
)

rank1_probabilities.to_csv(
    RUN_ROOT /
    "08_architecture_rank1_bootstrap_probability.csv",
    index=False
)

display(
    architecture_bootstrap[
        architecture_bootstrap["ci_excludes_zero"]
    ].round(5)
)

# %% [markdown] Cell 16
# ## 7. McNemar exacto entre arquitecturas + Holm

# %% Cell 17
def exact_mcnemar_from_correct(a_correct, b_correct):
    a_correct = np.asarray(a_correct, dtype=int)
    b_correct = np.asarray(b_correct, dtype=int)

    b = int(
        np.sum(
            (a_correct == 1)
            & (b_correct == 0)
        )
    )

    c = int(
        np.sum(
            (a_correct == 0)
            & (b_correct == 1)
        )
    )

    n_discordant = b + c

    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            binomtest(
                b,
                n_discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )

    discordance_or = (
        (b + 0.5) / (c + 0.5)
    )

    return b, c, n_discordant, p_value, discordance_or


mcnemar_arch_rows = []

for experiment, df in [
    ("P1", p1),
    ("P4", p4),
    ("P5", p5),
]:
    for protocol in df["protocol"].unique():
        sub = df[df["protocol"] == protocol]

        for seed in SEEDS:
            s = sub[sub["seed"] == seed]

            family_rows = []

            for model_a, model_b in itertools.combinations(
                MODELS, 2
            ):
                a = (
                    s[s["model_key"] == model_a]
                    [["image_id", "correct"]]
                    .rename(
                        columns={
                            "correct": "correct_a"
                        }
                    )
                )

                bdf = (
                    s[s["model_key"] == model_b]
                    [["image_id", "correct"]]
                    .rename(
                        columns={
                            "correct": "correct_b"
                        }
                    )
                )

                merged = a.merge(
                    bdf,
                    on="image_id",
                    validate="one_to_one"
                )

                b_count, c_count, n_disc, p, dor = (
                    exact_mcnemar_from_correct(
                        merged["correct_a"],
                        merged["correct_b"]
                    )
                )

                family_rows.append({
                    "experiment": experiment,
                    "protocol": protocol,
                    "seed": int(seed),
                    "model_a": model_a,
                    "model_b": model_b,
                    "accuracy_a": float(
                        merged["correct_a"].mean()
                    ),
                    "accuracy_b": float(
                        merged["correct_b"].mean()
                    ),
                    "accuracy_delta_a_minus_b": float(
                        merged["correct_a"].mean()
                        - merged["correct_b"].mean()
                    ),
                    "a_correct_b_wrong": b_count,
                    "a_wrong_b_correct": c_count,
                    "n_discordant": n_disc,
                    "discordance_odds_ratio": dor,
                    "p_raw": p,
                })

            pvals = [
                row["p_raw"]
                for row in family_rows
            ]

            reject, p_holm, _, _ = multipletests(
                pvals,
                alpha=ALPHA,
                method="holm"
            )

            for row, q, rj in zip(
                family_rows,
                p_holm,
                reject
            ):
                row["p_holm"] = float(q)
                row["reject_holm_0_05"] = bool(rj)

            mcnemar_arch_rows.extend(
                family_rows
            )


mcnemar_architecture = pd.DataFrame(
    mcnemar_arch_rows
)

mcnemar_architecture.to_csv(
    RUN_ROOT /
    "08_mcnemar_architecture_by_protocol_seed_holm.csv",
    index=False
)

mcnemar_arch_stability = (
    mcnemar_architecture
    .groupby(
        [
            "experiment",
            "protocol",
            "model_a",
            "model_b",
        ]
    )
    .agg(
        significant_seeds_holm=(
            "reject_holm_0_05",
            "sum"
        ),
        mean_accuracy_delta=(
            "accuracy_delta_a_minus_b",
            "mean"
        ),
        min_p_holm=("p_holm", "min"),
        max_p_holm=("p_holm", "max"),
    )
    .reset_index()
)

mcnemar_arch_stability.to_csv(
    RUN_ROOT /
    "08_mcnemar_architecture_stability_across_seeds.csv",
    index=False
)

display(
    mcnemar_arch_stability[
        mcnemar_arch_stability[
            "significant_seeds_holm"
        ] >= 2
    ].round(5)
)

# %% [markdown] Cell 18
# ## 8. Efecto cross-domain con el **mismo target fijo**
# 
# Esta es la comparación inferencial preferida para P4 y P5.
# 
# Ejemplo para target ADT45:
# 
# \[
# ADT45 \rightarrow ADT45
# \quad \text{vs} \quad
# OTHER \rightarrow ADT45
# \]
# 
# Las dos predicciones se hacen sobre **exactamente las mismas imágenes ADT45**.
# 
# Esto permite:
# 
# - bootstrap pareado de la diferencia de Macro-F1;
# - McNemar exacto de corrección por imagen.
# 
# Se repite análogamente para target OTHER.

# %% Cell 19
def prepare_fixed_target_arrays(
    df,
    target_domain,
    model
):
    control_source = target_domain
    cross_source = (
        "OTHER"
        if target_domain == "ADT45"
        else "ADT45"
    )

    reference = (
        df[
            (df["model_key"] == model)
            & (df["seed"] == SEEDS[0])
            & (df["train_domain"] == control_source)
            & (df["eval_domain"] == target_domain)
        ]
        [["image_id", "y_true"]]
        .sort_values("image_id")
        .reset_index(drop=True)
    )

    units = reference["image_id"].to_numpy()
    y_true = reference["y_true"].to_numpy()
    labels = sorted(np.unique(y_true))

    control_preds = {}
    cross_preds = {}

    for seed in SEEDS:
        c = (
            df[
                (df["model_key"] == model)
                & (df["seed"] == seed)
                & (df["train_domain"] == control_source)
                & (df["eval_domain"] == target_domain)
            ]
            [["image_id", "y_pred"]]
            .set_index("image_id")
            .reindex(units)
        )

        x = (
            df[
                (df["model_key"] == model)
                & (df["seed"] == seed)
                & (df["train_domain"] == cross_source)
                & (df["eval_domain"] == target_domain)
            ]
            [["image_id", "y_pred"]]
            .set_index("image_id")
            .reindex(units)
        )

        if c["y_pred"].isna().any() or x["y_pred"].isna().any():
            raise RuntimeError(
                "❌ Apareamiento incompleto en fixed-target."
            )

        control_preds[seed] = c["y_pred"].to_numpy()
        cross_preds[seed] = x["y_pred"].to_numpy()

    return (
        units,
        y_true,
        labels,
        control_preds,
        cross_preds,
        control_source,
        cross_source,
    )


def bootstrap_fixed_target_drop(
    df,
    experiment,
    target_domain,
    model,
    n_boot=N_BOOT,
    rng_seed=BOOT_SEED
):
    (
        units,
        y_true,
        labels,
        control_preds,
        cross_preds,
        control_source,
        cross_source,
    ) = prepare_fixed_target_arrays(
        df,
        target_domain,
        model
    )

    original_control = np.mean([
        macro_f1_selected(
            y_true,
            control_preds[seed],
            labels
        )
        for seed in SEEDS
    ])

    original_cross = np.mean([
        macro_f1_selected(
            y_true,
            cross_preds[seed],
            labels
        )
        for seed in SEEDS
    ])

    rng = np.random.default_rng(rng_seed)
    drops = np.zeros(n_boot)

    for b in range(n_boot):
        idx = stratified_boot_indices(
            y_true,
            rng
        )

        yt = y_true[idx]

        control_score = np.mean([
            macro_f1_selected(
                yt,
                control_preds[seed][idx],
                labels
            )
            for seed in SEEDS
        ])

        cross_score = np.mean([
            macro_f1_selected(
                yt,
                cross_preds[seed][idx],
                labels
            )
            for seed in SEEDS
        ])

        drops[b] = (
            control_score
            - cross_score
        )

    lo, hi = np.quantile(
        drops,
        [ALPHA / 2, 1 - ALPHA / 2]
    )

    return {
        "experiment": experiment,
        "target_domain": target_domain,
        "model_key": model,
        "control_source": control_source,
        "cross_source": cross_source,
        "control_macro_f1": original_control,
        "cross_macro_f1": original_cross,
        "drop_control_minus_cross": (
            original_control
            - original_cross
        ),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_probability_drop_gt_0": float(
            np.mean(drops > 0)
        ),
        "ci_excludes_zero": bool(
            (lo > 0) or (hi < 0)
        ),
        "n_boot": int(n_boot),
    }


fixed_target_boot_rows = []

for experiment, df in [
    ("P4", p4),
    ("P5", p5),
]:
    for target_idx, target_domain in enumerate(
        ["ADT45", "OTHER"]
    ):
        for model_idx, model in enumerate(MODELS):
            print(
                f"Fixed-target bootstrap: "
                f"{experiment} target={target_domain} "
                f"{MODEL_SHORT[model]}"
            )

            fixed_target_boot_rows.append(
                bootstrap_fixed_target_drop(
                    df,
                    experiment,
                    target_domain,
                    model,
                    n_boot=N_BOOT,
                    rng_seed=(
                        BOOT_SEED
                        + 100
                        + target_idx * 10
                        + model_idx
                        + (1000 if experiment == "P5" else 0)
                    )
                )
            )


fixed_target_bootstrap = pd.DataFrame(
    fixed_target_boot_rows
)

fixed_target_bootstrap.to_csv(
    RUN_ROOT /
    "08_fixed_target_cross_domain_macroF1_bootstrap.csv",
    index=False
)

display(
    fixed_target_bootstrap.round(5)
)

# %% [markdown] Cell 20
# ## 9. McNemar del efecto cross-domain sobre el mismo target

# %% Cell 21
fixed_target_mcnemar_rows = []

for experiment, df in [
    ("P4", p4),
    ("P5", p5),
]:
    for target_domain in ["ADT45", "OTHER"]:
        control_source = target_domain
        cross_source = (
            "OTHER"
            if target_domain == "ADT45"
            else "ADT45"
        )

        family_rows = []

        for model in MODELS:
            for seed in SEEDS:
                control = (
                    df[
                        (df["model_key"] == model)
                        & (df["seed"] == seed)
                        & (df["train_domain"] == control_source)
                        & (df["eval_domain"] == target_domain)
                    ]
                    [["image_id", "correct"]]
                    .rename(
                        columns={
                            "correct": "control_correct"
                        }
                    )
                )

                cross = (
                    df[
                        (df["model_key"] == model)
                        & (df["seed"] == seed)
                        & (df["train_domain"] == cross_source)
                        & (df["eval_domain"] == target_domain)
                    ]
                    [["image_id", "correct"]]
                    .rename(
                        columns={
                            "correct": "cross_correct"
                        }
                    )
                )

                merged = control.merge(
                    cross,
                    on="image_id",
                    validate="one_to_one"
                )

                b, c, n_disc, p, dor = (
                    exact_mcnemar_from_correct(
                        merged["control_correct"],
                        merged["cross_correct"]
                    )
                )

                family_rows.append({
                    "experiment": experiment,
                    "target_domain": target_domain,
                    "model_key": model,
                    "seed": int(seed),
                    "control_source": control_source,
                    "cross_source": cross_source,
                    "control_accuracy": float(
                        merged[
                            "control_correct"
                        ].mean()
                    ),
                    "cross_accuracy": float(
                        merged[
                            "cross_correct"
                        ].mean()
                    ),
                    "accuracy_drop": float(
                        merged[
                            "control_correct"
                        ].mean()
                        - merged[
                            "cross_correct"
                        ].mean()
                    ),
                    "control_correct_cross_wrong": b,
                    "control_wrong_cross_correct": c,
                    "n_discordant": n_disc,
                    "discordance_odds_ratio": dor,
                    "p_raw": p,
                })

        # Holm dentro de cada experimento × target:
        # 4 modelos × 3 seeds = 12 contrastes.
        pvals = [
            row["p_raw"]
            for row in family_rows
        ]

        reject, p_holm, _, _ = multipletests(
            pvals,
            alpha=ALPHA,
            method="holm"
        )

        for row, q, rj in zip(
            family_rows,
            p_holm,
            reject
        ):
            row["p_holm"] = float(q)
            row["reject_holm_0_05"] = bool(rj)

        fixed_target_mcnemar_rows.extend(
            family_rows
        )


fixed_target_mcnemar = pd.DataFrame(
    fixed_target_mcnemar_rows
)

fixed_target_mcnemar.to_csv(
    RUN_ROOT /
    "08_fixed_target_cross_domain_mcnemar_holm.csv",
    index=False
)

fixed_target_mcnemar_stability = (
    fixed_target_mcnemar
    .groupby(
        [
            "experiment",
            "target_domain",
            "model_key",
        ]
    )
    .agg(
        significant_seeds_holm=(
            "reject_holm_0_05",
            "sum"
        ),
        mean_accuracy_drop=(
            "accuracy_drop",
            "mean"
        ),
        max_p_holm=("p_holm", "max"),
    )
    .reset_index()
)

fixed_target_mcnemar_stability.to_csv(
    RUN_ROOT /
    "08_fixed_target_mcnemar_stability.csv",
    index=False
)

display(
    fixed_target_mcnemar_stability.round(5)
)

# %% [markdown] Cell 22
# ## 10. P5: ventaja bidireccional de **domain match** bajo matching exacto de edad
# 
# Para cada `matched_pair_id` tenemos:
# 
# - una imagen ADT45;
# - una imagen OTHER;
# - misma enfermedad;
# - misma edad.
# 
# Para cada arquitectura y seed se construye el diseño 2×2:
# 
# | Entrenamiento | Imagen ADT45 | Imagen OTHER |
# |---|---|---|
# | ADT45 | control | cross |
# | OTHER | cross | control |
# 
# La estimación principal es:
# 
# \[
# \text{Domain-match advantage}
# =
# \frac{F1_{A\to A}+F1_{O\to O}}{2}
# -
# \frac{F1_{O\to A}+F1_{A\to O}}{2}
# \]
# 
# El bootstrap se realiza a nivel de `matched_pair_id`.

# %% Cell 23
def get_p5_pair_arrays(model):
    reference = (
        p5[
            (p5["model_key"] == model)
            & (p5["seed"] == SEEDS[0])
            & (p5["train_domain"] == "ADT45")
            & (p5["eval_domain"] == "ADT45")
        ]
        [["matched_pair_id", "y_true", "age"]]
        .sort_values("matched_pair_id")
        .reset_index(drop=True)
    )

    pair_ids = reference[
        "matched_pair_id"
    ].to_numpy()

    y_true = reference["y_true"].to_numpy()
    ages = reference["age"].to_numpy()
    labels = P5_CLASSES

    arrays = {}

    for train_domain in ["ADT45", "OTHER"]:
        for eval_domain in ["ADT45", "OTHER"]:
            for seed in SEEDS:
                temp = (
                    p5[
                        (p5["model_key"] == model)
                        & (p5["seed"] == seed)
                        & (p5["train_domain"] == train_domain)
                        & (p5["eval_domain"] == eval_domain)
                    ]
                    [
                        [
                            "matched_pair_id",
                            "y_true",
                            "age",
                            "y_pred",
                        ]
                    ]
                    .set_index("matched_pair_id")
                    .reindex(pair_ids)
                )

                if temp["y_pred"].isna().any():
                    raise RuntimeError(
                        "❌ P5 pair matching incompleto."
                    )

                # Deben coincidir enfermedad y edad.
                if not np.array_equal(
                    temp["y_true"].to_numpy(),
                    y_true
                ):
                    raise RuntimeError(
                        "❌ P5: label cambia dentro del par."
                    )

                if not np.array_equal(
                    temp["age"].to_numpy(),
                    ages
                ):
                    raise RuntimeError(
                        "❌ P5: age cambia dentro del par."
                    )

                arrays[
                    (train_domain, eval_domain, seed)
                ] = temp["y_pred"].to_numpy()

    return pair_ids, y_true, labels, arrays


p5_domain_match_rows = []

for model_idx, model in enumerate(MODELS):
    (
        pair_ids,
        y_true,
        labels,
        arrays,
    ) = get_p5_pair_arrays(model)

    def bidirectional_scores(idx=None):
        if idx is None:
            idx = np.arange(len(y_true))

        yt = y_true[idx]

        controls = []
        crosses = []

        for seed in SEEDS:
            f_aa = macro_f1_selected(
                yt,
                arrays[
                    ("ADT45", "ADT45", seed)
                ][idx],
                labels
            )

            f_oo = macro_f1_selected(
                yt,
                arrays[
                    ("OTHER", "OTHER", seed)
                ][idx],
                labels
            )

            f_oa = macro_f1_selected(
                yt,
                arrays[
                    ("OTHER", "ADT45", seed)
                ][idx],
                labels
            )

            f_ao = macro_f1_selected(
                yt,
                arrays[
                    ("ADT45", "OTHER", seed)
                ][idx],
                labels
            )

            controls.append(
                (f_aa + f_oo) / 2
            )

            crosses.append(
                (f_oa + f_ao) / 2
            )

        return (
            float(np.mean(controls)),
            float(np.mean(crosses))
        )

    control_original, cross_original = (
        bidirectional_scores()
    )

    rng = np.random.default_rng(
        BOOT_SEED + 3000 + model_idx
    )

    advantages = np.zeros(N_BOOT)

    for b in range(N_BOOT):
        idx = stratified_boot_indices(
            y_true,
            rng
        )

        c_score, x_score = (
            bidirectional_scores(idx)
        )

        advantages[b] = (
            c_score - x_score
        )

    lo, hi = np.quantile(
        advantages,
        [ALPHA / 2, 1 - ALPHA / 2]
    )

    p5_domain_match_rows.append({
        "model_key": model,
        "control_bidirectional_macro_f1": control_original,
        "cross_bidirectional_macro_f1": cross_original,
        "domain_match_advantage": (
            control_original
            - cross_original
        ),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "bootstrap_probability_advantage_gt_0": float(
            np.mean(advantages > 0)
        ),
        "ci_excludes_zero": bool(
            (lo > 0) or (hi < 0)
        ),
        "n_pairs": int(len(pair_ids)),
        "n_boot": int(N_BOOT),
    })


p5_domain_match_advantage = pd.DataFrame(
    p5_domain_match_rows
)

p5_domain_match_advantage.to_csv(
    RUN_ROOT /
    "08_P5_exact_age_domain_match_advantage.csv",
    index=False
)

display(
    p5_domain_match_advantage.round(5)
)

# %% [markdown] Cell 24
# ## 11. Sensibilidad P4→P5 restringiendo P4 a las mismas cinco clases
# 
# **Importante:** esta sección es descriptiva. No se calcula un p-valor causal para “cuánto explica la edad”.
# 
# Aunque se restrinja P4 a las cinco clases de P5, siguen cambiando:
# 
# - el conjunto de entrenamiento;
# - el número de clases de salida durante el entrenamiento (10 en P4, 5 en P5);
# - las imágenes seleccionadas;
# - el diseño exacto de matching.
# 
# Por eso la diferencia P4−P5 solo se usa para mostrar si el gap se atenúa o persiste.

# %% Cell 25
def fixed_target_gap_metric(
    df,
    experiment,
    target_domain,
    model,
    seed,
    labels
):
    control_source = target_domain
    cross_source = (
        "OTHER"
        if target_domain == "ADT45"
        else "ADT45"
    )

    control = df[
        (df["model_key"] == model)
        & (df["seed"] == seed)
        & (df["train_domain"] == control_source)
        & (df["eval_domain"] == target_domain)
    ].copy()

    cross = df[
        (df["model_key"] == model)
        & (df["seed"] == seed)
        & (df["train_domain"] == cross_source)
        & (df["eval_domain"] == target_domain)
    ].copy()

    if experiment == "P4":
        control = control[
            control["y_true"].isin(labels)
        ]
        cross = cross[
            cross["y_true"].isin(labels)
        ]

    control_f1 = macro_f1_selected(
        control["y_true"],
        control["y_pred"],
        labels
    )

    cross_f1 = macro_f1_selected(
        cross["y_true"],
        cross["y_pred"],
        labels
    )

    return control_f1, cross_f1


sensitivity_rows = []

for experiment, df in [
    ("P4_5class_restricted", p4),
    ("P5_exact_age", p5),
]:
    exp_source = (
        "P4"
        if experiment.startswith("P4")
        else "P5"
    )

    for target_domain in ["ADT45", "OTHER"]:
        for model in MODELS:
            for seed in SEEDS:
                c, x = fixed_target_gap_metric(
                    df,
                    exp_source,
                    target_domain,
                    model,
                    seed,
                    labels=P5_CLASSES
                )

                sensitivity_rows.append({
                    "analysis": experiment,
                    "target_domain": target_domain,
                    "model_key": model,
                    "seed": int(seed),
                    "control_macro_f1": c,
                    "cross_macro_f1": x,
                    "gap_control_minus_cross": c - x,
                })


sensitivity_by_seed = pd.DataFrame(
    sensitivity_rows
)

sensitivity_summary = (
    sensitivity_by_seed
    .groupby(
        ["analysis", "model_key"]
    )
    .agg(
        mean_control_macro_f1=(
            "control_macro_f1",
            "mean"
        ),
        mean_cross_macro_f1=(
            "cross_macro_f1",
            "mean"
        ),
        mean_gap=(
            "gap_control_minus_cross",
            "mean"
        ),
        gap_sd=(
            "gap_control_minus_cross",
            "std"
        ),
    )
    .reset_index()
)

pivot_gap = (
    sensitivity_summary
    .pivot(
        index="model_key",
        columns="analysis",
        values="mean_gap"
    )
    .reset_index()
)

pivot_gap[
    "descriptive_gap_reduction_P4minusP5"
] = (
    pivot_gap["P4_5class_restricted"]
    - pivot_gap["P5_exact_age"]
)

pivot_gap[
    "interpretation"
] = (
    "Descriptive sensitivity only; not causal decomposition"
)

sensitivity_by_seed.to_csv(
    RUN_ROOT /
    "08_P4_P5_5class_sensitivity_by_seed.csv",
    index=False
)

pivot_gap.to_csv(
    RUN_ROOT /
    "08_P4_P5_5class_gap_sensitivity_summary.csv",
    index=False
)

display(pivot_gap.round(5))

# %% [markdown] Cell 26
# ## 12. GEE binomial: arquitectura × cambio de dominio
# 
# El GEE modela **corrección por imagen** (0/1), no Macro-F1.
# 
# Por tanto:
# 
# - **bootstrap de Macro-F1** = inferencia principal de desempeño;
# - **GEE** = análisis secundario a nivel de imagen.
# 
# Modelos:
# 
# ### P1
# \[
# correct \sim architecture + seed
# \]
# 
# cluster = `image_id`.
# 
# ### P4
# \[
# correct \sim architecture * eval\_type + target\_domain + seed
# \]
# 
# cluster = `image_id`.
# 
# ### P5
# La misma fórmula, pero cluster = `matched_pair_id`, aprovechando el matching exacto.

# %% Cell 27
def fit_gee_and_save(
    df,
    experiment,
    formula,
    group_col
):
    data = df.copy()

    data["model_key"] = (
        pd.Categorical(
            data["model_key"],
            categories=MODELS
        )
    )

    data["seed"] = data["seed"].astype(str)

    try:
        model = gee(
            formula,
            groups=group_col,
            data=data,
            family=Binomial(),
            cov_struct=Exchangeable()
        )

        result = model.fit(
            maxiter=100
        )

        cov_used = "Exchangeable"

    except Exception as e:
        warnings.warn(
            f"{experiment}: Exchangeable falló ({e}). "
            "Se reintenta con Independence."
        )

        model = gee(
            formula,
            groups=group_col,
            data=data,
            family=Binomial(),
            cov_struct=Independence()
        )

        result = model.fit(
            maxiter=100
        )

        cov_used = "Independence"

    params = pd.DataFrame({
        "term": result.params.index,
        "coef_log_odds": result.params.values,
        "std_error_robust": result.bse.values,
        "p_value": result.pvalues.values,
    })

    ci = result.conf_int()

    params["ci95_logodds_low"] = ci[0].values
    params["ci95_logodds_high"] = ci[1].values

    params["odds_ratio"] = np.exp(
        params["coef_log_odds"]
    )

    params["or_ci95_low"] = np.exp(
        params["ci95_logodds_low"]
    )

    params["or_ci95_high"] = np.exp(
        params["ci95_logodds_high"]
    )

    params["experiment"] = experiment
    params["working_correlation"] = cov_used

    params.to_csv(
        RUN_ROOT /
        f"08_GEE_{experiment}_coefficients.csv",
        index=False
    )

    with open(
        RUN_ROOT /
        f"08_GEE_{experiment}_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(result.summary().as_text())

    try:
        wald = result.wald_test_terms(
            skip_single=False,
            scalar=True
        )

        wald_df = (
            wald.table
            .reset_index()
            .rename(
                columns={
                    "index": "term",
                    "statistic": "wald_chi2",
                    "pvalue": "p_value",
                }
            )
        )

    except Exception as e:
        warnings.warn(
            f"No se pudo generar Wald terms para {experiment}: {e}"
        )

        wald_df = pd.DataFrame()

    if not wald_df.empty:
        wald_df["experiment"] = experiment
        wald_df["working_correlation"] = cov_used

        wald_df.to_csv(
            RUN_ROOT /
            f"08_GEE_{experiment}_wald_terms.csv",
            index=False
        )

    return params, wald_df


gee_outputs = {}

p1_gee, p1_wald = fit_gee_and_save(
    p1,
    experiment="P1",
    formula=(
        "correct ~ "
        "C(model_key) + C(seed)"
    ),
    group_col="image_id"
)

p4_gee, p4_wald = fit_gee_and_save(
    p4,
    experiment="P4",
    formula=(
        "correct ~ "
        "C(model_key) * C(eval_type) "
        "+ C(eval_domain) + C(seed)"
    ),
    group_col="image_id"
)

p5_gee, p5_wald = fit_gee_and_save(
    p5,
    experiment="P5",
    formula=(
        "correct ~ "
        "C(model_key) * C(eval_type) "
        "+ C(eval_domain) + C(seed)"
    ),
    group_col="matched_pair_id"
)

wald_all = pd.concat(
    [p1_wald, p4_wald, p5_wald],
    ignore_index=True
)

wald_all.to_csv(
    RUN_ROOT /
    "08_GEE_all_wald_terms.csv",
    index=False
)

display(
    wald_all.round(6)
)

# %% [markdown] Cell 28
# ## 13. Cambios de ranking: IID vs cross-domain

# %% Cell 29
ranking_table = (
    metrics_summary[
        [
            "experiment",
            "protocol",
            "model_key",
            "macro_f1_mean",
        ]
    ]
    .copy()
)

ranking_table["rank"] = (
    ranking_table
    .groupby(
        ["experiment", "protocol"]
    )["macro_f1_mean"]
    .rank(
        method="min",
        ascending=False
    )
)

ranking_table.to_csv(
    RUN_ROOT /
    "08_architecture_rankings_by_protocol.csv",
    index=False
)

p1_rank = (
    ranking_table[
        ranking_table["protocol"] == "P1_IID"
    ]
    .set_index("model_key")["rank"]
)

rank_corr_rows = []

for experiment, protocol in [
    ("P4", "P4A_ADT45_to_OTHER"),
    ("P4", "P4C_OTHER_to_ADT45"),
    ("P5", "P5A_ADT45_to_OTHER_age_matched"),
    ("P5", "P5C_OTHER_to_ADT45_age_matched"),
]:
    r = (
        ranking_table[
            (ranking_table["experiment"] == experiment)
            & (ranking_table["protocol"] == protocol)
        ]
        .set_index("model_key")["rank"]
        .reindex(MODELS)
    )

    base = p1_rank.reindex(MODELS)

    rho, p = spearmanr(
        base.to_numpy(),
        r.to_numpy()
    )

    rank_corr_rows.append({
        "comparison": f"P1_IID vs {protocol}",
        "spearman_rho": float(rho),
        "p_value_descriptive_only": float(p),
        "note": (
            "Only 4 architectures; use descriptively, "
            "not as primary inference."
        )
    })


rank_correlations = pd.DataFrame(
    rank_corr_rows
)

rank_correlations.to_csv(
    RUN_ROOT /
    "08_IID_vs_cross_rank_correlations.csv",
    index=False
)

display(
    ranking_table.sort_values(
        ["experiment", "protocol", "rank"]
    )
)

display(
    rank_correlations.round(5)
)

# %% [markdown] Cell 30
# ## 14. Consolidar hallazgos inferenciales clave

# %% Cell 31
key_rows = []

# Bootstrap fixed-target
for _, row in fixed_target_bootstrap.iterrows():
    key_rows.append({
        "analysis_family": "Fixed-target Macro-F1 bootstrap",
        "experiment": row["experiment"],
        "contrast": (
            f"target {row['target_domain']}: "
            f"{row['control_source']} control vs "
            f"{row['cross_source']} cross"
        ),
        "model_key": row["model_key"],
        "estimate": row["drop_control_minus_cross"],
        "ci95_low": row["ci95_low"],
        "ci95_high": row["ci95_high"],
        "p_or_probability": (
            row["bootstrap_probability_drop_gt_0"]
        ),
        "significant_or_supported": (
            row["ci_excludes_zero"]
        ),
    })

# P5 bidirectional matched
for _, row in p5_domain_match_advantage.iterrows():
    key_rows.append({
        "analysis_family": (
            "P5 exact-age bidirectional domain-match bootstrap"
        ),
        "experiment": "P5",
        "contrast": "matched domain vs cross domain",
        "model_key": row["model_key"],
        "estimate": row["domain_match_advantage"],
        "ci95_low": row["ci95_low"],
        "ci95_high": row["ci95_high"],
        "p_or_probability": (
            row[
                "bootstrap_probability_advantage_gt_0"
            ]
        ),
        "significant_or_supported": (
            row["ci_excludes_zero"]
        ),
    })

# GEE Wald terms
for _, row in wald_all.iterrows():
    if row["term"] in [
        "C(model_key)",
        "C(eval_type)",
        "C(model_key):C(eval_type)",
    ]:
        key_rows.append({
            "analysis_family": "GEE Wald",
            "experiment": row["experiment"],
            "contrast": row["term"],
            "model_key": "ALL",
            "estimate": row["wald_chi2"],
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_or_probability": row["p_value"],
            "significant_or_supported": bool(
                row["p_value"] < ALPHA
            ),
        })


key_inference = pd.DataFrame(
    key_rows
)

key_inference.to_csv(
    RUN_ROOT /
    "08_key_inference_summary.csv",
    index=False
)

display(key_inference.round(6))

# %% [markdown] Cell 32
# ## 15. Figuras para revisión científica

# %% Cell 33
# Figura P1
p1_plot = (
    metrics_summary[
        metrics_summary["experiment"] == "P1"
    ]
    .sort_values(
        "macro_f1_mean",
        ascending=False
    )
)

fig = plt.figure(figsize=(9, 5))

plt.errorbar(
    np.arange(len(p1_plot)),
    p1_plot["macro_f1_mean"],
    yerr=p1_plot["macro_f1_sd"].fillna(0),
    fmt="o",
    capsize=5
)

plt.xticks(
    np.arange(len(p1_plot)),
    p1_plot["model"].tolist(),
    rotation=20,
    ha="right"
)

plt.ylabel("Macro-F1 (mean ± SD across seeds)")
plt.xlabel("Architecture")
plt.title("P1 — Clean IID benchmark")
plt.tight_layout()

plt.savefig(
    FIGURE_ROOT /
    "08_P1_macroF1_architectures.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()


# Figura P4 fixed-target drop
p4_drop = (
    fixed_target_bootstrap[
        fixed_target_bootstrap["experiment"] == "P4"
    ]
    .copy()
)

p4_drop["label"] = (
    p4_drop["model_key"].map(MODEL_SHORT)
    + " | target "
    + p4_drop["target_domain"]
)

fig = plt.figure(figsize=(10, 6))

y = np.arange(len(p4_drop))

xerr = np.vstack([
    p4_drop["drop_control_minus_cross"]
    - p4_drop["ci95_low"],
    p4_drop["ci95_high"]
    - p4_drop["drop_control_minus_cross"],
])

plt.errorbar(
    p4_drop["drop_control_minus_cross"],
    y,
    xerr=xerr,
    fmt="o",
    capsize=4
)

plt.yticks(y, p4_drop["label"])
plt.axvline(0, linestyle="--")
plt.xlabel("Macro-F1 drop: matched-domain control − cross-domain")
plt.ylabel("Architecture / target")
plt.title("P4 — Cross-cultivar degradation, fixed target")
plt.tight_layout()

plt.savefig(
    FIGURE_ROOT /
    "08_P4_fixed_target_macroF1_drop_CI.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()


# Figura P5 fixed-target drop
p5_drop = (
    fixed_target_bootstrap[
        fixed_target_bootstrap["experiment"] == "P5"
    ]
    .copy()
)

p5_drop["label"] = (
    p5_drop["model_key"].map(MODEL_SHORT)
    + " | target "
    + p5_drop["target_domain"]
)

fig = plt.figure(figsize=(10, 6))

y = np.arange(len(p5_drop))

xerr = np.vstack([
    p5_drop["drop_control_minus_cross"]
    - p5_drop["ci95_low"],
    p5_drop["ci95_high"]
    - p5_drop["drop_control_minus_cross"],
])

plt.errorbar(
    p5_drop["drop_control_minus_cross"],
    y,
    xerr=xerr,
    fmt="o",
    capsize=4
)

plt.yticks(y, p5_drop["label"])
plt.axvline(0, linestyle="--")
plt.xlabel("Macro-F1 drop: matched-domain control − cross-domain")
plt.ylabel("Architecture / target")
plt.title("P5 — Exact-age-matched cross-domain degradation")
plt.tight_layout()

plt.savefig(
    FIGURE_ROOT /
    "08_P5_fixed_target_macroF1_drop_CI.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()


# Figura P5 ventaja bidireccional
plot_df = p5_domain_match_advantage.copy()
plot_df["model"] = plot_df["model_key"].map(MODEL_SHORT)

fig = plt.figure(figsize=(9, 5))

y = np.arange(len(plot_df))

xerr = np.vstack([
    plot_df["domain_match_advantage"]
    - plot_df["ci95_low"],
    plot_df["ci95_high"]
    - plot_df["domain_match_advantage"],
])

plt.errorbar(
    plot_df["domain_match_advantage"],
    y,
    xerr=xerr,
    fmt="o",
    capsize=5
)

plt.yticks(y, plot_df["model"])
plt.axvline(0, linestyle="--")
plt.xlabel("Bidirectional Macro-F1 domain-match advantage")
plt.ylabel("Architecture")
plt.title("P5 — Disease- and age-matched domain effect")
plt.tight_layout()

plt.savefig(
    FIGURE_ROOT /
    "08_P5_bidirectional_domain_match_advantage_CI.png",
    dpi=250,
    bbox_inches="tight"
)

plt.show()

print("✅ Figuras guardadas en:", FIGURE_ROOT)

# %% [markdown] Cell 34
# ## 16. Nota metodológica reproducible

# %% Cell 35
methodology_note = f"""
PADDY DOCTOR Q1 — NOTEBOOK 08
JOINT STATISTICAL INFERENCE

Primary outcome:
- Macro-F1.

Training seeds:
- {SEEDS}.
- Seeds are repeated stochastic training realizations, not independent datasets.

Macro-F1 inference:
- Stratified bootstrap by true class.
- Same resampled test units are applied to all architectures being compared.
- Within each bootstrap replicate, Macro-F1 is computed separately for each seed
  and then averaged across the three observed seeds.
- Number of bootstrap replicates: {N_BOOT}.
- Percentile 95% confidence intervals.
- These intervals quantify test-sample uncertainty conditional on the observed
  training seeds; between-seed SD is reported separately.

Architecture comparisons:
- Pairwise bootstrap differences in Macro-F1.
- Exact McNemar tests on binary correctness for each seed separately.
- Holm correction across the six architecture pairs within each
  experiment/protocol/seed family.

Cross-domain inference:
- The preferred comparison keeps the target images fixed.
- Target ADT45: ADT45->ADT45 control vs OTHER->ADT45 cross.
- Target OTHER: OTHER->OTHER control vs ADT45->OTHER cross.
- This enables paired bootstrap and exact McNemar on identical target images.

P5:
- Exact matching by disease label and plant age.
- Bidirectional domain-match advantage is bootstrapped at matched_pair_id level.
- P5 reduces observable phenological confounding but is not interpreted as
  complete causal identification of cultivar.

GEE:
- Secondary image-level analysis of correctness (0/1).
- Binomial family with robust covariance.
- P1 clustered by image_id.
- P4 clustered by image_id.
- P5 clustered by matched_pair_id.
- Architecture x eval_type interaction evaluates whether degradation under
  cross-domain testing depends on architecture.

P4 vs P5 sensitivity:
- P4 is descriptively restricted to P5's five target classes.
- No causal p-value is assigned to the P4-P5 gap reduction because the training
  task, image subset, and classifier output dimensionality differ.
"""

with open(
    RUN_ROOT /
    "08_statistical_methodology_notes.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(methodology_note)

print(methodology_note)

# %% [markdown] Cell 36
# ## 17. Verificación final

# %% Cell 37
required_outputs = [
    "08_descriptive_summary.csv",
    "08_pairwise_architecture_macroF1_bootstrap.csv",
    "08_architecture_rank1_bootstrap_probability.csv",
    "08_mcnemar_architecture_by_protocol_seed_holm.csv",
    "08_fixed_target_cross_domain_macroF1_bootstrap.csv",
    "08_fixed_target_cross_domain_mcnemar_holm.csv",
    "08_P5_exact_age_domain_match_advantage.csv",
    "08_P4_P5_5class_gap_sensitivity_summary.csv",
    "08_GEE_all_wald_terms.csv",
    "08_key_inference_summary.csv",
    "08_statistical_methodology_notes.txt",
]

missing = [
    name
    for name in required_outputs
    if not (RUN_ROOT / name).exists()
]

if missing:
    raise RuntimeError(
        f"❌ Faltan outputs de Notebook 08: {missing}"
    )

print("=" * 80)
print("✅ NOTEBOOK 08 COMPLETADO")
print("=" * 80)
print("Bootstrap:", N_BOOT)
print("P1 + P4 + P5 integrados.")
print("McNemar exacto + Holm: OK")
print("Fixed-target paired inference: OK")
print("P5 exact-age domain-match analysis: OK")
print("GEE: OK")
print("Sensitivity P4→P5: OK")
print("Figuras:", FIGURE_ROOT)

# %% [markdown] Cell 38
# ## 18. Exportar resultados del Notebook 08 a ZIP

# %% Cell 39
from google.colab import files
import shutil

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

package_name = (
    "08_PaddyDoctor_Joint_Statistical_Inference_"
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

# Copiar únicamente inferencia y figuras finales,
# no duplicar predicciones masivas de P1/P4/P5.
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
    temp_package / "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 08\n"
        "Joint Statistical Inference: P1 + P4 + P5\n\n"
        f"Bootstrap replicates: {N_BOOT}\n"
        "Primary outcome: Macro-F1\n"
        "Exact McNemar + Holm correction included.\n"
        "GEE image-level correctness models included.\n"
        "P5 exact-age matched domain analysis included.\n"
        "P4-P5 five-class comparison is descriptive sensitivity only.\n"
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

print("✅ ZIP creado:")
print(zip_path)
print(f"Tamaño: {size_mb:.2f} MB")

files.download(zip_path)

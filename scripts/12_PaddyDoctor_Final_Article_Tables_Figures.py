"""Auto-exported from 12_PaddyDoctor_Final_Article_Tables_Figures.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 12
# ## Consolidación final para el artículo: tablas, figuras y paquete editorial
# 
# ### Objetivo
# Este notebook **no realiza nuevos experimentos**. Integra los resultados ya cerrados de:
# 
# - **P1** — evaluación IID limpia.
# - **P4** — generalización cross-cultivar, 10 clases.
# - **P5** — sensibilidad exact-age-matched, 5 clases.
# - **Notebook 08** — inferencia estadística.
# - **Notebook 09** — calibración probabilística.
# - **Notebook 10** — explainability/faithfulness.
# - **Notebook 11** — eficiencia computacional.
# 
# ### Productos
# 1. Tablas principales del manuscrito.
# 2. Tablas suplementarias.
# 3. Figuras finales en **PNG 300 dpi + PDF**.
# 4. Resumen narrativo de resultados listo para apoyar la redacción.
# 5. Matriz de mensajes científicos: qué podemos afirmar y qué debemos evitar.
# 6. Paquete ZIP final para redacción del paper.
# 
# ### Regla
# No se recalculan modelos ni se modifica ningún resultado.  
# Las tablas se construyen únicamente a partir de los CSV ya generados.

# %% [markdown] Cell 2
# ## 0. Montar Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path
import os
import time

try:
    drive.mount("/content/drive", force_remount=False)
except Exception:
    drive.mount("/content/drive", force_remount=True)

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
EXPORTS_DIR = PROJECT / "exports"

R08 = RESULTS_DIR / "08_joint_statistical_inference"
R09 = RESULTS_DIR / "09_calibration_reliability"
R10 = RESULTS_DIR / "10_explainability_faithfulness"
R11 = RESULTS_DIR / "11_computational_efficiency"

RUN_ROOT = RESULTS_DIR / "12_final_article_package"
MAIN_TABLES = RUN_ROOT / "main_tables"
SUPP_TABLES = RUN_ROOT / "supplementary_tables"
TEXT_ROOT = RUN_ROOT / "manuscript_support"

FIGURE_ROOT = FIGURES_DIR / "12_final_article_figures"

for folder in [
    RUN_ROOT,
    MAIN_TABLES,
    SUPP_TABLES,
    TEXT_ROOT,
    FIGURE_ROOT,
    EXPORTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

print("✅ Proyecto:", PROJECT)
print("✅ Salida N12:", RUN_ROOT)

# %% [markdown] Cell 4
# ## 1. Librerías y configuración editorial

# %% Cell 5
import json
import shutil
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_ORDER = [
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

PROTOCOL_DISPLAY = {
    "P1_IID": "P1 — IID clean",
    "P4B_ADT45_to_ADT45": "P4 — ADT45→ADT45",
    "P4A_ADT45_to_OTHER": "P4 — ADT45→OTHER",
    "P4D_OTHER_to_OTHER": "P4 — OTHER→OTHER",
    "P4C_OTHER_to_ADT45": "P4 — OTHER→ADT45",
    "P5B_ADT45_to_ADT45_age_control": "P5 — ADT45→ADT45",
    "P5A_ADT45_to_OTHER_age_matched": "P5 — ADT45→OTHER",
    "P5D_OTHER_to_OTHER_age_control": "P5 — OTHER→OTHER",
    "P5C_OTHER_to_ADT45_age_matched": "P5 — OTHER→ADT45",
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

DPI = 300

CONFIG = {
    "notebook": "12_final_article_tables_figures",
    "model_order": MODEL_ORDER,
    "protocol_order": PROTOCOL_ORDER,
    "figure_dpi": DPI,
    "no_new_experiments": True,
}

with open(
    RUN_ROOT / "12_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(CONFIG, f, indent=2)

print(json.dumps(CONFIG, indent=2))

# %% [markdown] Cell 6
# ## 2. Verificar todos los outputs requeridos

# %% Cell 7
REQUIRED = {
    "08_descriptive": R08 / "08_descriptive_summary.csv",
    "08_fixed_target": R08 / "08_fixed_target_cross_domain_macroF1_bootstrap.csv",
    "08_p5_matched": R08 / "08_P5_exact_age_domain_match_advantage.csv",
    "08_sensitivity": R08 / "08_P4_P5_5class_gap_sensitivity_summary.csv",
    "08_gee": R08 / "08_GEE_all_wald_terms.csv",
    "08_pairwise": R08 / "08_pairwise_architecture_macroF1_bootstrap.csv",
    "08_mcnemar": R08 / "08_fixed_target_cross_domain_mcnemar_holm.csv",
    "08_rank": R08 / "08_architecture_rank1_bootstrap_probability.csv",

    "09_aggregate": R09 / "09_calibration_aggregate_summary.csv",
    "09_fixed_target": R09 / "09_fixed_target_calibration_shift_summary.csv",
    "09_transfer": R09 / "09_temperature_scaling_transfer_summary.csv",
    "09_temp": R09 / "09_temperature_validation_summary.csv",

    "10_faithfulness": R10 / "10_faithfulness_summary.csv",
    "10_seed_stability": R10 / "10_cross_seed_saliency_stability_summary.csv",
    "10_control_cross": R10 / "10_control_vs_cross_saliency_similarity_summary.csv",
    "10_boot": R10 / "10_control_vs_cross_faithfulness_paired_bootstrap.csv",
    "10_audit": R10 / "10_prediction_reproduction_audit.csv",

    "11_efficiency": R11 / "11_efficiency_primary_summary.csv",
    "11_all_benchmark": R11 / "11_latency_throughput_vram_all_settings.csv",
    "11_complexity": R11 / "11_model_complexity.csv",
}

missing = {
    key: str(path)
    for key, path in REQUIRED.items()
    if not path.exists()
}

if missing:
    raise FileNotFoundError(
        "❌ Faltan outputs previos. Ejecuta N08–N11 antes de N12:\n"
        + json.dumps(missing, indent=2)
    )

print("✅ Todos los outputs N08–N11 están disponibles.")

# %% [markdown] Cell 8
# ## 3. Cargar resultados

# %% Cell 9
D08 = pd.read_csv(REQUIRED["08_descriptive"])
FT08 = pd.read_csv(REQUIRED["08_fixed_target"])
P5MATCH = pd.read_csv(REQUIRED["08_p5_matched"])
SENS = pd.read_csv(REQUIRED["08_sensitivity"])
GEE = pd.read_csv(REQUIRED["08_gee"])
PAIRWISE = pd.read_csv(REQUIRED["08_pairwise"])
MCNEMAR = pd.read_csv(REQUIRED["08_mcnemar"])
RANKBOOT = pd.read_csv(REQUIRED["08_rank"])

CAL = pd.read_csv(REQUIRED["09_aggregate"])
CAL_FT = pd.read_csv(REQUIRED["09_fixed_target"])
CAL_TRANSFER = pd.read_csv(REQUIRED["09_transfer"])
TEMP = pd.read_csv(REQUIRED["09_temp"])

FAITH = pd.read_csv(REQUIRED["10_faithfulness"])
SEEDSTAB = pd.read_csv(REQUIRED["10_seed_stability"])
SALSHIFT = pd.read_csv(REQUIRED["10_control_cross"])
FAITHBOOT = pd.read_csv(REQUIRED["10_boot"])
XAI_AUDIT = pd.read_csv(REQUIRED["10_audit"])

EFF = pd.read_csv(REQUIRED["11_efficiency"])
BENCH = pd.read_csv(REQUIRED["11_all_benchmark"])
COMPLEXITY = pd.read_csv(REQUIRED["11_complexity"])

print("✅ Resultados cargados.")
print("N08 descriptive:", D08.shape)
print("N09 calibration:", CAL.shape)
print("N10 faithfulness:", FAITH.shape)
print("N11 efficiency:", EFF.shape)

# %% [markdown] Cell 10
# ## 4. Tabla 1 — Diseño experimental

# %% Cell 11
TABLE1 = pd.DataFrame([
    {
        "Experiment": "P1",
        "Scientific role": "Clean IID reference",
        "Classes": 10,
        "Train images": 7377,
        "Validation images": 1476,
        "Test images": 1476,
        "Domains": "Mixed",
        "Control": "Leakage-safe group-clean IID",
        "Primary interpretation": "In-distribution upper reference",
    },
    {
        "Experiment": "P4",
        "Scientific role": "Realistic cross-cultivar/domain shift",
        "Classes": 10,
        "Train images": 1848,
        "Validation images": 616,
        "Test images": 616,
        "Domains": "ADT45 vs OTHER",
        "Control": "Class-matched between domains; age not exactly matched",
        "Primary interpretation": "Domain/cultivar-associated shift, potentially phenology-confounded",
    },
    {
        "Experiment": "P5",
        "Scientific role": "Phenology-controlled sensitivity analysis",
        "Classes": 5,
        "Train images": 300,
        "Validation images": 100,
        "Test images": 100,
        "Domains": "ADT45 vs OTHER",
        "Control": "Exact disease + age matching; pair-level split",
        "Primary interpretation": "Tests whether cross-domain degradation persists after observable age control",
    },
])

TABLE1.to_csv(
    MAIN_TABLES / "Table1_Experimental_Design.csv",
    index=False
)

display(TABLE1)

# %% [markdown] Cell 12
# ## 5. Tabla 2 — Rendimiento predictivo principal

# %% Cell 13
perf = D08.copy()

perf["Model"] = perf["model_key"].map(MODEL_SHORT)
perf["Protocol"] = perf["protocol"].map(PROTOCOL_DISPLAY)

TABLE2 = perf[
    [
        "experiment",
        "Protocol",
        "Model",
        "macro_f1_mean",
        "macro_f1_sd",
        "balanced_accuracy_mean",
        "balanced_accuracy_sd",
        "accuracy_mean",
        "accuracy_sd",
    ]
].copy()

TABLE2 = TABLE2.rename(columns={
    "experiment": "Experiment",
    "macro_f1_mean": "Macro-F1 mean",
    "macro_f1_sd": "Macro-F1 SD",
    "balanced_accuracy_mean": "Balanced accuracy mean",
    "balanced_accuracy_sd": "Balanced accuracy SD",
    "accuracy_mean": "Accuracy mean",
    "accuracy_sd": "Accuracy SD",
})

TABLE2.to_csv(
    MAIN_TABLES / "Table2_Predictive_Performance.csv",
    index=False
)

display(TABLE2.round(4))

# %% [markdown] Cell 14
# ## 6. Tabla 3 — Domain shift con target fijo e IC95%

# %% Cell 15
TABLE3 = FT08.copy()

TABLE3["Model"] = TABLE3["model_key"].map(MODEL_SHORT)

TABLE3 = TABLE3[
    [
        "experiment",
        "target_domain",
        "Model",
        "control_source",
        "cross_source",
        "control_macro_f1",
        "cross_macro_f1",
        "drop_control_minus_cross",
        "ci95_low",
        "ci95_high",
        "bootstrap_probability_drop_gt_0",
        "ci_excludes_zero",
    ]
].rename(columns={
    "experiment": "Experiment",
    "target_domain": "Target domain",
    "control_source": "Matched-domain source",
    "cross_source": "Cross-domain source",
    "control_macro_f1": "Control Macro-F1",
    "cross_macro_f1": "Cross-domain Macro-F1",
    "drop_control_minus_cross": "Macro-F1 drop",
    "ci95_low": "95% CI low",
    "ci95_high": "95% CI high",
    "bootstrap_probability_drop_gt_0": "Pboot(drop>0)",
    "ci_excludes_zero": "CI excludes 0",
})

TABLE3.to_csv(
    MAIN_TABLES / "Table3_Fixed_Target_Domain_Shift.csv",
    index=False
)

display(TABLE3.round(4))

# %% [markdown] Cell 16
# ## 7. Tabla 4 — P5 exact-age-matched: ventaja de domain match

# %% Cell 17
TABLE4 = P5MATCH.copy()

TABLE4["Model"] = TABLE4["model_key"].map(MODEL_SHORT)

TABLE4 = TABLE4[
    [
        "Model",
        "control_bidirectional_macro_f1",
        "cross_bidirectional_macro_f1",
        "domain_match_advantage",
        "ci95_low",
        "ci95_high",
        "bootstrap_probability_advantage_gt_0",
        "n_pairs",
    ]
].rename(columns={
    "control_bidirectional_macro_f1": "Bidirectional control Macro-F1",
    "cross_bidirectional_macro_f1": "Bidirectional cross Macro-F1",
    "domain_match_advantage": "Domain-match advantage",
    "ci95_low": "95% CI low",
    "ci95_high": "95% CI high",
    "bootstrap_probability_advantage_gt_0": "Pboot(advantage>0)",
    "n_pairs": "Matched test pairs",
})

TABLE4.to_csv(
    MAIN_TABLES / "Table4_P5_ExactAge_DomainMatch.csv",
    index=False
)

display(TABLE4.round(4))

# %% [markdown] Cell 18
# ## 8. Tabla 5 — Calibración probabilística

# %% Cell 19
cal_work = CAL.copy()

# Resumen por experimento y tipo de evaluación, promediando arquitecturas/protocolos
# dentro del mismo rol (control/cross/IID). La tabla detallada se conserva en suplemento.
TABLE5 = (
    cal_work
    .groupby(
        ["experiment", "eval_type"],
        dropna=False
    )
    .agg(
        Raw_ECE=("raw_ece_mean", "mean"),
        TS_ECE=("ts_ece_mean", "mean"),
        Raw_NLL=("raw_nll_mean", "mean"),
        TS_NLL=("ts_nll_mean", "mean"),
        Raw_Brier=("raw_brier_mean", "mean"),
        TS_Brier=("ts_brier_mean", "mean"),
        Raw_overconfidence=("raw_overconfidence_mean", "mean"),
        TS_overconfidence=("ts_overconfidence_mean", "mean"),
    )
    .reset_index()
    .rename(columns={
        "experiment": "Experiment",
        "eval_type": "Evaluation type",
    })
)

TABLE5.to_csv(
    MAIN_TABLES / "Table5_Calibration_Summary.csv",
    index=False
)

display(TABLE5.round(4))

# %% [markdown] Cell 20
# ## 9. Tabla 6 — Estabilidad de explicaciones

# %% Cell 21
# Estabilidad entre seeds por experimento/condición.
stab = SEEDSTAB.copy()

def protocol_to_eval_type(protocol):
    if protocol == "P1_IID":
        return "IID"
    if "to_ADT45" in protocol:
        left = protocol.split("_to_")[0]
        return "control" if left.endswith("ADT45") else "cross"
    if "to_OTHER" in protocol:
        left = protocol.split("_to_")[0]
        return "control" if left.endswith("OTHER") else "cross"
    return "unknown"

stab["Evaluation type"] = stab["protocol"].map(protocol_to_eval_type)

seed_summary = (
    stab
    .groupby(
        ["experiment", "Evaluation type", "model_key"]
    )
    .agg(
        Cross_seed_Spearman=("mean_spearman", "mean"),
    )
    .reset_index()
)

sal = SALSHIFT.copy()

sal_summary = (
    sal
    .groupby(
        ["experiment", "model_key"]
    )
    .agg(
        Control_vs_cross_Spearman=("mean_spearman", "mean"),
    )
    .reset_index()
)

TABLE6 = seed_summary.merge(
    sal_summary,
    on=["experiment", "model_key"],
    how="left"
)

TABLE6["Model"] = TABLE6["model_key"].map(MODEL_SHORT)

TABLE6 = TABLE6[
    [
        "experiment",
        "Evaluation type",
        "Model",
        "Cross_seed_Spearman",
        "Control_vs_cross_Spearman",
    ]
].rename(columns={
    "experiment": "Experiment",
})

TABLE6.to_csv(
    MAIN_TABLES / "Table6_Explainability_Stability.csv",
    index=False
)

display(TABLE6.round(4))

# %% [markdown] Cell 22
# ## 10. Tabla 7 — Eficiencia computacional
# 
# **Corrección editorial importante:**  
# Para latencia individual se usa **FP32 batch=1**, porque en la L4 fue más rápido que AMP para batch 1.  
# Para throughput y VRAM se mantiene **AMP/FP16 batch=32**.

# %% Cell 23
fp32_b1 = (
    BENCH[
        (BENCH["precision"] == "fp32")
        & (BENCH["batch_size"] == 1)
    ][
        [
            "model_key",
            "latency_mean_ms_per_image",
            "batch_latency_p95_ms",
        ]
    ]
    .rename(columns={
        "latency_mean_ms_per_image": "Latency FP32 b1 ms/image",
        "batch_latency_p95_ms": "Latency FP32 b1 p95 ms",
    })
)

amp_b32 = (
    BENCH[
        (BENCH["precision"] == "amp_fp16")
        & (BENCH["batch_size"] == 32)
    ][
        [
            "model_key",
            "throughput_images_per_s",
            "vram_peak_total_mb",
        ]
    ]
    .rename(columns={
        "throughput_images_per_s": "Throughput AMP b32 img/s",
        "vram_peak_total_mb": "VRAM AMP b32 peak MB",
    })
)

TABLE7 = (
    COMPLEXITY[
        [
            "model_key",
            "parameters_millions",
            "checkpoint_size_mb",
            "gflops_fvcore",
            "unsupported_ops_json",
        ]
    ]
    .merge(fp32_b1, on="model_key", validate="one_to_one")
    .merge(amp_b32, on="model_key", validate="one_to_one")
)

TABLE7["Model"] = TABLE7["model_key"].map(MODEL_SHORT)

TABLE7 = TABLE7[
    [
        "Model",
        "parameters_millions",
        "checkpoint_size_mb",
        "gflops_fvcore",
        "Latency FP32 b1 ms/image",
        "Latency FP32 b1 p95 ms",
        "Throughput AMP b32 img/s",
        "VRAM AMP b32 peak MB",
        "unsupported_ops_json",
    ]
].rename(columns={
    "parameters_millions": "Parameters (M)",
    "checkpoint_size_mb": "Checkpoint (MB)",
    "gflops_fvcore": "fvcore-counted GFLOPs",
    "unsupported_ops_json": "fvcore unsupported ops",
})

TABLE7.to_csv(
    MAIN_TABLES / "Table7_Computational_Efficiency.csv",
    index=False
)

display(TABLE7.round(3))

# %% [markdown] Cell 24
# ## 11. Tabla 8 — Análisis de sensibilidad P4→P5

# %% Cell 25
TABLE8 = SENS.copy()

TABLE8["Model"] = TABLE8["model_key"].map(MODEL_SHORT)

TABLE8 = TABLE8[
    [
        "Model",
        "P4_5class_restricted",
        "P5_exact_age",
        "descriptive_gap_reduction_P4minusP5",
        "interpretation",
    ]
].rename(columns={
    "P4_5class_restricted": "P4 gap — restricted to P5 classes",
    "P5_exact_age": "P5 exact-age gap",
    "descriptive_gap_reduction_P4minusP5": "Descriptive gap reduction",
    "interpretation": "Interpretation",
})

TABLE8.to_csv(
    MAIN_TABLES / "Table8_P4_P5_Sensitivity.csv",
    index=False
)

display(TABLE8.round(4))

# %% [markdown] Cell 26
# ## 12. Material suplementario

# %% Cell 27
# Guardar tablas completas y no resumidas.
supplementary = {
    "S01_Pairwise_Architecture_Bootstrap.csv": PAIRWISE,
    "S02_Fixed_Target_McNemar_Holm.csv": MCNEMAR,
    "S03_GEE_Wald_Terms.csv": GEE,
    "S04_Architecture_Rank1_Bootstrap.csv": RANKBOOT,
    "S05_Calibration_Detailed.csv": CAL,
    "S06_Calibration_Fixed_Target.csv": CAL_FT,
    "S07_Temperature_Transfer.csv": CAL_TRANSFER,
    "S08_Temperature_Validation.csv": TEMP,
    "S09_Faithfulness_Detailed.csv": FAITH,
    "S10_Explainability_CrossSeed.csv": SEEDSTAB,
    "S11_Explainability_ControlCross.csv": SALSHIFT,
    "S12_Faithfulness_Bootstrap.csv": FAITHBOOT,
    "S13_XAI_Reproduction_Audit.csv": XAI_AUDIT,
    "S14_Efficiency_All_Settings.csv": BENCH,
}

for name, df in supplementary.items():
    df.to_csv(
        SUPP_TABLES / name,
        index=False
    )

print("✅ Tablas suplementarias:", len(supplementary))

# %% [markdown] Cell 28
# ## 13. Figura 1 — Macro-F1 en P1, P4 y P5

# %% Cell 29
plot_df = D08.copy()

plot_df["Protocol display"] = (
    plot_df["protocol"].map(PROTOCOL_DISPLAY)
)

plot_df["Model"] = (
    plot_df["model_key"].map(MODEL_SHORT)
)

# Una figura por arquitectura: evita sobrecargar la visualización.
for model_key in MODEL_ORDER:
    sub = (
        plot_df[
            plot_df["model_key"] == model_key
        ]
        .set_index("protocol")
        .reindex(PROTOCOL_ORDER)
        .reset_index()
    )

    sub = sub[
        sub["macro_f1_mean"].notna()
    ]

    x = np.arange(len(sub))

    fig = plt.figure(figsize=(11, 5))

    plt.errorbar(
        x,
        sub["macro_f1_mean"],
        yerr=sub["macro_f1_sd"].fillna(0),
        fmt="o",
        capsize=4
    )

    plt.xticks(
        x,
        sub["Protocol display"],
        rotation=55,
        ha="right"
    )

    plt.ylim(0, 1.05)
    plt.ylabel("Macro-F1")
    plt.xlabel("Evaluation protocol")
    plt.title(
        f"Predictive performance across evaluation regimes — "
        f"{MODEL_SHORT[model_key]}"
    )

    plt.tight_layout()

    base = (
        FIGURE_ROOT /
        f"Figure1_Performance_{model_key}"
    )

    plt.savefig(
        str(base) + ".png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.savefig(
        str(base) + ".pdf",
        bbox_inches="tight"
    )

    plt.show()

print("✅ Figure 1 series generated.")

# %% [markdown] Cell 30
# ## 14. Figura 2 — Caída de Macro-F1 por domain shift con IC95%

# %% Cell 31
for experiment in ["P4", "P5"]:
    sub = FT08[
        FT08["experiment"] == experiment
    ].copy()

    sub["label"] = (
        sub["model_key"].map(MODEL_SHORT)
        + " | target "
        + sub["target_domain"]
    )

    x = sub["drop_control_minus_cross"].to_numpy()

    xerr = np.vstack([
        x - sub["ci95_low"].to_numpy(),
        sub["ci95_high"].to_numpy() - x,
    ])

    y = np.arange(len(sub))

    fig = plt.figure(figsize=(10, 6))

    plt.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.yticks(
        y,
        sub["label"]
    )

    plt.xlabel(
        "Macro-F1 drop: matched-domain control − cross-domain"
    )

    plt.ylabel(
        "Architecture / target domain"
    )

    plt.title(
        f"{experiment} — fixed-target domain-shift degradation"
    )

    plt.tight_layout()

    base = (
        FIGURE_ROOT /
        f"Figure2_{experiment}_FixedTarget_Drop"
    )

    plt.savefig(
        str(base) + ".png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.savefig(
        str(base) + ".pdf",
        bbox_inches="tight"
    )

    plt.show()

print("✅ Figure 2 generated.")

# %% [markdown] Cell 32
# ## 15. Figura 3 — Calibración: control vs cross-domain

# %% Cell 33
for experiment in ["P4", "P5"]:
    sub = (
        CAL[
            CAL["experiment"] == experiment
        ]
        .groupby(
            ["eval_type"]
        )
        .agg(
            raw_ece=("raw_ece_mean", "mean"),
            ts_ece=("ts_ece_mean", "mean"),
        )
        .reset_index()
    )

    # Mantener solo control/cross.
    sub = sub[
        sub["eval_type"].isin(
            ["control", "cross"]
        )
    ]

    x = np.arange(len(sub))

    fig = plt.figure(figsize=(7, 5))

    width = 0.35

    plt.bar(
        x - width / 2,
        sub["raw_ece"],
        width=width,
        label="Raw"
    )

    plt.bar(
        x + width / 2,
        sub["ts_ece"],
        width=width,
        label="Temperature-scaled"
    )

    plt.xticks(
        x,
        sub["eval_type"]
    )

    plt.ylabel(
        "Expected Calibration Error (15 bins)"
    )

    plt.xlabel(
        "Evaluation type"
    )

    plt.title(
        f"{experiment} — calibration under domain shift"
    )

    plt.legend()
    plt.tight_layout()

    base = (
        FIGURE_ROOT /
        f"Figure3_{experiment}_Calibration"
    )

    plt.savefig(
        str(base) + ".png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.savefig(
        str(base) + ".pdf",
        bbox_inches="tight"
    )

    plt.show()

print("✅ Figure 3 generated.")

# %% [markdown] Cell 34
# ## 16. Figura 4 — Estabilidad de explicaciones

# %% Cell 35
# Control-vs-cross saliency similarity: una figura por experimento.
for experiment in ["P4", "P5"]:
    sub = (
        SALSHIFT[
            SALSHIFT["experiment"] == experiment
        ]
        .groupby(
            ["model_key"]
        )
        .agg(
            mean_spearman=("mean_spearman", "mean")
        )
        .reset_index()
    )

    sub["Model"] = (
        sub["model_key"].map(MODEL_SHORT)
    )

    fig = plt.figure(figsize=(8, 5))

    plt.bar(
        sub["Model"],
        sub["mean_spearman"]
    )

    plt.ylim(0, 1)
    plt.ylabel(
        "Spearman similarity: control vs cross-domain saliency"
    )

    plt.xlabel(
        "Architecture"
    )

    plt.title(
        f"{experiment} — explanation stability under domain shift"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    base = (
        FIGURE_ROOT /
        f"Figure4_{experiment}_Saliency_Stability"
    )

    plt.savefig(
        str(base) + ".png",
        dpi=DPI,
        bbox_inches="tight"
    )

    plt.savefig(
        str(base) + ".pdf",
        bbox_inches="tight"
    )

    plt.show()

print("✅ Figure 4 generated.")

# %% [markdown] Cell 36
# ## 17. Figura 5 — Eficiencia: latencia y throughput

# %% Cell 37
# Latencia individual FP32 batch=1.
lat = TABLE7.copy()

fig = plt.figure(figsize=(8, 5))

plt.bar(
    lat["Model"],
    lat["Latency FP32 b1 ms/image"]
)

plt.ylabel(
    "Latency (ms/image)"
)

plt.xlabel(
    "Architecture"
)

plt.title(
    "Single-image inference latency — FP32, batch 1"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

base = (
    FIGURE_ROOT /
    "Figure5A_Latency_FP32_Batch1"
)

plt.savefig(
    str(base) + ".png",
    dpi=DPI,
    bbox_inches="tight"
)

plt.savefig(
    str(base) + ".pdf",
    bbox_inches="tight"
)

plt.show()


# Throughput AMP batch=32.
fig = plt.figure(figsize=(8, 5))

plt.bar(
    lat["Model"],
    lat["Throughput AMP b32 img/s"]
)

plt.ylabel(
    "Images/s"
)

plt.xlabel(
    "Architecture"
)

plt.title(
    "Batch inference throughput — AMP/FP16, batch 32"
)

plt.xticks(
    rotation=20,
    ha="right"
)

plt.tight_layout()

base = (
    FIGURE_ROOT /
    "Figure5B_Throughput_AMP_Batch32"
)

plt.savefig(
    str(base) + ".png",
    dpi=DPI,
    bbox_inches="tight"
)

plt.savefig(
    str(base) + ".pdf",
    bbox_inches="tight"
)

plt.show()

print("✅ Figure 5 generated.")

# %% [markdown] Cell 38
# ## 18. Figura 6 — Trade-off entre robustez P5 y eficiencia
# 
# Esta figura cruza dos dimensiones distintas:
# 
# - eje X: throughput AMP/FP16 batch 32;
# - eje Y: Macro-F1 cross-domain bidireccional en P5 exact-age-matched.
# 
# No pretende definir un “ganador universal”; muestra el compromiso **robustez–eficiencia**.

# %% Cell 39
trade = P5MATCH[
    [
        "model_key",
        "cross_bidirectional_macro_f1",
    ]
].merge(
    TABLE7[
        [
            "Model",
            "Throughput AMP b32 img/s",
            "Parameters (M)",
        ]
    ],
    left_on=P5MATCH["model_key"].map(MODEL_SHORT),
    right_on="Model",
    how="left"
)

# El merge anterior crea key_0 en algunas versiones de pandas.
if "key_0" in trade.columns:
    trade = trade.drop(columns=["key_0"])

fig = plt.figure(figsize=(8, 6))

plt.scatter(
    trade["Throughput AMP b32 img/s"],
    trade["cross_bidirectional_macro_f1"],
    s=trade["Parameters (M)"] * 8
)

for _, row in trade.iterrows():
    plt.annotate(
        row["Model"],
        (
            row["Throughput AMP b32 img/s"],
            row["cross_bidirectional_macro_f1"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )

plt.xlabel(
    "Throughput — AMP/FP16, batch 32 (images/s)"
)

plt.ylabel(
    "P5 bidirectional cross-domain Macro-F1"
)

plt.title(
    "Robustness–efficiency trade-off"
)

plt.tight_layout()

base = (
    FIGURE_ROOT /
    "Figure6_Robustness_Efficiency_Tradeoff"
)

plt.savefig(
    str(base) + ".png",
    dpi=DPI,
    bbox_inches="tight"
)

plt.savefig(
    str(base) + ".pdf",
    bbox_inches="tight"
)

plt.show()

print("✅ Figure 6 generated.")

# %% [markdown] Cell 40
# ## 19. Mensajes científicos defendibles

# %% Cell 41
# Extraer términos GEE relevantes.
def gee_p(experiment, term):
    row = GEE[
        (GEE["experiment"] == experiment)
        & (GEE["term"] == term)
    ]

    if len(row) != 1:
        return np.nan

    return float(
        row.iloc[0]["p_value"]
    )

p4_shift = gee_p("P4", "C(eval_type)")
p4_interaction = gee_p(
    "P4",
    "C(model_key):C(eval_type)"
)

p5_shift = gee_p("P5", "C(eval_type)")
p5_interaction = gee_p(
    "P5",
    "C(model_key):C(eval_type)"
)

# Calibración promedio.
def avg_cal(exp, eval_type, column):
    x = CAL[
        (CAL["experiment"] == exp)
        & (CAL["eval_type"] == eval_type)
    ]

    return float(
        x[column].mean()
    )

p4_ece_control = avg_cal(
    "P4", "control", "raw_ece_mean"
)
p4_ece_cross = avg_cal(
    "P4", "cross", "raw_ece_mean"
)

p5_ece_control = avg_cal(
    "P5", "control", "raw_ece_mean"
)
p5_ece_cross = avg_cal(
    "P5", "cross", "raw_ece_mean"
)

# P1 top model and comparison.
p1 = D08[
    D08["experiment"] == "P1"
].copy()

p1_top = p1.sort_values(
    "macro_f1_mean",
    ascending=False
).iloc[0]

# XAI audit.
xai_mismatches = int(
    XAI_AUDIT[
        "prediction_mismatches_vs_saved_test"
    ].sum()
)

DEFENSIBLE = [
    "High IID performance does not guarantee cross-domain generalization.",
    "Cross-domain degradation is large and statistically supported when target images are held fixed.",
    "The degradation persists in P5 after exact matching by disease label and plant age.",
    "P4→P5 gap reduction is descriptive sensitivity evidence, not a causal decomposition of phenology.",
    "Domain shift substantially deteriorates probabilistic calibration and increases overconfidence.",
    "Temperature scaling fitted only on source validation improves calibration in many settings but does not restore source-domain reliability.",
    "Architecture affects predictive performance, but no single architecture dominates all IID, robustness, calibration and efficiency dimensions.",
    "Explainability stability is architecture-dependent; attribution maps should not be interpreted as validated lesion localization.",
    "ViT-S/16 provides a strong efficiency profile on the NVIDIA L4, but this does not imply that it is the most robust architecture under every shift direction.",
]

AVOID = [
    "Do not claim a pure causal cultivar effect: acquisition/source/context differences may remain.",
    "Do not claim that age explains a specific percentage of the gap.",
    "Do not call Swin-Tiny the definitive IID winner over ViT-S/16 unless the pairwise CI excludes zero.",
    "Do not claim ConvNeXt is universally more robust than all Transformers; architecture×shift interaction is not significant in P5.",
    "Do not interpret Grad-CAM/Attention Rollout as expert-validated lesion localization.",
    "Do not present fvcore GFLOPs as exact total operations because unsupported operators were reported.",
    "Do not compare latency/throughput directly with studies using different hardware, precision, batch size or input resolution.",
]

messages = {
    "key_statistics": {
        "P4_GEE_domain_shift_p": p4_shift,
        "P4_GEE_architecture_x_shift_p": p4_interaction,
        "P5_GEE_domain_shift_p": p5_shift,
        "P5_GEE_architecture_x_shift_p": p5_interaction,
        "P4_raw_ECE_control_mean": p4_ece_control,
        "P4_raw_ECE_cross_mean": p4_ece_cross,
        "P5_raw_ECE_control_mean": p5_ece_control,
        "P5_raw_ECE_cross_mean": p5_ece_cross,
        "P1_highest_mean_model": str(p1_top["model"]),
        "P1_highest_mean_macroF1": float(p1_top["macro_f1_mean"]),
        "XAI_prediction_reproduction_mismatches": xai_mismatches,
    },
    "defensible_claims": DEFENSIBLE,
    "claims_to_avoid": AVOID,
}

with open(
    TEXT_ROOT /
    "12_Scientific_Claims_Guardrails.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        messages,
        f,
        indent=2
    )

print("### Defensible claims")
for item in DEFENSIBLE:
    print("•", item)

print("\n### Claims to avoid")
for item in AVOID:
    print("•", item)

# %% [markdown] Cell 42
# ## 20. Resumen narrativo de resultados para iniciar el manuscrito

# %% Cell 43
# Cálculos centrales.
p1_ranked = (
    D08[
        D08["experiment"] == "P1"
    ]
    .sort_values(
        "macro_f1_mean",
        ascending=False
    )
)

best_p1 = p1_ranked.iloc[0]
second_p1 = p1_ranked.iloc[1]

p5_best_cross = (
    P5MATCH
    .sort_values(
        "cross_bidirectional_macro_f1",
        ascending=False
    )
    .iloc[0]
)

p5_worst_cross = (
    P5MATCH
    .sort_values(
        "cross_bidirectional_macro_f1",
        ascending=True
    )
    .iloc[0]
)

eff_fastest = (
    TABLE7
    .sort_values(
        "Throughput AMP b32 img/s",
        ascending=False
    )
    .iloc[0]
)

results_text = f'''
RESULTS SYNTHESIS — PADDY DOCTOR Q1

1. IID reference
Across three training seeds, {best_p1["model"]} achieved the highest mean
IID Macro-F1 ({best_p1["macro_f1_mean"]:.4f}), followed by
{second_p1["model"]} ({second_p1["macro_f1_mean"]:.4f}). Pairwise bootstrap
results should be consulted before describing any small difference between the
top architectures as statistically resolved.

2. Cross-domain generalization
Fixed-target bootstrap analysis showed large positive Macro-F1 drops for every
architecture in P4 and P5, with all reported 95% confidence intervals excluding
zero. Exact McNemar analyses further supported the degradation across seeds.
The dominant scientific result is therefore not the IID ranking, but the
failure of high-performing classifiers to preserve performance when the
training domain changes.

3. Phenology-controlled sensitivity
P5 retained exact matching by disease label and plant age. Nevertheless, the
bidirectional domain-match advantage remained large:
ResNet-50={P5MATCH.loc[P5MATCH.model_key=="resnet50","domain_match_advantage"].iloc[0]:.4f},
ConvNeXt-Tiny={P5MATCH.loc[P5MATCH.model_key=="convnext_tiny","domain_match_advantage"].iloc[0]:.4f},
ViT-S/16={P5MATCH.loc[P5MATCH.model_key=="vit_small_patch16_224","domain_match_advantage"].iloc[0]:.4f},
and Swin-Tiny={P5MATCH.loc[P5MATCH.model_key=="swin_tiny_patch4_window7_224","domain_match_advantage"].iloc[0]:.4f}.
This demonstrates persistence of the cross-domain problem after controlling
observable plant age. It should not be interpreted as pure causal
identification of cultivar.

4. Calibration
Raw ECE increased from {p4_ece_control:.4f} in P4 matched-domain controls to
{p4_ece_cross:.4f} under P4 cross-domain evaluation. In P5, raw ECE increased
from {p5_ece_control:.4f} to {p5_ece_cross:.4f}. Source-validation temperature
scaling improved calibration in many conditions, but cross-domain ECE and NLL
remained substantially worse than matched-domain controls.

5. Explainability
Attribution stability was architecture-dependent. The explanation analysis
supports the conclusion that predictive degradation and attribution stability
do not necessarily move together. These maps are not lesion-localization
ground truth, and no clinical or pathological localization claim is justified.
The reproduction audit recorded {xai_mismatches} prediction mismatches across
the explanation runs, which should be documented and excluded from selected
illustrative examples if necessary.

6. Efficiency
On the NVIDIA L4, {eff_fastest["Model"]} achieved the highest batch-32
AMP/FP16 throughput ({eff_fastest["Throughput AMP b32 img/s"]:.1f} images/s).
Single-image latency should be reported from FP32 batch=1 because AMP introduced
overhead at batch=1 in this hardware/software configuration. Efficiency and
robustness form a trade-off rather than a single universal ranking.

Overall interpretation
The combined evidence supports a paper centered on the discrepancy between
excellent IID classification and poor cross-cultivar/domain generalization,
with exact-age matching demonstrating that observable phenology alone does not
explain the degradation. Calibration failure adds a safety/reliability
dimension, while explainability and computational efficiency provide secondary
evidence about architecture-specific behavior and deployment trade-offs.
'''

with open(
    TEXT_ROOT /
    "12_Results_Synthesis_for_Manuscript.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(results_text)

print(results_text)

# %% [markdown] Cell 44
# ## 21. Propuesta de estructura final de resultados

# %% Cell 45
ARTICLE_RESULTS_STRUCTURE = pd.DataFrame([
    {
        "Section": "3.1",
        "Title": "Clean IID benchmark",
        "Primary evidence": "Table 2 + Figure 1",
        "Main message": "High in-domain performance; small differences among top architectures require inferential caution.",
    },
    {
        "Section": "3.2",
        "Title": "Cross-cultivar/domain generalization",
        "Primary evidence": "Table 3 + Figure 2",
        "Main message": "Large, reproducible performance degradation when training domain changes.",
    },
    {
        "Section": "3.3",
        "Title": "Phenology-controlled sensitivity analysis",
        "Primary evidence": "Table 4 + Table 8",
        "Main message": "Cross-domain degradation persists after exact disease-age matching.",
    },
    {
        "Section": "3.4",
        "Title": "Statistical inference and architecture dependence",
        "Primary evidence": "Supplementary S01–S04 + GEE",
        "Main message": "Domain shift is dominant; architecture×shift interaction differs between P4 and P5.",
    },
    {
        "Section": "3.5",
        "Title": "Probabilistic calibration under shift",
        "Primary evidence": "Table 5 + Figure 3",
        "Main message": "Domain shift causes severe miscalibration and overconfidence; validation-only temperature scaling helps but does not restore reliability.",
    },
    {
        "Section": "3.6",
        "Title": "Explainability stability and faithfulness",
        "Primary evidence": "Table 6 + Figure 4 + selected N10 qualitative figures",
        "Main message": "Attribution stability is architecture-dependent and not equivalent to predictive robustness.",
    },
    {
        "Section": "3.7",
        "Title": "Computational efficiency",
        "Primary evidence": "Table 7 + Figures 5–6",
        "Main message": "Deployment trade-offs differ from predictive rankings; ViT-S/16 is particularly efficient on L4.",
    },
])

ARTICLE_RESULTS_STRUCTURE.to_csv(
    TEXT_ROOT /
    "12_Recommended_Results_Section_Structure.csv",
    index=False
)

display(ARTICLE_RESULTS_STRUCTURE)

# %% [markdown] Cell 46
# ## 22. Inventario de tablas y figuras

# %% Cell 47
main_table_files = sorted(
    p.name
    for p in MAIN_TABLES.glob("*.csv")
)

supp_table_files = sorted(
    p.name
    for p in SUPP_TABLES.glob("*.csv")
)

figure_files = sorted(
    p.name
    for p in FIGURE_ROOT.glob("*")
    if p.suffix.lower() in {".png", ".pdf"}
)

inventory = {
    "main_tables": main_table_files,
    "supplementary_tables": supp_table_files,
    "figures": figure_files,
}

with open(
    RUN_ROOT /
    "12_Output_Inventory.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        inventory,
        f,
        indent=2
    )

print("Main tables:", len(main_table_files))
print("Supplementary tables:", len(supp_table_files))
print("Figure files:", len(figure_files))

# %% [markdown] Cell 48
# ## 23. Verificación final

# %% Cell 49
required_main = [
    "Table1_Experimental_Design.csv",
    "Table2_Predictive_Performance.csv",
    "Table3_Fixed_Target_Domain_Shift.csv",
    "Table4_P5_ExactAge_DomainMatch.csv",
    "Table5_Calibration_Summary.csv",
    "Table6_Explainability_Stability.csv",
    "Table7_Computational_Efficiency.csv",
    "Table8_P4_P5_Sensitivity.csv",
]

missing_main = [
    name
    for name in required_main
    if not (
        MAIN_TABLES / name
    ).exists()
]

if missing_main:
    raise RuntimeError(
        f"❌ Faltan tablas principales: {missing_main}"
    )

if len(list(SUPP_TABLES.glob("*.csv"))) < 14:
    raise RuntimeError(
        "❌ Se esperaban al menos 14 tablas suplementarias."
    )

if len(list(FIGURE_ROOT.glob("*.png"))) < 12:
    raise RuntimeError(
        "❌ Se esperaban al menos 12 figuras PNG."
    )

if not (
    TEXT_ROOT /
    "12_Results_Synthesis_for_Manuscript.txt"
).exists():
    raise RuntimeError(
        "❌ Falta resumen narrativo."
    )

print("=" * 80)
print("✅ NOTEBOOK 12 COMPLETADO")
print("=" * 80)
print("Tablas principales:", len(list(MAIN_TABLES.glob('*.csv'))))
print("Tablas suplementarias:", len(list(SUPP_TABLES.glob('*.csv'))))
print("PNG:", len(list(FIGURE_ROOT.glob('*.png'))))
print("PDF:", len(list(FIGURE_ROOT.glob('*.pdf'))))
print("No se realizaron nuevos experimentos.")

# %% [markdown] Cell 50
# ## 24. Exportar paquete editorial final

# %% Cell 51
from google.colab import files

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

package_name = (
    "12_PaddyDoctor_Final_Article_Package_"
    f"{timestamp}"
)

temp = (
    Path("/content") /
    package_name
)

if temp.exists():
    shutil.rmtree(temp)

(temp / "main_tables").mkdir(
    parents=True,
    exist_ok=True
)

(temp / "supplementary_tables").mkdir(
    parents=True,
    exist_ok=True
)

(temp / "figures").mkdir(
    parents=True,
    exist_ok=True
)

(temp / "manuscript_support").mkdir(
    parents=True,
    exist_ok=True
)

for path in MAIN_TABLES.glob("*"):
    if path.is_file():
        shutil.copy2(
            path,
            temp /
            "main_tables" /
            path.name
        )

for path in SUPP_TABLES.glob("*"):
    if path.is_file():
        shutil.copy2(
            path,
            temp /
            "supplementary_tables" /
            path.name
        )

for path in FIGURE_ROOT.glob("*"):
    if path.is_file():
        shutil.copy2(
            path,
            temp /
            "figures" /
            path.name
        )

for path in TEXT_ROOT.glob("*"):
    if path.is_file():
        shutil.copy2(
            path,
            temp /
            "manuscript_support" /
            path.name
        )

for name in [
    "12_config.json",
    "12_Output_Inventory.json",
]:
    source = RUN_ROOT / name
    if source.exists():
        shutil.copy2(
            source,
            temp / source.name
        )

with open(
    temp /
    "README_FINAL_PACKAGE.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "PADDY DOCTOR Q1 — FINAL ARTICLE PACKAGE\n\n"
        "Generated by Notebook 12.\n"
        "No new model training or test-based tuning was performed.\n"
        "Contains:\n"
        "- 8 main manuscript tables\n"
        "- supplementary statistical/calibration/XAI/efficiency tables\n"
        "- final figures in PNG 300 dpi and PDF\n"
        "- results synthesis and scientific claim guardrails\n"
        "- recommended Results-section structure\n\n"
        "The package is intended as the evidence base for manuscript drafting.\n"
    )

zip_path = shutil.make_archive(
    str(
        EXPORTS_DIR /
        package_name
    ),
    "zip",
    root_dir=temp
)

print("=" * 80)
print("✅ PAQUETE FINAL CREADO")
print("=" * 80)
print(zip_path)
print(
    f"Tamaño: "
    f"{Path(zip_path).stat().st_size/(1024**2):.2f} MB"
)

files.download(
    zip_path
)

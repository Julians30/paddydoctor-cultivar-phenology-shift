"""Auto-exported from 04_PaddyDoctor_PhenologyMatched_Protocols.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 04
# ## Phenology overlap audit + exact age-matched sensitivity protocols
# 
# ### Por qué existe este notebook
# El Notebook 03 construyó un benchmark `P4` igualado por clase entre `ADT45` y `OTHER`.
# Sin embargo, **igualar por clase no garantiza igualar por edad de la planta**.
# 
# Este notebook:
# 
# 1. Monta Google Drive.
# 2. Restaura automáticamente la autenticación persistente de Kaggle.
# 3. Recupera Paddy Doctor si el runtime es nuevo.
# 4. Audita cuantitativamente el **phenology shift** de P4.
# 5. Calcula soporte de edades compartidas por clase y dominio.
# 6. Construye un benchmark de sensibilidad `P5` con:
#    - 5 clases con soporte suficiente;
#    - 100 pares por clase;
#    - mismo `label`;
#    - misma `age`;
#    - un miembro ADT45 y un miembro OTHER;
#    - división 60/20/20 por pares.
# 7. Genera cuatro protocolos emparejados:
#    - `P5A_ADT45_to_OTHER_age_matched`
#    - `P5B_ADT45_to_ADT45_age_control`
#    - `P5C_OTHER_to_ADT45_age_matched`
#    - `P5D_OTHER_to_OTHER_age_control`
# 8. Congela un registro final de protocolos.
# 9. Exporta resultados a ZIP.
# 
# > **No requiere GPU.**

# %% [markdown] Cell 2
# ## 0. Conexión obligatoria a Google Drive

# %% Cell 3
from google.colab import drive
from pathlib import Path

drive.mount("/content/drive")

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

DATA_DIR = PROJECT / "data"
RAW_DIR = DATA_DIR / "raw" / "paddy_doctor"
METADATA_DIR = DATA_DIR / "metadata"
SPLITS_DIR = DATA_DIR / "splits"
PROCESSED_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT / "notebooks"
MODELS_DIR = PROJECT / "models"
RESULTS_DIR = PROJECT / "results"
FIGURES_DIR = PROJECT / "figures"
MANUSCRIPT_DIR = PROJECT / "manuscript"
EXPORTS_DIR = PROJECT / "exports"
CONFIG_DIR = PROJECT / "config"

for folder in [
    PROJECT, DATA_DIR, RAW_DIR, METADATA_DIR, SPLITS_DIR,
    PROCESSED_DIR, NOTEBOOKS_DIR, MODELS_DIR, RESULTS_DIR,
    FIGURES_DIR, MANUSCRIPT_DIR, EXPORTS_DIR, CONFIG_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("✅ GOOGLE DRIVE CONECTADO")
print("=" * 80)
print("Proyecto:", PROJECT)

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

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "kagglehub"
    ],
    check=True
)

import kagglehub

print("✅ Kaggle autenticado para esta sesión.")

# %% [markdown] Cell 6
# ## 2. Recuperar dataset y productos del Notebook 03

# %% Cell 7
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

DATASET_PATH = Path(
    "/root/.cache/kagglehub/competitions/paddy-disease-classification"
)

TRAIN_CSV = DATASET_PATH / "train.csv"
TRAIN_IMAGES = DATASET_PATH / "train_images"

if not TRAIN_CSV.exists():
    print("⬇️ Runtime nuevo: descargando Paddy Doctor...")
    DATASET_PATH = Path(
        kagglehub.competition_download(
            "paddy-disease-classification"
        )
    )
    TRAIN_CSV = DATASET_PATH / "train.csv"
    TRAIN_IMAGES = DATASET_PATH / "train_images"

NEARCLEAN_CSV = (
    METADATA_DIR /
    "03_paddy_doctor_nearclean.csv"
)

P4_ASSIGNMENTS_CSV = (
    SPLITS_DIR /
    "P4_matched_domain_assignments.csv"
)

if not NEARCLEAN_CSV.exists():
    raise FileNotFoundError(
        "❌ Falta 03_paddy_doctor_nearclean.csv. "
        "Ejecuta primero el Notebook 03."
    )

if not P4_ASSIGNMENTS_CSV.exists():
    raise FileNotFoundError(
        "❌ Falta P4_matched_domain_assignments.csv. "
        "Ejecuta primero el Notebook 03."
    )

nearclean_df = pd.read_csv(
    NEARCLEAN_CSV
)

p4_assignments = pd.read_csv(
    P4_ASSIGNMENTS_CSV
)

nearclean_df["domain"] = np.where(
    nearclean_df["variety"] == "ADT45",
    "ADT45",
    "OTHER"
)

print(f"Near-clean total: {len(nearclean_df):,}")
print(f"P4 matched pool: {len(p4_assignments):,}")
print(f"Clases: {nearclean_df['label'].nunique()}")

# %% [markdown] Cell 8
# ## 3. Auditoría del desplazamiento fenológico en P4
# 
# Calculamos por clase:
# 
# - media y desviación de edad;
# - diferencia absoluta de medias;
# - distancia de Wasserstein en días;
# - conjunto de edades observadas por dominio.
# 
# La distancia de Wasserstein tiene una interpretación directa: cuántos días, en promedio, habría que desplazar una distribución para alinearla con la otra.

# %% Cell 9
from scipy.stats import wasserstein_distance

age_rows = []

for label in sorted(
    p4_assignments["label"].unique()
):
    a = p4_assignments[
        (p4_assignments["label"] == label) &
        (p4_assignments["domain"] == "ADT45")
    ]["age"].to_numpy()

    b = p4_assignments[
        (p4_assignments["label"] == label) &
        (p4_assignments["domain"] == "OTHER")
    ]["age"].to_numpy()

    ages_a = sorted(
        np.unique(a).astype(int).tolist()
    )

    ages_b = sorted(
        np.unique(b).astype(int).tolist()
    )

    shared_ages = sorted(
        set(ages_a) & set(ages_b)
    )

    age_rows.append({
        "label": label,
        "n_ADT45": len(a),
        "n_OTHER": len(b),
        "mean_age_ADT45": round(float(np.mean(a)), 3),
        "mean_age_OTHER": round(float(np.mean(b)), 3),
        "abs_mean_age_difference_days": round(
            abs(float(np.mean(a) - np.mean(b))),
            3
        ),
        "wasserstein_age_days": round(
            float(wasserstein_distance(a, b)),
            3
        ),
        "n_unique_ages_ADT45": len(ages_a),
        "n_unique_ages_OTHER": len(ages_b),
        "n_shared_exact_ages": len(shared_ages),
        "shared_exact_ages": " | ".join(
            map(str, shared_ages)
        )
    })

p4_age_shift = pd.DataFrame(
    age_rows
).sort_values(
    "wasserstein_age_days",
    ascending=False
)

display(p4_age_shift)

p4_age_shift.to_csv(
    RESULTS_DIR /
    "04_P4_phenology_shift_by_class.csv",
    index=False
)

print(
    "\nMayor diferencia media de edad:",
    p4_age_shift[
        "abs_mean_age_difference_days"
    ].max(),
    "días"
)

print(
    "Mayor Wasserstein:",
    p4_age_shift[
        "wasserstein_age_days"
    ].max(),
    "días"
)

# %% [markdown] Cell 10
# ## 4. Soporte exacto label × age entre ADT45 y OTHER
# 
# Para cada combinación de enfermedad y edad calculamos:
# 
# `matched_possible = min(n_ADT45, n_OTHER)`
# 
# Eso indica cuántos **pares exactos** pueden formarse sin extrapolar edad.

# %% Cell 11
age_support = (
    nearclean_df
    .groupby(
        ["label", "age", "domain"]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

for required_domain in [
    "ADT45",
    "OTHER"
]:
    if required_domain not in age_support.columns:
        age_support[
            required_domain
        ] = 0

age_support = age_support[
    ["ADT45", "OTHER"]
].copy()

age_support[
    "matched_possible"
] = age_support[
    ["ADT45", "OTHER"]
].min(axis=1)

age_support = (
    age_support
    .reset_index()
    .sort_values(
        ["label", "age"]
    )
)

display(
    age_support.head(100)
)

age_support.to_csv(
    RESULTS_DIR /
    "04_exact_age_support_by_class.csv",
    index=False
)

matched_support_class = (
    age_support
    .groupby("label")
    .agg(
        n_exact_age_matched_per_domain=(
            "matched_possible",
            "sum"
        ),
        n_shared_age_cells=(
            "matched_possible",
            lambda x: int(
                (x > 0).sum()
            )
        )
    )
    .reset_index()
    .sort_values(
        "n_exact_age_matched_per_domain",
        ascending=False
    )
)

display(
    matched_support_class
)

matched_support_class.to_csv(
    RESULTS_DIR /
    "04_exact_age_overlap_summary_by_class.csv",
    index=False
)

# %% [markdown] Cell 12
# ## 5. Selección automática de clases para P5
# 
# Criterio predefinido:
# 
# - al menos **100 pares exactos label × age** disponibles por dominio.
# 
# Para mantener la comparación balanceada, usaremos exactamente:
# 
# - **100 pares por clase**;
# - 5 clases esperadas con los datos auditados;
# - 500 imágenes ADT45 + 500 imágenes OTHER.
# 
# Si el dataset cambia y no hay al menos 4 clases elegibles, el notebook se detiene.

# %% Cell 13
MIN_PAIRS_PER_CLASS = 100
TARGET_PAIRS_PER_CLASS = 100

eligible_classes = (
    matched_support_class[
        matched_support_class[
            "n_exact_age_matched_per_domain"
        ] >= MIN_PAIRS_PER_CLASS
    ]["label"]
    .sort_values()
    .tolist()
)

print(
    "Clases elegibles:",
    len(eligible_classes)
)

for c in eligible_classes:
    print("•", c)

if len(eligible_classes) < 4:
    raise RuntimeError(
        "❌ Soporte insuficiente para construir P5."
    )

eligible_support = (
    matched_support_class[
        matched_support_class[
            "label"
        ].isin(
            eligible_classes
        )
    ]
    .copy()
)

display(
    eligible_support
)

# %% [markdown] Cell 14
# ## 6. Construcción de pares exactos
# 
# Dentro de cada `label × age`:
# 
# 1. barajamos ADT45 y OTHER de forma reproducible;
# 2. emparejamos uno a uno hasta `min(n_ADT45,n_OTHER)`;
# 3. después seleccionamos exactamente 100 pares por clase.
# 
# El resultado garantiza que cada par tiene:
# 
# - misma enfermedad;
# - misma edad;
# - un miembro ADT45;
# - un miembro OTHER.

# %% Cell 15
RANDOM_SEED = 2026

pair_rows = []

for label in eligible_classes:
    label_df = nearclean_df[
        nearclean_df[
            "label"
        ] == label
    ].copy()

    for age in sorted(
        label_df["age"].unique()
    ):
        a = label_df[
            (label_df["age"] == age) &
            (label_df["domain"] == "ADT45")
        ].sample(
            frac=1,
            random_state=(
                RANDOM_SEED +
                int(age) +
                len(label)
            )
        )

        b = label_df[
            (label_df["age"] == age) &
            (label_df["domain"] == "OTHER")
        ].sample(
            frac=1,
            random_state=(
                RANDOM_SEED +
                int(age) +
                len(label) +
                1000
            )
        )

        n = min(
            len(a),
            len(b)
        )

        if n == 0:
            continue

        a = a.iloc[:n].reset_index(
            drop=True
        )

        b = b.iloc[:n].reset_index(
            drop=True
        )

        for i in range(n):
            pair_rows.append({
                "label": label,
                "age": int(age),
                "ADT45_image_id":
                    a.loc[i, "image_id"],
                "ADT45_variety":
                    a.loc[i, "variety"],
                "OTHER_image_id":
                    b.loc[i, "image_id"],
                "OTHER_variety":
                    b.loc[i, "variety"]
            })

all_exact_pairs = pd.DataFrame(
    pair_rows
)

print(
    "Pares exactos disponibles:",
    len(all_exact_pairs)
)

display(
    all_exact_pairs[
        "label"
    ].value_counts()
    .rename_axis("label")
    .reset_index(name="n_pairs")
)

selected_pair_parts = []

for label in eligible_classes:
    sub = all_exact_pairs[
        all_exact_pairs[
            "label"
        ] == label
    ].copy()

    if len(sub) < TARGET_PAIRS_PER_CLASS:
        raise RuntimeError(
            f"❌ {label} tiene solo "
            f"{len(sub)} pares."
        )

    selected = sub.sample(
        n=TARGET_PAIRS_PER_CLASS,
        replace=False,
        random_state=RANDOM_SEED
    ).copy()

    selected_pair_parts.append(
        selected
    )

selected_pairs = pd.concat(
    selected_pair_parts,
    ignore_index=True
)

selected_pairs = (
    selected_pairs
    .sort_values(
        ["label", "age",
         "ADT45_image_id",
         "OTHER_image_id"]
    )
    .reset_index(drop=True)
)

selected_pairs[
    "matched_pair_id"
] = [
    f"PAIR_{i:05d}"
    for i in range(
        len(selected_pairs)
    )
]

print(
    "\nPares seleccionados:",
    len(selected_pairs)
)

display(
    selected_pairs[
        "label"
    ].value_counts()
    .rename_axis("label")
    .reset_index(name="n_pairs")
)

selected_pairs.to_csv(
    SPLITS_DIR /
    "P5_exact_age_matched_pairs.csv",
    index=False
)

# %% [markdown] Cell 16
# ## 7. Split 60/20/20 a nivel de pares
# 
# El `matched_pair_id` es la unidad de partición.
# 
# Un par completo recibe el mismo `pair_split`, de modo que ADT45 y OTHER conservan exactamente la misma distribución de:
# 
# - clases;
# - edades;
# - tamaños de train/validation/test.

# %% Cell 17
from sklearn.model_selection import train_test_split

pair_train, pair_temp = train_test_split(
    selected_pairs,
    test_size=0.40,
    stratify=selected_pairs[
        "label"
    ],
    random_state=2026
)

pair_val, pair_test = train_test_split(
    pair_temp,
    test_size=0.50,
    stratify=pair_temp[
        "label"
    ],
    random_state=2027
)

pair_assignments = (
    selected_pairs.copy()
)

pair_assignments[
    "pair_split"
] = "UNASSIGNED"

pair_assignments.loc[
    pair_assignments[
        "matched_pair_id"
    ].isin(
        pair_train[
            "matched_pair_id"
        ]
    ),
    "pair_split"
] = "train"

pair_assignments.loc[
    pair_assignments[
        "matched_pair_id"
    ].isin(
        pair_val[
            "matched_pair_id"
        ]
    ),
    "pair_split"
] = "validation"

pair_assignments.loc[
    pair_assignments[
        "matched_pair_id"
    ].isin(
        pair_test[
            "matched_pair_id"
        ]
    ),
    "pair_split"
] = "test"

display(
    pd.crosstab(
        pair_assignments[
            "label"
        ],
        pair_assignments[
            "pair_split"
        ]
    )
)

pair_assignments.to_csv(
    SPLITS_DIR /
    "P5_pair_assignments.csv",
    index=False
)

# %% [markdown] Cell 18
# ## 8. Convertir pares a formato largo

# %% Cell 19
long_rows = []

lookup = (
    nearclean_df
    .set_index("image_id")
)

for row in pair_assignments.itertuples(
    index=False
):
    for domain, image_id in [
        ("ADT45", row.ADT45_image_id),
        ("OTHER", row.OTHER_image_id)
    ]:
        meta = lookup.loc[
            image_id
        ]

        long_rows.append({
            "matched_pair_id":
                row.matched_pair_id,
            "pair_split":
                row.pair_split,
            "image_id":
                image_id,
            "label":
                row.label,
            "age":
                int(row.age),
            "domain":
                domain,
            "variety":
                meta["variety"],
            "sha256":
                meta.get(
                    "sha256",
                    np.nan
                ),
            "visual_group_id":
                meta.get(
                    "visual_group_id",
                    np.nan
                ),
            "visual_group_size":
                meta.get(
                    "visual_group_size",
                    np.nan
                )
        })

p5_long = pd.DataFrame(
    long_rows
)

print(
    "P5 imágenes totales:",
    len(p5_long)
)

display(
    pd.crosstab(
        [
            p5_long["domain"],
            p5_long["label"]
        ],
        p5_long["pair_split"]
    )
)

p5_long.to_csv(
    SPLITS_DIR /
    "P5_exact_age_matched_long.csv",
    index=False
)

# %% [markdown] Cell 20
# ## 9. Verificación matemática del emparejamiento
# 
# Para cada split y cada clase, comprobamos que la distribución exacta de edades sea **idéntica** entre ADT45 y OTHER.

# %% Cell 21
verification_rows = []

for split_name in [
    "train",
    "validation",
    "test"
]:
    split_df = p5_long[
        p5_long[
            "pair_split"
        ] == split_name
    ]

    for label in eligible_classes:
        sub = split_df[
            split_df[
                "label"
            ] == label
        ]

        a_counts = (
            sub[
                sub["domain"] ==
                "ADT45"
            ]["age"]
            .value_counts()
            .sort_index()
        )

        b_counts = (
            sub[
                sub["domain"] ==
                "OTHER"
            ]["age"]
            .value_counts()
            .sort_index()
        )

        all_ages = sorted(
            set(
                a_counts.index
            ) |
            set(
                b_counts.index
            )
        )

        identical = True
        max_abs_difference = 0

        for age in all_ages:
            diff = abs(
                int(
                    a_counts.get(
                        age, 0
                    )
                ) -
                int(
                    b_counts.get(
                        age, 0
                    )
                )
            )

            max_abs_difference = max(
                max_abs_difference,
                diff
            )

            if diff != 0:
                identical = False

        verification_rows.append({
            "split":
                split_name,
            "label":
                label,
            "age_distribution_identical":
                identical,
            "max_age_count_difference":
                max_abs_difference,
            "n_ADT45":
                int(
                    (
                        sub["domain"] ==
                        "ADT45"
                    ).sum()
                ),
            "n_OTHER":
                int(
                    (
                        sub["domain"] ==
                        "OTHER"
                    ).sum()
                )
        })

verification_df = pd.DataFrame(
    verification_rows
)

display(
    verification_df
)

verification_df.to_csv(
    RESULTS_DIR /
    "04_P5_age_matching_verification.csv",
    index=False
)

if not verification_df[
    "age_distribution_identical"
].all():
    raise RuntimeError(
        "❌ P5 no quedó exactamente "
        "emparejado por edad."
    )

print(
    "✅ Distribuciones de edad idénticas "
    "entre dominios en todos los splits."
)

# %% [markdown] Cell 22
# ## 10. Construir P5A–P5D

# %% Cell 23
def build_p5_protocol(
    data,
    train_domain,
    test_domain,
    protocol_name
):
    out = data.copy()

    out[
        "split"
    ] = "EXCLUDED"

    out.loc[
        (
            out["domain"] ==
            train_domain
        ) &
        (
            out["pair_split"] ==
            "train"
        ),
        "split"
    ] = "train"

    out.loc[
        (
            out["domain"] ==
            train_domain
        ) &
        (
            out["pair_split"] ==
            "validation"
        ),
        "split"
    ] = "validation"

    out.loc[
        (
            out["domain"] ==
            test_domain
        ) &
        (
            out["pair_split"] ==
            "test"
        ),
        "split"
    ] = "test"

    out[
        "protocol"
    ] = protocol_name

    return out

p5a = build_p5_protocol(
    p5_long,
    "ADT45",
    "OTHER",
    "P5A_ADT45_to_OTHER_age_matched"
)

p5b = build_p5_protocol(
    p5_long,
    "ADT45",
    "ADT45",
    "P5B_ADT45_to_ADT45_age_control"
)

p5c = build_p5_protocol(
    p5_long,
    "OTHER",
    "ADT45",
    "P5C_OTHER_to_ADT45_age_matched"
)

p5d = build_p5_protocol(
    p5_long,
    "OTHER",
    "OTHER",
    "P5D_OTHER_to_OTHER_age_control"
)

p5_protocols = {
    "P5A_ADT45_to_OTHER_age_matched":
        p5a,
    "P5B_ADT45_to_ADT45_age_control":
        p5b,
    "P5C_OTHER_to_ADT45_age_matched":
        p5c,
    "P5D_OTHER_to_OTHER_age_control":
        p5d
}

for name, data in p5_protocols.items():
    data.to_csv(
        SPLITS_DIR /
        f"{name}.csv",
        index=False
    )

    active = data[
        data["split"] !=
        "EXCLUDED"
    ]

    print("\n", name)
    print(
        active[
            "split"
        ].value_counts()
    )

    table = pd.crosstab(
        active["label"],
        active["split"]
    )

    display(
        table
    )

    table.to_csv(
        SPLITS_DIR /
        f"{name}_class_distribution.csv"
    )

# %% [markdown] Cell 24
# ## 11. Verificar que no haya fuga entre splits

# %% Cell 25
leak_rows = []

for name, data in p5_protocols.items():
    active = data[
        data["split"] !=
        "EXCLUDED"
    ].copy()

    split_sets = {
        split_name:
            set(group["image_id"])
        for split_name, group
        in active.groupby("split")
    }

    split_names = sorted(
        split_sets
    )

    for i in range(
        len(split_names)
    ):
        for j in range(
            i + 1,
            len(split_names)
        ):
            a = split_names[i]
            b = split_names[j]

            overlap = (
                split_sets[a] &
                split_sets[b]
            )

            leak_rows.append({
                "protocol":
                    name,
                "split_a":
                    a,
                "split_b":
                    b,
                "image_overlap":
                    len(overlap)
            })

p5_leakage = pd.DataFrame(
    leak_rows
)

display(
    p5_leakage
)

p5_leakage.to_csv(
    RESULTS_DIR /
    "04_P5_leakage_check.csv",
    index=False
)

if (
    p5_leakage[
        "image_overlap"
    ] > 0
).any():
    raise RuntimeError(
        "❌ Hay fuga de image_id "
        "entre splits."
    )

print(
    "✅ Sin image_id compartidos "
    "entre splits."
)

# %% [markdown] Cell 26
# ## 12. Resumen y registro final de protocolos
# 
# ### Interpretación prevista
# 
# - `P1_group_clean`: rendimiento IID limpio en las 10 clases.
# - `P4A/P4C`: cambio de dominio con 10 clases, igualado por clase, pero **no completamente separable de fenología**.
# - `P4B/P4D`: controles in-domain con el mismo presupuesto.
# - `P5A/P5C`: sensibilidad cross-domain con edad **exactamente emparejada**.
# - `P5B/P5D`: controles in-domain del mismo subconjunto emparejado.
# 
# P5 no reemplaza P4: actúa como análisis de sensibilidad causalmente más limpio sobre las clases con soporte suficiente.

# %% Cell 27
protocol_registry_rows = [
    {
        "protocol_family":
            "P1",
        "protocol":
            "P1_group_clean",
        "n_classes":
            10,
        "purpose":
            "Clean IID benchmark",
        "class_matched":
            False,
        "exact_age_matched":
            False
    },
    {
        "protocol_family":
            "P4",
        "protocol":
            "P4A/P4B/P4C/P4D",
        "n_classes":
            10,
        "purpose":
            "Class-matched cultivar/domain benchmark",
        "class_matched":
            True,
        "exact_age_matched":
            False
    },
    {
        "protocol_family":
            "P5",
        "protocol":
            "P5A/P5B/P5C/P5D",
        "n_classes":
            len(
                eligible_classes
            ),
        "purpose":
            "Exact-age-matched sensitivity benchmark",
        "class_matched":
            True,
        "exact_age_matched":
            True
    }
]

protocol_registry = pd.DataFrame(
    protocol_registry_rows
)

display(
    protocol_registry
)

protocol_registry.to_csv(
    SPLITS_DIR /
    "04_FINAL_PROTOCOL_REGISTRY.csv",
    index=False
)

summary = pd.DataFrame([
    {
        "metric":
            "P4_max_abs_mean_age_difference_days",
        "value":
            p4_age_shift[
                "abs_mean_age_difference_days"
            ].max()
    },
    {
        "metric":
            "P4_max_wasserstein_age_days",
        "value":
            p4_age_shift[
                "wasserstein_age_days"
            ].max()
    },
    {
        "metric":
            "P5_eligible_classes",
        "value":
            len(
                eligible_classes
            )
    },
    {
        "metric":
            "P5_pairs_per_class",
        "value":
            TARGET_PAIRS_PER_CLASS
    },
    {
        "metric":
            "P5_total_pairs",
        "value":
            len(
                selected_pairs
            )
    },
    {
        "metric":
            "P5_total_images_both_domains",
        "value":
            len(
                p5_long
            )
    },
    {
        "metric":
            "P5_age_matching_all_verified",
        "value":
            int(
                verification_df[
                    "age_distribution_identical"
                ].all()
            )
    }
])

display(
    summary
)

summary.to_csv(
    RESULTS_DIR /
    "04_final_protocol_summary.csv",
    index=False
)

print("=" * 80)
print("✅ NOTEBOOK 04 FINALIZADO")
print("=" * 80)
print(
    "Si todas las verificaciones son correctas, "
    "el siguiente paso será entrenamiento."
)

# %% [markdown] Cell 28
# ## 13. Exportar resultados a ZIP

# %% Cell 29
import shutil
import time
from google.colab import files

timestamp = time.strftime(
    "%Y%m%d_%H%M%S"
)

PACKAGE_NAME = (
    "04_PaddyDoctor_PhenologyMatched_"
    f"Protocols_RESULTS_{timestamp}"
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

include_patterns = {
    "data/splits": [
        "P5*.csv",
        "04_*.csv"
    ],
    "results": [
        "04_*.csv"
    ],
    "figures": [
        "04_*.png"
    ]
}

copied = []

for relative_dir, patterns in (
    include_patterns.items()
):
    source_dir = (
        PROJECT /
        relative_dir
    )

    if not source_dir.exists():
        continue

    dest_dir = (
        TEMP_PACKAGE /
        relative_dir
    )

    dest_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for pattern in patterns:
        for source in source_dir.glob(
            pattern
        ):
            if source.is_file():
                destination = (
                    dest_dir /
                    source.name
                )

                shutil.copy2(
                    source,
                    destination
                )

                copied.append(
                    str(
                        source.relative_to(
                            PROJECT
                        )
                    )
                )

with open(
    TEMP_PACKAGE /
    "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 04\n"
        "Phenology overlap audit + exact age-matched protocols\n"
        f"Timestamp: {timestamp}\n\n"
        "Files included:\n"
    )

    for item in copied:
        f.write(
            f"- {item}\n"
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

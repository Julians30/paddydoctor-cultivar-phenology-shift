"""Auto-exported from 03_PaddyDoctor_NearDuplicates_MatchedProtocols.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 03
# ## Near-duplicate audit + matched cross-cultivar benchmark
# 
# ### Objetivos
# 1. Conectar Google Drive y restaurar automáticamente la autenticación de Kaggle.
# 2. Recuperar Paddy Doctor si el runtime de Colab es nuevo.
# 3. Recalcular baselines de metadatos tratando `age` también como variable categórica.
# 4. Detectar **near-duplicates perceptuales** mediante pHash + SSIM.
# 5. Auditar near-duplicates dentro del train y entre train ↔ test oficial de Kaggle.
# 6. Crear una versión **near-clean** del conjunto etiquetado.
# 7. Construir un nuevo `P1_group_clean` sin grupos visualmente redundantes cruzando splits.
# 8. Crear un benchmark **matched ADT45 ↔ OTHER** que iguala por clase la cantidad disponible en ambos dominios.
# 9. Exportar automáticamente todos los resultados a ZIP.
# 
# > **No requiere GPU.**

# %% [markdown] Cell 2
# ## 0. Google Drive
# Cada notebook del proyecto es autónomo: siempre monta `Mi unidad` y crea/verifica la estructura del proyecto.

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
# La primera vez solicita `kaggle.json` y lo guarda de forma privada en `PADDY_DOCTOR_Q1/config/`.
# Después lo restaura automáticamente en cada runtime nuevo.

# %% Cell 5
from google.colab import files
import shutil, os, sys, subprocess

DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"
LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"

if not DRIVE_KAGGLE_JSON.exists():
    print("🔑 Selecciona tu archivo kaggle.json")
    uploaded = files.upload()

    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("Debes seleccionar exactamente kaggle.json")

    with open(DRIVE_KAGGLE_JSON, "wb") as f:
        f.write(uploaded["kaggle.json"])

    print("✅ Credencial guardada en Drive")
else:
    print("✅ Credencial persistente encontrada")

shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
os.chmod(LOCAL_KAGGLE_JSON, 0o600)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "kagglehub", "scikit-image"],
    check=True
)

import kagglehub

print("✅ Kaggle autenticado para esta sesión")

# %% [markdown] Cell 6
# ## 2. Recuperar dataset y resultados del Notebook 02

# %% Cell 7
import pandas as pd
import numpy as np
import warnings
import hashlib
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

DATASET_PATH = Path(
    "/root/.cache/kagglehub/competitions/paddy-disease-classification"
)

TRAIN_CSV = DATASET_PATH / "train.csv"
TRAIN_IMAGES = DATASET_PATH / "train_images"
TEST_IMAGES = DATASET_PATH / "test_images"

if not TRAIN_CSV.exists():
    print("⬇️ Cache de KaggleHub no disponible. Descargando Paddy Doctor...")
    DATASET_PATH = Path(
        kagglehub.competition_download("paddy-disease-classification")
    )
    TRAIN_CSV = DATASET_PATH / "train.csv"
    TRAIN_IMAGES = DATASET_PATH / "train_images"
    TEST_IMAGES = DATASET_PATH / "test_images"

DEDUP_CSV = METADATA_DIR / "02_paddy_doctor_deduplicated.csv"

if not DEDUP_CSV.exists():
    raise FileNotFoundError(
        "❌ Falta data/metadata/02_paddy_doctor_deduplicated.csv. "
        "Ejecuta primero el Notebook 02."
    )

df = pd.read_csv(TRAIN_CSV)
dedup_df = pd.read_csv(DEDUP_CSV)

print(f"Original: {len(df):,}")
print(f"Exact-deduplicado: {len(dedup_df):,}")
print(f"Clases: {dedup_df['label'].nunique()}")
print(f"Variedades: {dedup_df['variety'].nunique()}")

# %% [markdown] Cell 8
# ## 3. Baselines de metadatos refinados
# El Notebook 02 trató `age` como variable numérica lineal. Aquí también evaluamos la edad como **categoría**, porque la relación enfermedad–edad puede ser no lineal.

# %% Cell 9
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def metadata_cv(dataframe, features, categorical_features, numeric_features, name):
    X = dataframe[features].copy()
    y = dataframe["label"].copy()

    transformers = []

    if categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        )

    if numeric_features:
        transformers.append(
            ("num", StandardScaler(), numeric_features)
        )

    pipe = Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            random_state=2026
        ))
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=2026
    )

    rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])

        rows.append({
            "model": name,
            "fold": fold,
            "accuracy": accuracy_score(y.iloc[te], pred),
            "balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
            "macro_f1": f1_score(y.iloc[te], pred, average="macro")
        })

    return pd.DataFrame(rows)

# Convertimos age a string solo para los modelos categóricos
meta_df = dedup_df.copy()
meta_df["age_cat"] = meta_df["age"].astype(str)

metadata_models = pd.concat([
    metadata_cv(
        meta_df,
        ["variety"],
        ["variety"],
        [],
        "variety_only"
    ),
    metadata_cv(
        meta_df,
        ["age"],
        [],
        ["age"],
        "age_numeric"
    ),
    metadata_cv(
        meta_df,
        ["age_cat"],
        ["age_cat"],
        [],
        "age_categorical"
    ),
    metadata_cv(
        meta_df,
        ["variety", "age"],
        ["variety"],
        ["age"],
        "variety_plus_age_numeric"
    ),
    metadata_cv(
        meta_df,
        ["variety", "age_cat"],
        ["variety", "age_cat"],
        [],
        "variety_plus_age_categorical"
    ),
], ignore_index=True)

metadata_summary = (
    metadata_models
    .groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
    )
    .reset_index()
    .round(4)
)

display(metadata_summary)

metadata_models.to_csv(
    RESULTS_DIR / "03_metadata_only_refined_raw.csv",
    index=False
)

metadata_summary.to_csv(
    RESULTS_DIR / "03_metadata_only_refined_summary.csv",
    index=False
)

# %% [markdown] Cell 10
# ## 4. Preparar índice físico de imágenes

# %% Cell 11
from PIL import Image
from scipy.fftpack import dct
from skimage.metrics import structural_similarity as ssim

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

train_paths = {
    p.name: p
    for p in TRAIN_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
}

test_paths = {
    p.name: p
    for p in TEST_IMAGES.rglob("*")
    if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
}

missing_train = [
    x for x in dedup_df["image_id"]
    if x not in train_paths
]

print(f"Train exact-deduplicado: {len(dedup_df):,}")
print(f"Archivos train indexados: {len(train_paths):,}")
print(f"Archivos test indexados: {len(test_paths):,}")
print(f"Train faltantes: {len(missing_train)}")

if missing_train:
    raise RuntimeError("❌ Hay imágenes de train no localizadas.")

# %% [markdown] Cell 12
# ## 5. pHash perceptual de 64 bits
# 
# Usamos un hash perceptual basado en DCT.  
# La distancia de Hamming entre hashes pequeños identifica imágenes visualmente casi idénticas aunque cambie ligeramente la compresión JPEG.

# %% Cell 13
def phash64(path, hash_size=8, highfreq_factor=4):
    size = hash_size * highfreq_factor

    with Image.open(path) as img:
        img = img.convert("L").resize(
            (size, size),
            Image.Resampling.LANCZOS
        )
        pixels = np.asarray(img, dtype=np.float32)

    # DCT 2D
    dct_rows = dct(pixels, axis=0, norm="ortho")
    dct_vals = dct(dct_rows, axis=1, norm="ortho")

    low = dct_vals[:hash_size, :hash_size].copy()

    # Excluir componente DC para la mediana
    median = np.median(low.flatten()[1:])
    bits = (low > median).flatten()

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)

    return int(value)

train_phash = {}

for image_id in tqdm(
    dedup_df["image_id"],
    desc="pHash train"
):
    train_phash[image_id] = phash64(
        train_paths[image_id]
    )

test_phash = {}

for image_id, path in tqdm(
    list(test_paths.items()),
    desc="pHash test"
):
    test_phash[image_id] = phash64(path)

phash_train_df = pd.DataFrame({
    "image_id": list(train_phash.keys()),
    "phash64": [f"{v:016x}" for v in train_phash.values()]
})

phash_test_df = pd.DataFrame({
    "image_id": list(test_phash.keys()),
    "phash64": [f"{v:016x}" for v in test_phash.values()]
})

phash_train_df.to_csv(
    METADATA_DIR / "03_train_phash64.csv",
    index=False
)

phash_test_df.to_csv(
    METADATA_DIR / "03_test_phash64.csv",
    index=False
)

print("✅ pHash calculado.")

# %% [markdown] Cell 14
# ## 6. Generación eficiente de candidatos por bandas
# 
# Dividimos el hash de 64 bits en ocho bandas de 8 bits.  
# Dos hashes con Hamming ≤ 6 necesariamente comparten al menos dos bandas, lo que permite reducir drásticamente el número de comparaciones.

# %% Cell 15
from collections import defaultdict

def band_keys(value):
    return [
        (band, (value >> (band * 8)) & 0xFF)
        for band in range(8)
    ]

def candidate_pairs_within(hash_dict):
    buckets = defaultdict(list)

    for image_id, value in hash_dict.items():
        for key in band_keys(value):
            buckets[key].append(image_id)

    pairs = set()

    for members in buckets.values():
        if len(members) < 2:
            continue

        members = sorted(members)

        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))

    return pairs

def candidate_pairs_cross(hash_a, hash_b):
    buckets_a = defaultdict(list)

    for image_id, value in hash_a.items():
        for key in band_keys(value):
            buckets_a[key].append(image_id)

    pairs = set()

    for image_id_b, value_b in hash_b.items():
        for key in band_keys(value_b):
            for image_id_a in buckets_a.get(key, []):
                pairs.add((image_id_a, image_id_b))

    return pairs

train_candidates = candidate_pairs_within(train_phash)
cross_candidates = candidate_pairs_cross(train_phash, test_phash)

print(f"Candidatos train↔train: {len(train_candidates):,}")
print(f"Candidatos train↔test : {len(cross_candidates):,}")

# %% [markdown] Cell 16
# ## 7. Hamming + SSIM
# 
# Criterio conservador utilizado para **near-duplicate confirmado**:
# 
# - pHash Hamming ≤ 4
# - SSIM ≥ 0.95
# 
# También guardamos candidatos Hamming 5–6 para análisis de sensibilidad, pero no los usamos automáticamente para eliminar imágenes.

# %% Cell 17
def hamming64(a, b):
    return (int(a) ^ int(b)).bit_count()

def load_gray_128(path):
    with Image.open(path) as img:
        arr = np.asarray(
            img.convert("L").resize(
                (128, 128),
                Image.Resampling.LANCZOS
            ),
            dtype=np.float32
        )
    return arr

# Cache para evitar leer repetidamente las mismas imágenes
gray_cache = {}

def get_gray(key, path):
    if key not in gray_cache:
        gray_cache[key] = load_gray_128(path)
    return gray_cache[key]

def pair_ssim(key_a, path_a, key_b, path_b):
    a = get_gray(key_a, path_a)
    b = get_gray(key_b, path_b)

    return float(
        ssim(
            a,
            b,
            data_range=255.0
        )
    )

train_pair_rows = []

# Primero filtrar por Hamming
train_hamming_candidates = [
    (a, b, hamming64(train_phash[a], train_phash[b]))
    for a, b in train_candidates
    if hamming64(train_phash[a], train_phash[b]) <= 6
]

print(
    f"Train↔train con Hamming ≤ 6: "
    f"{len(train_hamming_candidates):,}"
)

for a, b, hd in tqdm(
    train_hamming_candidates,
    desc="SSIM train"
):
    score = pair_ssim(
        "tr_" + a, train_paths[a],
        "tr_" + b, train_paths[b]
    )

    train_pair_rows.append({
        "image_a": a,
        "image_b": b,
        "phash_hamming": hd,
        "ssim": round(score, 6),
        "confirmed_near_duplicate": (
            hd <= 4 and score >= 0.95
        )
    })

train_pairs_df = pd.DataFrame(train_pair_rows)

if len(train_pairs_df):
    train_pairs_df = train_pairs_df.sort_values(
        ["confirmed_near_duplicate", "ssim"],
        ascending=[False, False]
    )

train_pairs_df.to_csv(
    METADATA_DIR / "03_train_near_duplicate_pairs.csv",
    index=False
)

confirmed_train_pairs = (
    train_pairs_df[
        train_pairs_df["confirmed_near_duplicate"] == True
    ].copy()
    if len(train_pairs_df)
    else pd.DataFrame()
)

print(
    "Near-duplicate pairs confirmados dentro de train:",
    len(confirmed_train_pairs)
)

# %% Cell 18
# Train ↔ test oficial
cross_hamming_candidates = [
    (a, b, hamming64(train_phash[a], test_phash[b]))
    for a, b in cross_candidates
    if hamming64(train_phash[a], test_phash[b]) <= 6
]

print(
    f"Train↔test con Hamming ≤ 6: "
    f"{len(cross_hamming_candidates):,}"
)

cross_rows = []

for a, b, hd in tqdm(
    cross_hamming_candidates,
    desc="SSIM train-test"
):
    score = pair_ssim(
        "tr_" + a, train_paths[a],
        "te_" + b, test_paths[b]
    )

    cross_rows.append({
        "train_image": a,
        "test_image": b,
        "phash_hamming": hd,
        "ssim": round(score, 6),
        "confirmed_near_duplicate": (
            hd <= 4 and score >= 0.95
        )
    })

cross_pairs_df = pd.DataFrame(cross_rows)

if len(cross_pairs_df):
    cross_pairs_df = cross_pairs_df.sort_values(
        ["confirmed_near_duplicate", "ssim"],
        ascending=[False, False]
    )

cross_pairs_df.to_csv(
    METADATA_DIR / "03_train_test_near_duplicate_pairs.csv",
    index=False
)

confirmed_cross_pairs = (
    cross_pairs_df[
        cross_pairs_df["confirmed_near_duplicate"] == True
    ].copy()
    if len(cross_pairs_df)
    else pd.DataFrame()
)

print(
    "Near-duplicate pairs confirmados train↔test:",
    len(confirmed_cross_pairs)
)

if len(confirmed_cross_pairs):
    print(
        "Test images únicas involucradas:",
        confirmed_cross_pairs["test_image"].nunique()
    )

# %% [markdown] Cell 19
# ## 8. Crear grupos de near-duplicates dentro del train
# Los pares confirmados se convierten en componentes conexos.  
# Cada componente recibe un `visual_group_id`.

# %% Cell 20
# Union-Find
parent = {}

def find(x):
    if x not in parent:
        parent[x] = x

    if parent[x] != x:
        parent[x] = find(parent[x])

    return parent[x]

def union(a, b):
    ra = find(a)
    rb = find(b)

    if ra != rb:
        parent[rb] = ra

# Inicializar todas las imágenes como grupo individual
for image_id in dedup_df["image_id"]:
    parent[image_id] = image_id

if len(confirmed_train_pairs):
    for row in confirmed_train_pairs.itertuples(index=False):
        union(row.image_a, row.image_b)

groups = defaultdict(list)

for image_id in dedup_df["image_id"]:
    groups[find(image_id)].append(image_id)

group_rows = []

for idx, members in enumerate(
    sorted(groups.values(), key=lambda x: x[0])
):
    group_id = f"VG{idx:05d}"

    for image_id in sorted(members):
        group_rows.append({
            "image_id": image_id,
            "visual_group_id": group_id,
            "visual_group_size": len(members)
        })

group_df = pd.DataFrame(group_rows)

grouped_df = dedup_df.merge(
    group_df,
    on="image_id",
    how="left"
)

group_summary = (
    grouped_df
    .groupby("visual_group_id")
    .agg(
        n_images=("image_id", "size"),
        n_labels=("label", "nunique"),
        n_varieties=("variety", "nunique"),
        n_ages=("age", "nunique")
    )
    .reset_index()
)

problem_groups = group_summary[
    (group_summary["n_labels"] > 1) |
    (group_summary["n_varieties"] > 1) |
    (group_summary["n_ages"] > 1)
].copy()

print(f"Visual groups totales: {len(group_summary):,}")
print(
    "Grupos con >1 imagen:",
    int((group_summary["n_images"] > 1).sum())
)
print(
    "Grupos con metadata/clase inconsistente:",
    len(problem_groups)
)

grouped_df.to_csv(
    METADATA_DIR / "03_dedup_with_visual_groups.csv",
    index=False
)

group_summary.to_csv(
    METADATA_DIR / "03_visual_group_summary.csv",
    index=False
)

problem_groups.to_csv(
    METADATA_DIR / "03_visual_group_inconsistencies.csv",
    index=False
)

# %% [markdown] Cell 21
# ## 9. Crear dataset near-clean
# Para el benchmark principal conservamos un representante por `visual_group_id`.  
# Esto elimina redundancia perceptual de forma conservadora.

# %% Cell 22
nearclean_df = (
    grouped_df
    .sort_values("image_id")
    .drop_duplicates(
        subset="visual_group_id",
        keep="first"
    )
    .reset_index(drop=True)
)

removed_near = len(grouped_df) - len(nearclean_df)

print(f"Exact-deduplicado: {len(grouped_df):,}")
print(f"Near-clean:         {len(nearclean_df):,}")
print(f"Adicionales removidas por near-duplicate: {removed_near:,}")

nearclean_df.to_csv(
    METADATA_DIR / "03_paddy_doctor_nearclean.csv",
    index=False
)

# %% [markdown] Cell 23
# ## 10. P1_group_clean
# Se crea un split ≈71/14/14 mediante `StratifiedGroupKFold`, preservando clases y evitando que un grupo visual cruce train/validation/test.

# %% Cell 24
from sklearn.model_selection import StratifiedGroupKFold

# Como nearclean_df ya conserva solo un representante por grupo,
# los grupos son únicos; mantenemos la columna para trazabilidad.
sgkf_test = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=2026
)

X_dummy = np.zeros(len(nearclean_df))
y = nearclean_df["label"].values
groups_arr = nearclean_df["visual_group_id"].values

trainval_idx, test_idx = next(
    sgkf_test.split(
        X_dummy,
        y,
        groups_arr
    )
)

trainval_df = nearclean_df.iloc[
    trainval_idx
].copy()

test_df = nearclean_df.iloc[
    test_idx
].copy()

sgkf_val = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=2027
)

X_tv = np.zeros(len(trainval_df))
y_tv = trainval_df["label"].values
g_tv = trainval_df["visual_group_id"].values

train_rel, val_rel = next(
    sgkf_val.split(
        X_tv,
        y_tv,
        g_tv
    )
)

train_df = trainval_df.iloc[
    train_rel
].copy()

val_df = trainval_df.iloc[
    val_rel
].copy()

p1_group = nearclean_df.copy()
p1_group["split"] = "UNASSIGNED"

p1_group.loc[
    p1_group["image_id"].isin(train_df["image_id"]),
    "split"
] = "train"

p1_group.loc[
    p1_group["image_id"].isin(val_df["image_id"]),
    "split"
] = "validation"

p1_group.loc[
    p1_group["image_id"].isin(test_df["image_id"]),
    "split"
] = "test"

p1_group["protocol"] = "P1_group_clean"

print(p1_group["split"].value_counts())

display(
    pd.crosstab(
        p1_group["label"],
        p1_group["split"]
    )
)

p1_group.to_csv(
    SPLITS_DIR / "P1_group_clean.csv",
    index=False
)

pd.crosstab(
    p1_group["label"],
    p1_group["split"]
).to_csv(
    SPLITS_DIR / "P1_group_clean_class_distribution.csv"
)

# %% [markdown] Cell 25
# ## 11. Benchmark matched ADT45 ↔ OTHER
# 
# Este protocolo controla explícitamente:
# - número de muestras por clase;
# - distribución de clases entre dominios;
# - tamaño de entrenamiento;
# - tamaño de prueba.
# 
# Para cada clase se toma el mismo número de imágenes de ADT45 y OTHER:  
# `min(n_ADT45, n_OTHER)`.
# 
# Cada dominio se divide 60% / 20% / 20%.
# 
# Esto permite cuatro comparaciones directas:
# 
# - **P4A:** entrenar ADT45 → probar OTHER.
# - **P4B:** entrenar ADT45 → probar ADT45 (control in-domain).
# - **P4C:** entrenar OTHER → probar ADT45.
# - **P4D:** entrenar OTHER → probar OTHER (control in-domain).

# %% Cell 26
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(2026)

nearclean_df["domain"] = np.where(
    nearclean_df["variety"] == "ADT45",
    "ADT45",
    "OTHER"
)

support = pd.crosstab(
    nearclean_df["label"],
    nearclean_df["domain"]
)

support["matched_n_per_domain"] = support[
    ["ADT45", "OTHER"]
].min(axis=1)

display(support)

support.to_csv(
    SPLITS_DIR / "P4_matched_support.csv"
)

matched_parts = []

for label, row in support.iterrows():
    n_match = int(row["matched_n_per_domain"])

    for domain in ["ADT45", "OTHER"]:
        sub = nearclean_df[
            (nearclean_df["label"] == label) &
            (nearclean_df["domain"] == domain)
        ].copy()

        # Muestreo determinista
        chosen = sub.sample(
            n=n_match,
            replace=False,
            random_state=2026
        )

        matched_parts.append(chosen)

matched_pool = pd.concat(
    matched_parts,
    ignore_index=True
)

print(
    "Matched pool total:",
    len(matched_pool)
)

display(
    pd.crosstab(
        matched_pool["label"],
        matched_pool["domain"]
    )
)

# %% Cell 27
# Crear una partición 60/20/20 separada pero simétrica para cada dominio

domain_split_parts = []

for domain in ["ADT45", "OTHER"]:
    domain_df = matched_pool[
        matched_pool["domain"] == domain
    ].copy()

    # 60% train, 40% temp
    train_part, temp_part = train_test_split(
        domain_df,
        test_size=0.40,
        stratify=domain_df["label"],
        random_state=2026
    )

    # 20% validation, 20% test
    val_part, test_part = train_test_split(
        temp_part,
        test_size=0.50,
        stratify=temp_part["label"],
        random_state=2027
    )

    train_part["domain_split"] = "train"
    val_part["domain_split"] = "validation"
    test_part["domain_split"] = "test"

    domain_split_parts.extend([
        train_part,
        val_part,
        test_part
    ])

matched_assignments = pd.concat(
    domain_split_parts,
    ignore_index=True
)

display(
    pd.crosstab(
        [
            matched_assignments["domain"],
            matched_assignments["label"]
        ],
        matched_assignments["domain_split"]
    )
)

matched_assignments.to_csv(
    SPLITS_DIR / "P4_matched_domain_assignments.csv",
    index=False
)

# %% Cell 28
def build_p4_protocol(assignments, train_domain, test_domain, name):
    out = assignments.copy()
    out["split"] = "EXCLUDED"

    # Train + validation provienen del dominio de entrenamiento
    out.loc[
        (out["domain"] == train_domain) &
        (out["domain_split"] == "train"),
        "split"
    ] = "train"

    out.loc[
        (out["domain"] == train_domain) &
        (out["domain_split"] == "validation"),
        "split"
    ] = "validation"

    # Test proviene del dominio de prueba
    out.loc[
        (out["domain"] == test_domain) &
        (out["domain_split"] == "test"),
        "split"
    ] = "test"

    out["protocol"] = name

    return out

p4a = build_p4_protocol(
    matched_assignments,
    "ADT45",
    "OTHER",
    "P4A_ADT45_to_OTHER_matched"
)

p4b = build_p4_protocol(
    matched_assignments,
    "ADT45",
    "ADT45",
    "P4B_ADT45_to_ADT45_control"
)

p4c = build_p4_protocol(
    matched_assignments,
    "OTHER",
    "ADT45",
    "P4C_OTHER_to_ADT45_matched"
)

p4d = build_p4_protocol(
    matched_assignments,
    "OTHER",
    "OTHER",
    "P4D_OTHER_to_OTHER_control"
)

p4_protocols = {
    "P4A_ADT45_to_OTHER_matched": p4a,
    "P4B_ADT45_to_ADT45_control": p4b,
    "P4C_OTHER_to_ADT45_matched": p4c,
    "P4D_OTHER_to_OTHER_control": p4d,
}

for name, data in p4_protocols.items():
    data.to_csv(
        SPLITS_DIR / f"{name}.csv",
        index=False
    )

    active = data[
        data["split"] != "EXCLUDED"
    ]

    print("\n", name)
    print(active["split"].value_counts())

    table = pd.crosstab(
        active["label"],
        active["split"]
    )

    display(table)

    table.to_csv(
        SPLITS_DIR / f"{name}_class_distribution.csv"
    )

# %% [markdown] Cell 29
# ## 12. Auditoría final de integridad de protocolos

# %% Cell 30
protocol_rows = []

all_protocols = {
    "P1_group_clean": p1_group,
    **p4_protocols
}

for name, data in all_protocols.items():
    active = data[
        data["split"] != "EXCLUDED"
    ].copy()

    for split_name, sub in active.groupby("split"):
        protocol_rows.append({
            "protocol": name,
            "split": split_name,
            "n_images": len(sub),
            "n_classes": sub["label"].nunique(),
            "n_varieties": sub["variety"].nunique(),
            "n_visual_groups": sub["visual_group_id"].nunique(),
            "min_age": sub["age"].min(),
            "max_age": sub["age"].max(),
        })

protocol_summary = pd.DataFrame(
    protocol_rows
)

display(protocol_summary)

protocol_summary.to_csv(
    SPLITS_DIR / "03_protocol_summary.csv",
    index=False
)

# Verificación de image_id repetidos entre splits de cada protocolo
leakage_rows = []

for name, data in all_protocols.items():
    active = data[
        data["split"] != "EXCLUDED"
    ]

    split_sets = {
        s: set(g["image_id"])
        for s, g in active.groupby("split")
    }

    splits = sorted(split_sets)

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            a, b = splits[i], splits[j]
            overlap = split_sets[a] & split_sets[b]

            leakage_rows.append({
                "protocol": name,
                "split_a": a,
                "split_b": b,
                "image_overlap": len(overlap)
            })

leakage_df = pd.DataFrame(leakage_rows)

display(leakage_df)

leakage_df.to_csv(
    SPLITS_DIR / "03_protocol_leakage_check.csv",
    index=False
)

if (leakage_df["image_overlap"] > 0).any():
    raise RuntimeError(
        "❌ Se detectó solapamiento de image_id entre splits."
    )

print("✅ Ningún image_id cruza splits.")

# %% [markdown] Cell 31
# ## 13. Figura de ejemplos de near-duplicates
# Se guarda una cuadrícula con hasta 12 pares confirmados para inspección visual.

# %% Cell 32
import matplotlib.pyplot as plt

if len(confirmed_train_pairs) > 0:
    sample_pairs = (
        confirmed_train_pairs
        .sort_values("ssim", ascending=False)
        .head(12)
    )

    fig, axes = plt.subplots(
        len(sample_pairs),
        2,
        figsize=(8, 3 * len(sample_pairs))
    )

    if len(sample_pairs) == 1:
        axes = np.array([axes])

    for row_idx, row in enumerate(
        sample_pairs.itertuples(index=False)
    ):
        for col_idx, image_id in enumerate(
            [row.image_a, row.image_b]
        ):
            with Image.open(train_paths[image_id]) as img:
                axes[row_idx, col_idx].imshow(img.convert("RGB"))

            axes[row_idx, col_idx].axis("off")
            axes[row_idx, col_idx].set_title(image_id)

        axes[row_idx, 0].set_ylabel(
            f"H={row.phash_hamming}\nSSIM={row.ssim:.4f}"
        )

    plt.tight_layout()

    fig_path = (
        FIGURES_DIR /
        "03_confirmed_near_duplicate_examples.png"
    )

    plt.savefig(
        fig_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print("✅ Figura:", fig_path)
else:
    print("No hubo pares confirmados para graficar.")

# %% [markdown] Cell 33
# ## 14. Resumen

# %% Cell 34
summary_rows = [
    {
        "metric": "original_images",
        "value": len(df)
    },
    {
        "metric": "exact_deduplicated_images",
        "value": len(dedup_df)
    },
    {
        "metric": "confirmed_train_near_duplicate_pairs",
        "value": len(confirmed_train_pairs)
    },
    {
        "metric": "nearclean_images",
        "value": len(nearclean_df)
    },
    {
        "metric": "additional_images_removed_near_duplicates",
        "value": removed_near
    },
    {
        "metric": "confirmed_train_test_near_duplicate_pairs",
        "value": len(confirmed_cross_pairs)
    },
    {
        "metric": "unique_test_images_near_duplicate_with_train",
        "value": (
            confirmed_cross_pairs["test_image"].nunique()
            if len(confirmed_cross_pairs)
            else 0
        )
    },
    {
        "metric": "visual_groups_with_metadata_inconsistency",
        "value": len(problem_groups)
    }
]

summary_df = pd.DataFrame(summary_rows)

display(summary_df)

summary_df.to_csv(
    RESULTS_DIR / "03_near_duplicate_summary.csv",
    index=False
)

print("=" * 80)
print("✅ NOTEBOOK 03 FINALIZADO")
print("=" * 80)
print("Todavía NO entrenar modelos.")

# %% [markdown] Cell 35
# ## 15. Exportar resultados a ZIP
# El ZIP se guarda también en `PADDY_DOCTOR_Q1/exports/` y se descarga automáticamente.
# La carpeta privada `config/` **no se incluye**.

# %% Cell 36
import shutil
import time
from google.colab import files

timestamp = time.strftime("%Y%m%d_%H%M%S")
PACKAGE_NAME = f"03_PaddyDoctor_NearDuplicates_MatchedProtocols_RESULTS_{timestamp}"

TEMP_PACKAGE = Path("/content") / PACKAGE_NAME

if TEMP_PACKAGE.exists():
    shutil.rmtree(TEMP_PACKAGE)

TEMP_PACKAGE.mkdir(parents=True, exist_ok=True)

# Solo archivos de este Notebook 03 para evitar ZIPs innecesariamente enormes.
include_patterns = {
    "data/metadata": [
        "03_*.csv"
    ],
    "data/splits": [
        "P1_group_clean*.csv",
        "P4*.csv",
        "03_*.csv"
    ],
    "results": [
        "03_*.csv"
    ],
    "figures": [
        "03_*.png"
    ]
}

copied = []

for relative_dir, patterns in include_patterns.items():
    source_dir = PROJECT / relative_dir

    if not source_dir.exists():
        continue

    dest_dir = TEMP_PACKAGE / relative_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    for pattern in patterns:
        for source in source_dir.glob(pattern):
            if source.is_file():
                destination = dest_dir / source.name
                shutil.copy2(source, destination)
                copied.append(str(source.relative_to(PROJECT)))

# README
with open(
    TEMP_PACKAGE / "README_RESULTS.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(
        "Paddy Doctor Q1 — Notebook 03\n"
        "Near-duplicate audit + matched cross-cultivar benchmark\n"
        f"Export timestamp: {timestamp}\n\n"
        "Files included:\n"
    )

    for item in copied:
        f.write(f"- {item}\n")

zip_base = EXPORTS_DIR / PACKAGE_NAME

zip_path = shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=TEMP_PACKAGE
)

size_mb = Path(zip_path).stat().st_size / (1024 ** 2)

print("=" * 80)
print("✅ ZIP CREADO")
print("=" * 80)
print(zip_path)
print(f"Tamaño: {size_mb:.2f} MB")

files.download(zip_path)

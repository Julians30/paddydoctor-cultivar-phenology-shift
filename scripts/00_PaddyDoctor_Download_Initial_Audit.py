"""Auto-exported from 00_PaddyDoctor_Download_Initial_Audit.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 00
# ## Descarga oficial desde KaggleHub + auditoría inicial
# Descarga el dataset, verifica su estructura, cuenta clases y guarda metadatos iniciales en Drive.
# 
# **No requiere GPU.**

# %% [markdown] Cell 2
# ## Conexión obligatoria a Google Drive
# 
# Este notebook es **autónomo**. Puede ejecutarse en una sesión nueva de Google Colab.
# 
# La primera etapa:
# 1. Monta `Mi unidad`.
# 2. Verifica/crea `PADDY_DOCTOR_Q1`.
# 3. Verifica/crea las carpetas del proyecto.
# 4. Define las rutas permanentes que utilizará el notebook.

# %% Cell 3
# ======================================================================
# CONEXIÓN A GOOGLE DRIVE + ESTRUCTURA DEL PROYECTO
# ======================================================================

from google.colab import drive
from pathlib import Path

# Montar Mi unidad
drive.mount("/content/drive")

# Carpeta principal del proyecto
PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")

# Estructura permanente
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

# Crear las carpetas si no existen
for folder in [
    PROJECT,
    DATA_DIR,
    RAW_DIR,
    METADATA_DIR,
    SPLITS_DIR,
    PROCESSED_DIR,
    NOTEBOOKS_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    MANUSCRIPT_DIR,
    EXPORTS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("✅ GOOGLE DRIVE CONECTADO")
print("=" * 80)
print(f"Proyecto      : {PROJECT}")
print(f"Metadata      : {METADATA_DIR}")
print(f"Splits        : {SPLITS_DIR}")
print(f"Resultados    : {RESULTS_DIR}")
print(f"Figuras       : {FIGURES_DIR}")
print(f"Exportaciones : {EXPORTS_DIR}")

# %% [markdown] Cell 4
# ## Autenticación persistente de Kaggle
# 
# Cada sesión nueva de Colab borra el directorio local `~/.kaggle`.  
# Esta celda restaura automáticamente `kaggle.json` desde Google Drive.
# 
# - Primera ejecución: solicita `kaggle.json` y guarda una copia privada en  
#   `PADDY_DOCTOR_Q1/config/kaggle.json`.
# - Ejecuciones posteriores: lo recupera automáticamente desde Drive.
# - La carpeta `config/` **no se incluye en los ZIP de resultados**.

# %% Cell 5
# ======================================================================
# AUTENTICACIÓN PERSISTENTE DE KAGGLE
# ======================================================================

from google.colab import files
from pathlib import Path
import shutil
import os
import sys
import subprocess

CONFIG_DIR = PROJECT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Copia privada persistente en Mi unidad
DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"

# Ubicación que Kaggle/KaggleHub reconoce en cada runtime
LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"

if not DRIVE_KAGGLE_JSON.exists():
    print("=" * 80)
    print("🔑 PRIMERA CONFIGURACIÓN DE KAGGLE")
    print("=" * 80)
    print("Selecciona tu archivo kaggle.json.")
    print("Se guardará una copia privada en:")
    print(DRIVE_KAGGLE_JSON)

    uploaded = files.upload()

    if "kaggle.json" not in uploaded:
        raise FileNotFoundError(
            "❌ Debes seleccionar exactamente el archivo kaggle.json."
        )

    with open(DRIVE_KAGGLE_JSON, "wb") as f:
        f.write(uploaded["kaggle.json"])

    print("✅ kaggle.json guardado en Mi unidad.")
else:
    print("✅ Credencial persistente encontrada en Mi unidad.")

# Restaurar credencial al runtime actual
shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
os.chmod(LOCAL_KAGGLE_JSON, 0o600)

# Asegurar KaggleHub actualizado
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "kagglehub"],
    check=True
)

import kagglehub

print("✅ Kaggle autenticado para esta sesión de Colab.")
print("✅ Credencial restaurada en:", LOCAL_KAGGLE_JSON)

# %% Cell 6
from google.colab import drive, files
from pathlib import Path
import os, sys, subprocess, pandas as pd

# %% Cell 7
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kagglehub"], check=True)
import kagglehub
print("✅ kagglehub instalado")

# %% Cell 8
import kagglehub
path = kagglehub.competition_download("paddy-disease-classification")
DATASET_PATH = Path(path)

print("✅ Dataset:", DATASET_PATH)
for item in sorted(DATASET_PATH.iterdir()):
    print(("📁" if item.is_dir() else "📄"), item.name)

# %% Cell 9
TRAIN_CSV = DATASET_PATH / "train.csv"
TRAIN_IMAGES = DATASET_PATH / "train_images"
TEST_IMAGES = DATASET_PATH / "test_images"

df = pd.read_csv(TRAIN_CSV)
print(f"Registros: {len(df):,}")
print("Columnas:", list(df.columns))
display(df.head(10))

# %% Cell 10
counts = df["label"].value_counts().rename_axis("class_name").reset_index(name="n_images")
counts["percentage"] = (counts["n_images"] / len(df) * 100).round(2)
counts["class_id"] = range(len(counts))
counts = counts[["class_id","class_name","n_images","percentage"]]
display(counts)

largest = counts["n_images"].max()
smallest = counts["n_images"].min()
print(f"Clases: {len(counts)}")
print(f"Imágenes: {counts['n_images'].sum():,}")
print(f"Ratio max/min: {largest/smallest:.2f}:1")

# %% Cell 11
VALID_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
folders = sorted([p for p in TRAIN_IMAGES.iterdir() if p.is_dir()], key=lambda p: p.name)

audit = []
for folder in folders:
    imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
    audit.append({"class_name": folder.name, "n_images": len(imgs)})

folder_df = pd.DataFrame(audit)
display(folder_df)
print("Total físico:", folder_df["n_images"].sum())

# %% Cell 12
df.to_csv(METADATA_DIR / "paddy_doctor_train_original.csv", index=False)
counts.to_csv(METADATA_DIR / "paddy_doctor_classes.csv", index=False)
folder_df.to_csv(METADATA_DIR / "paddy_doctor_folder_audit.csv", index=False)

print("✅ Metadata inicial guardada en:", METADATA_DIR)
print("❌ Todavía no entrenar modelos.")

# %% [markdown] Cell 13
# ## Exportar resultados — 00_PaddyDoctor_Download_Initial_Audit
# 
# Esta celda crea un archivo ZIP con los resultados generados por este notebook y lo descarga automáticamente.
# 
# Después puedes subir ese ZIP directamente al chat para su revisión.

# %% Cell 14
# ======================================================================
# EXPORTAR RESULTADOS DEL 00_PaddyDoctor_Download_Initial_Audit
# ======================================================================

from pathlib import Path
from google.colab import files
import shutil
import time

PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
EXPORT_ROOT = PROJECT / "exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

timestamp = time.strftime("%Y%m%d_%H%M%S")
PACKAGE_NAME = f"00_PaddyDoctor_Download_Initial_Audit_RESULTS_{timestamp}"

TEMP_PACKAGE = Path("/content") / PACKAGE_NAME

if TEMP_PACKAGE.exists():
    shutil.rmtree(TEMP_PACKAGE)

TEMP_PACKAGE.mkdir(parents=True, exist_ok=True)

include_dirs = ['data/metadata/paddy_doctor_train_original.csv', 'data/metadata/paddy_doctor_classes.csv', 'data/metadata/paddy_doctor_folder_audit.csv']

copied_anything = False

for relative_path in include_dirs:
    source = PROJECT / relative_path

    if not source.exists():
        print(f"⚠️ No existe: {source}")
        continue

    destination = TEMP_PACKAGE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)

    copied_anything = True
    print(f"✅ Incluido: {relative_path}")

# Crear README del paquete
readme = TEMP_PACKAGE / "README_RESULTS.txt"

with open(readme, "w", encoding="utf-8") as f:
    f.write(
        "00_PaddyDoctor_Download_Initial_Audit\n"
        "Proyecto: PADDY_DOCTOR_Q1\n"
        f"Fecha de exportación: {timestamp}\n\n"
        "Este paquete contiene los resultados generados por el notebook.\n"
        "Puedes subir este ZIP directamente a ChatGPT para revisión.\n"
    )

if not copied_anything:
    print("⚠️ No se encontró ningún resultado, pero se creará el paquete con el README.")

zip_base = EXPORT_ROOT / PACKAGE_NAME

zip_path = shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=TEMP_PACKAGE
)

print("\n" + "=" * 80)
print("✅ ZIP DE RESULTADOS CREADO")
print("=" * 80)
print(zip_path)

size_mb = Path(zip_path).stat().st_size / (1024**2)
print(f"Tamaño: {size_mb:.2f} MB")

print("\n⬇️ Iniciando descarga al navegador...")

files.download(zip_path)

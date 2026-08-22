"""Auto-exported from 01_PaddyDoctor_Scientific_Audit.ipynb.
The notebook remains the canonical executable record; this script mirrors its code cells.
"""

# %% [markdown] Cell 1
# # Paddy Doctor Q1 — Notebook 01
# ## Auditoría científica del dataset
# Analiza variedad, edad, dimensiones, archivos corruptos y duplicados exactos.
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
for folder in [PROJECT,DATA_DIR,RAW_DIR,METADATA_DIR,SPLITS_DIR,PROCESSED_DIR,NOTEBOOKS_DIR,MODELS_DIR,RESULTS_DIR,FIGURES_DIR,MANUSCRIPT_DIR,EXPORTS_DIR]:
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

# %% Cell 5
from google.colab import files
from pathlib import Path
import shutil, os, sys, subprocess
CONFIG_DIR = PROJECT / "config"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_KAGGLE_JSON = CONFIG_DIR / "kaggle.json"
LOCAL_KAGGLE_DIR = Path.home() / ".kaggle"
LOCAL_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_KAGGLE_JSON = LOCAL_KAGGLE_DIR / "kaggle.json"
if not DRIVE_KAGGLE_JSON.exists():
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise FileNotFoundError("❌ Debes seleccionar exactamente el archivo kaggle.json.")
    with open(DRIVE_KAGGLE_JSON, "wb") as f:
        f.write(uploaded["kaggle.json"])
else:
    print("✅ Credencial persistente encontrada en Mi unidad.")
shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
os.chmod(LOCAL_KAGGLE_JSON, 0o600)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kagglehub"], check=True)
import kagglehub
print("✅ Kaggle autenticado para esta sesión de Colab.")

# %% Cell 6
from pathlib import Path
import pandas as pd
import hashlib, warnings
from PIL import Image, ImageFile
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = False
PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
METADATA_DIR = PROJECT / "data" / "metadata"
FIGURES_DIR = PROJECT / "figures"
DATASET_PATH = Path("/root/.cache/kagglehub/competitions/paddy-disease-classification")
TRAIN_CSV = DATASET_PATH / "train.csv"
TRAIN_IMAGES = DATASET_PATH / "train_images"
if not TRAIN_CSV.exists():
    import kagglehub
    DATASET_PATH = Path(kagglehub.competition_download("paddy-disease-classification"))
    TRAIN_CSV = DATASET_PATH / "train.csv"
    TRAIN_IMAGES = DATASET_PATH / "train_images"
df = pd.read_csv(TRAIN_CSV)
print(f"Imágenes: {len(df):,}")
print(f"Clases: {df['label'].nunique()}")
print(f"Variedades: {df['variety'].nunique()}")
print(f"Edades: {df['age'].nunique()}")

# %% Cell 7
variety_counts = df["variety"].value_counts().rename_axis("variety").reset_index(name="n_images")
variety_counts["percentage"] = (variety_counts["n_images"]/len(df)*100).round(2)
display(variety_counts)
variety_counts.to_csv(METADATA_DIR/"audit_variety_distribution.csv", index=False)
age_counts = df["age"].value_counts().sort_index().rename_axis("age").reset_index(name="n_images")
age_counts["percentage"] = (age_counts["n_images"]/len(df)*100).round(2)
display(age_counts)
age_counts.to_csv(METADATA_DIR/"audit_age_distribution.csv", index=False)

# %% Cell 8
class_variety = pd.crosstab(df["label"], df["variety"])
display(class_variety)
class_variety.to_csv(METADATA_DIR/"audit_class_by_variety.csv")
class_variety_pct = (pd.crosstab(df["label"], df["variety"], normalize="index")*100).round(2)
display(class_variety_pct)
class_variety_pct.to_csv(METADATA_DIR/"audit_class_by_variety_percentage.csv")

# %% Cell 9
records = []
for label in sorted(df["label"].unique()):
    sub = df[df["label"] == label]
    vc = sub["variety"].value_counts()
    records.append({"class_name": label,"n_images": len(sub),"n_varieties": sub["variety"].nunique(),"dominant_variety": vc.index[0],"dominant_variety_n": int(vc.iloc[0]),"dominant_variety_percentage": round(vc.iloc[0]/len(sub)*100,2)})
concentration_df = pd.DataFrame(records).sort_values("dominant_variety_percentage", ascending=False)
display(concentration_df)
concentration_df.to_csv(METADATA_DIR/"audit_class_variety_concentration.csv", index=False)

# %% Cell 10
class_age = pd.crosstab(df["label"], df["age"])
display(class_age)
class_age.to_csv(METADATA_DIR/"audit_class_by_age.csv")
age_by_class = df.groupby("label")["age"].agg(n="count", mean="mean", std="std", median="median", minimum="min", maximum="max", unique_ages="nunique").round(2).reset_index()
display(age_by_class)
age_by_class.to_csv(METADATA_DIR/"audit_age_by_class.csv", index=False)

# %% Cell 11
VALID_EXTENSIONS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
all_images = [p for p in TRAIN_IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
image_records, corrupt_records = [], []
for path in tqdm(all_images, desc="Imágenes"):
    try:
        with Image.open(path) as img: img.verify()
        with Image.open(path) as img: width, height = img.size; mode, fmt = img.mode, img.format
        image_records.append({"filename": path.name,"class_name": path.parent.name,"width": width,"height": height,"aspect_ratio": round(width/height,4),"megapixels": round(width*height/1_000_000,4),"mode": mode,"format": fmt,"file_size_kb": round(path.stat().st_size/1024,2),"image_path": str(path)})
    except Exception as e:
        corrupt_records.append({"filename": path.name,"class_name": path.parent.name,"image_path": str(path),"error": str(e)})
image_info = pd.DataFrame(image_records)
corrupt_df = pd.DataFrame(corrupt_records)
print(f"Válidas: {len(image_info):,}")
print(f"Corruptas: {len(corrupt_df):,}")
image_info.to_csv(METADATA_DIR/"audit_image_properties.csv", index=False)
corrupt_df.to_csv(METADATA_DIR/"audit_corrupt_images.csv", index=False)

# %% Cell 12
display(image_info[["width","height","aspect_ratio","megapixels","file_size_kb"]].describe().round(2))
resolution_counts = image_info.groupby(["width","height"]).size().reset_index(name="n_images").sort_values("n_images", ascending=False)
display(resolution_counts.head(20))
resolution_counts.to_csv(METADATA_DIR/"audit_resolutions.csv", index=False)

# %% Cell 13
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024*1024)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()
hash_records = []
for path in tqdm(all_images, desc="SHA-256"):
    hash_records.append({"filename": path.name,"class_name": path.parent.name,"sha256": sha256_file(path),"image_path": str(path)})
hash_df = pd.DataFrame(hash_records)
dup_mask = hash_df.duplicated(subset=["sha256"], keep=False)
duplicates_df = hash_df[dup_mask].sort_values("sha256").reset_index(drop=True)
n_duplicate_files = len(duplicates_df)
n_duplicate_groups = duplicates_df["sha256"].nunique() if len(duplicates_df) else 0
print("Archivos en duplicados:", n_duplicate_files)
print("Grupos duplicados:", n_duplicate_groups)
duplicates_df.to_csv(METADATA_DIR/"audit_exact_duplicates.csv", index=False)

# %% Cell 14
cross_class = []
if len(duplicates_df):
    for sha, group in duplicates_df.groupby("sha256"):
        classes = sorted(group["class_name"].unique())
        if len(classes) > 1:
            cross_class.append({"sha256": sha,"n_files": len(group),"classes": " | ".join(classes),"n_classes": len(classes)})
cross_class_df = pd.DataFrame(cross_class)
if len(cross_class_df) == 0: print("✅ Ningún duplicado exacto con etiquetas diferentes")
else: display(cross_class_df)
cross_class_df.to_csv(METADATA_DIR/"audit_cross_class_exact_duplicates.csv", index=False)

# %% Cell 15
image_lookup = image_info[["filename","width","height","aspect_ratio","megapixels","file_size_kb"]].rename(columns={"filename":"image_id"})
master_audit = df.merge(image_lookup, on="image_id", how="left")
master_audit.to_csv(METADATA_DIR/"paddy_doctor_master_audit.csv", index=False)
print("="*80)
print("RESUMEN NOTEBOOK 01")
print("="*80)
print(f"Imágenes: {len(df):,}")
print(f"Variedades: {df['variety'].nunique()}")
print(f"Edades: {df['age'].nunique()}")
print(f"Corruptas: {len(corrupt_df)}")
print(f"Archivos en duplicados: {n_duplicate_files}")
print(f"Grupos duplicados: {n_duplicate_groups}")
print(f"Duplicados entre clases: {len(cross_class_df)}")
print("✅ NOTEBOOK 01 FINALIZADO")

# %% [markdown] Cell 16
# ## Exportar resultados — 01_PaddyDoctor_Scientific_Audit

# %% Cell 17
from pathlib import Path
from google.colab import files
import shutil, time
PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
EXPORT_ROOT = PROJECT / "exports"
EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")
PACKAGE_NAME = f"01_PaddyDoctor_Scientific_Audit_RESULTS_{timestamp}"
TEMP_PACKAGE = Path("/content") / PACKAGE_NAME
if TEMP_PACKAGE.exists(): shutil.rmtree(TEMP_PACKAGE)
TEMP_PACKAGE.mkdir(parents=True, exist_ok=True)
include_dirs = ['data/metadata', 'figures']
copied_anything = False
for relative_path in include_dirs:
    source = PROJECT / relative_path
    if not source.exists():
        print(f"⚠️ No existe: {source}"); continue
    destination = TEMP_PACKAGE / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir(): shutil.copytree(source, destination, dirs_exist_ok=True)
    else: shutil.copy2(source, destination)
    copied_anything = True
    print(f"✅ Incluido: {relative_path}")
readme = TEMP_PACKAGE / "README_RESULTS.txt"
with open(readme, "w", encoding="utf-8") as f:
    f.write("01_PaddyDoctor_Scientific_Audit\nProyecto: PADDY_DOCTOR_Q1\n" + f"Fecha de exportación: {timestamp}\n\n" + "Este paquete contiene los resultados generados por el notebook.\nPuedes subir este ZIP directamente a ChatGPT para revisión.\n")
zip_base = EXPORT_ROOT / PACKAGE_NAME
zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=TEMP_PACKAGE)
print("\n" + "=" * 80)
print("✅ ZIP DE RESULTADOS CREADO")
print("=" * 80)
print(zip_path)
size_mb = Path(zip_path).stat().st_size / (1024**2)
print(f"Tamaño: {size_mb:.2f} MB")
print("\n⬇️ Iniciando descarga al navegador...")
files.download(zip_path)

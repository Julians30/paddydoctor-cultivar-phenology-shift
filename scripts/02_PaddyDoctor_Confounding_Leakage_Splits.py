"""Code-cell export of 02_PaddyDoctor_Confounding_Leakage_Splits.ipynb.
Notebook is the canonical executable source.
"""

# %% Code cell 1
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
for folder in [PROJECT,DATA_DIR,RAW_DIR,METADATA_DIR,SPLITS_DIR,PROCESSED_DIR,NOTEBOOKS_DIR,MODELS_DIR,RESULTS_DIR,FIGURES_DIR,MANUSCRIPT_DIR,EXPORTS_DIR]: folder.mkdir(parents=True, exist_ok=True)

# %% Code cell 2
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
    if "kaggle.json" not in uploaded: raise FileNotFoundError("❌ Debes seleccionar exactamente el archivo kaggle.json.")
    with open(DRIVE_KAGGLE_JSON, "wb") as f: f.write(uploaded["kaggle.json"])
shutil.copy2(DRIVE_KAGGLE_JSON, LOCAL_KAGGLE_JSON)
os.chmod(LOCAL_KAGGLE_JSON, 0o600)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kagglehub"], check=True)
import kagglehub

# %% Code cell 3
from pathlib import Path
import hashlib, warnings
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from scipy.stats import chi2_contingency
warnings.filterwarnings("ignore")
PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
METADATA_DIR = PROJECT / "data" / "metadata"
SPLITS_DIR = PROJECT / "data" / "splits"
RESULTS_DIR = PROJECT / "results"
for p in [METADATA_DIR, SPLITS_DIR, RESULTS_DIR]: p.mkdir(parents=True, exist_ok=True)
DATASET_PATH = Path("/root/.cache/kagglehub/competitions/paddy-disease-classification")
TRAIN_CSV = DATASET_PATH / "train.csv"
TRAIN_IMAGES = DATASET_PATH / "train_images"
TEST_IMAGES = DATASET_PATH / "test_images"
if not TRAIN_CSV.exists():
    import kagglehub
    DATASET_PATH = Path(kagglehub.competition_download("paddy-disease-classification"))
    TRAIN_CSV = DATASET_PATH / "train.csv"
    TRAIN_IMAGES = DATASET_PATH / "train_images"
    TEST_IMAGES = DATASET_PATH / "test_images"
df = pd.read_csv(TRAIN_CSV)

# %% Code cell 4
def cramers_v(x, y):
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.values.sum(); phi2 = chi2 / n; r, k = table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1); kcorr = k - ((k-1)**2)/(n-1)
    denom = min(kcorr-1, rcorr-1)
    return np.sqrt(phi2corr/denom) if denom > 0 else np.nan
v_variety = cramers_v(df["label"], df["variety"])
v_age = cramers_v(df["label"], df["age"].astype(str))
pd.DataFrame({"association":["disease_vs_variety","disease_vs_age"],"cramers_v":[v_variety,v_age]}).to_csv(RESULTS_DIR/"02_metadata_label_association.csv", index=False)

# %% Code cell 5
def evaluate_metadata_model(dataframe, features, name):
    X = dataframe[features].copy(); y = dataframe["label"].copy()
    categorical = [c for c in features if c == "variety"]
    numeric = [c for c in features if c == "age"]
    transformers = []
    if categorical: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric: transformers.append(("num", StandardScaler(), numeric))
    model = Pipeline([("preprocess", ColumnTransformer(transformers)),("classifier", LogisticRegression(max_iter=3000, class_weight="balanced"))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026); rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y), 1):
        model.fit(X.iloc[tr], y.iloc[tr]); pred = model.predict(X.iloc[te])
        rows.append({"model":name,"fold":fold,"accuracy":accuracy_score(y.iloc[te],pred),"balanced_accuracy":balanced_accuracy_score(y.iloc[te],pred),"macro_f1":f1_score(y.iloc[te],pred,average="macro")})
    return pd.DataFrame(rows)
metadata_results = pd.concat([evaluate_metadata_model(df,["variety"],"variety_only"),evaluate_metadata_model(df,["age"],"age_only"),evaluate_metadata_model(df,["variety","age"],"variety_plus_age")], ignore_index=True)
summary_metadata = metadata_results.groupby("model").agg(accuracy_mean=("accuracy","mean"),accuracy_std=("accuracy","std"),balanced_accuracy_mean=("balanced_accuracy","mean"),macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std")).reset_index().round(4)
metadata_results.to_csv(RESULTS_DIR/"02_metadata_only_cv_raw.csv", index=False)
summary_metadata.to_csv(RESULTS_DIR/"02_metadata_only_cv_summary.csv", index=False)

# %% Code cell 6
VALID_EXTENSIONS = {".jpg",".jpeg",".png"}
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024*1024)
            if not block: break
            h.update(block)
    return h.hexdigest()
train_paths = {p.name:p for p in TRAIN_IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS}
df["sha256"] = [sha256_file(train_paths[image_id]) if image_id in train_paths else None for image_id in tqdm(df["image_id"], desc="SHA train")]

# %% Code cell 7
duplicate_groups = df.groupby("sha256", dropna=True).filter(lambda g: len(g) > 1)
rows = []
for sha, group in duplicate_groups.groupby("sha256"):
    rows.append({"sha256":sha,"n_files":len(group),"n_labels":group["label"].nunique(),"n_varieties":group["variety"].nunique(),"n_ages":group["age"].nunique(),"labels":" | ".join(sorted(group["label"].astype(str).unique())),"varieties":" | ".join(sorted(group["variety"].astype(str).unique())),"ages":" | ".join(sorted(group["age"].astype(str).unique()))})
metadata_consistency_df = pd.DataFrame(rows)
if len(metadata_consistency_df):
    inconsistent_metadata = metadata_consistency_df[(metadata_consistency_df["n_labels"]>1)|(metadata_consistency_df["n_varieties"]>1)|(metadata_consistency_df["n_ages"]>1)]
else: inconsistent_metadata = pd.DataFrame()
metadata_consistency_df.to_csv(METADATA_DIR/"02_duplicate_metadata_consistency.csv", index=False)
inconsistent_metadata.to_csv(METADATA_DIR/"02_duplicate_metadata_inconsistency.csv", index=False)

# %% Code cell 8
dedup_df = df.sort_values("image_id").drop_duplicates(subset="sha256", keep="first").reset_index(drop=True)
dedup_df.to_csv(METADATA_DIR/"02_paddy_doctor_deduplicated.csv", index=False)

# %% Code cell 9
test_paths = [p for p in TEST_IMAGES.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS]
train_hash_set = set(df["sha256"].dropna()); overlap = []
for path in tqdm(test_paths, desc="SHA test"):
    sha = sha256_file(path)
    if sha in train_hash_set:
        for _, row in df[df["sha256"] == sha].iterrows():
            overlap.append({"test_image":path.name,"train_image":row["image_id"],"train_label":row["label"],"train_variety":row["variety"],"train_age":row["age"],"sha256":sha})
overlap_df = pd.DataFrame(overlap)
overlap_df.to_csv(METADATA_DIR/"02_train_test_exact_overlap.csv", index=False)

# %% Code cell 10
support = pd.crosstab(dedup_df["label"], np.where(dedup_df["variety"] == "ADT45", "ADT45", "OTHER"))
support.to_csv(METADATA_DIR/"02_class_support_ADT45_vs_other.csv")
if not (support > 0).all(axis=1).all(): raise RuntimeError("No todas las clases aparecen en ambos dominios")

# %% Code cell 11
train_val, test = train_test_split(dedup_df, test_size=0.15, stratify=dedup_df["label"], random_state=2026)
train, val = train_test_split(train_val, test_size=0.1764705882, stratify=train_val["label"], random_state=2026)
p1 = dedup_df.copy(); p1["split"] = "UNASSIGNED"
p1.loc[p1["image_id"].isin(train["image_id"]), "split"] = "train"
p1.loc[p1["image_id"].isin(val["image_id"]), "split"] = "validation"
p1.loc[p1["image_id"].isin(test["image_id"]), "split"] = "test"
p1["protocol"] = "P1_clean_IID"; p1.to_csv(SPLITS_DIR/"P1_clean_IID.csv", index=False)

# %% Code cell 12
def build_domain_protocol(dataframe, training_mask, test_mask, protocol_name):
    train_domain = dataframe[training_mask].copy(); external_test = dataframe[test_mask].copy()
    development_train, validation = train_test_split(train_domain,test_size=0.15,stratify=train_domain["label"],random_state=2026)
    output = dataframe.copy(); output["split"] = "EXCLUDED"
    output.loc[output["image_id"].isin(development_train["image_id"]), "split"] = "train"
    output.loc[output["image_id"].isin(validation["image_id"]), "split"] = "validation"
    output.loc[output["image_id"].isin(external_test["image_id"]), "split"] = "test"
    output["protocol"] = protocol_name
    return output
p2 = build_domain_protocol(dedup_df,dedup_df["variety"]=="ADT45",dedup_df["variety"]!="ADT45","P2_ADT45_to_OTHER")
p3 = build_domain_protocol(dedup_df,dedup_df["variety"]!="ADT45",dedup_df["variety"]=="ADT45","P3_OTHER_to_ADT45")
p2.to_csv(SPLITS_DIR/"P2_ADT45_to_OTHER.csv", index=False); p3.to_csv(SPLITS_DIR/"P3_OTHER_to_ADT45.csv", index=False)

# %% Code cell 13
protocols = {"P1_clean_IID":p1,"P2_ADT45_to_OTHER":p2,"P3_OTHER_to_ADT45":p3}
summary = []
for name, data in protocols.items():
    active = data[data["split"] != "EXCLUDED"]
    for split_name, sub in active.groupby("split"):
        summary.append({"protocol":name,"split":split_name,"n_images":len(sub),"n_classes":sub["label"].nunique(),"n_varieties":sub["variety"].nunique(),"min_age":sub["age"].min(),"max_age":sub["age"].max()})
protocol_summary_df = pd.DataFrame(summary); protocol_summary_df.to_csv(SPLITS_DIR/"protocol_summary.csv", index=False)
for name, data in protocols.items():
    active = data[data["split"] != "EXCLUDED"]
    pd.crosstab(active["label"], active["split"]).to_csv(SPLITS_DIR/f"{name}_class_distribution.csv")

# %% Code cell 14
print("✅ NOTEBOOK 02 FINALIZADO")

# %% Code cell 15
from pathlib import Path
from google.colab import files
import shutil, time
PROJECT = Path("/content/drive/MyDrive/PADDY_DOCTOR_Q1")
EXPORT_ROOT = PROJECT / "exports"; EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S"); PACKAGE_NAME = f"02_PaddyDoctor_Confounding_Leakage_Splits_RESULTS_{timestamp}"
TEMP_PACKAGE = Path("/content") / PACKAGE_NAME
if TEMP_PACKAGE.exists(): shutil.rmtree(TEMP_PACKAGE)
TEMP_PACKAGE.mkdir(parents=True, exist_ok=True)
for relative_path in ['data/metadata','data/splits','results']:
    source = PROJECT / relative_path
    if source.exists():
        destination = TEMP_PACKAGE / relative_path; destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir(): shutil.copytree(source, destination, dirs_exist_ok=True)
        else: shutil.copy2(source, destination)
zip_path = shutil.make_archive(str(EXPORT_ROOT / PACKAGE_NAME), "zip", root_dir=TEMP_PACKAGE)
files.download(zip_path)

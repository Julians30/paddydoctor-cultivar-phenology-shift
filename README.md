# Paddy Doctor - Cultivar Shift and Phenological Confounding

Reproducibility repository for the study:

**Reliability of deep vision models under cultivar-associated domain shift in rice disease classification: a phenology-controlled evaluation**

## Scientific objective

This project evaluates whether high in-domain performance in rice disease classification remains reliable under cultivar/domain shift and whether the degradation persists after exact matching on the observed plant-age variable.

The study compares **ResNet-50, ConvNeXt-Tiny, ViT-S/16, and Swin-Tiny** using three pre-specified seeds (**2026, 2027, 2028**) and the staged experimental design **P1 → P4 → P5**.

- **P1:** leakage-audited IID reference.
- **P4:** bidirectional ADT45 ↔ OTHER domain shift, class-matched.
- **P5:** sensitivity analysis with exact disease + age matching.

The dataset variable `age` is treated as the observed plant-age variable and used operationally to control age-related/phenology-related imbalance. It should not be interpreted as a botanically validated phenological-stage annotation unless independently mapped to such stages.

P5 is a sensitivity/control analysis and is **not** interpreted as a pure causal effect of cultivar. In the implemented exact-matching design, P5 uses the **5 disease classes with sufficient shared support**, with **100 matched ADT45–OTHER pairs per class** and a **60/20/20 split by matched pairs**. Therefore, P5 results should not be read as a 10-class benchmark.

### Protocol-development note

Notebook `02_PaddyDoctor_Confounding_Leakage_Splits.ipynb` also generates the directional splits `P2_ADT45_to_OTHER` and `P3_OTHER_to_ADT45`. These are retained for transparency as intermediate protocol-development outputs. The main inferential sequence documented for the manuscript is **P1 → P4 → P5**.

## Repository structure

```text
.
├── notebooks/          # Canonical Colab/Jupyter notebooks
├── scripts/            # Plain-Python exports of notebook code
├── data/README.md      # Dataset access and non-redistribution note
├── docs/
│   ├── REPRODUCIBILITY.md
│   └── ARCHIVING.md
├── results/README.md   # Expected result folders/files
├── figures/README.md   # Figure generation note
├── CITATION.cff
├── requirements.txt
├── environment.yml
└── .gitignore
```

## Canonical notebooks

1. `00_PaddyDoctor_Download_Initial_Audit.ipynb`
2. `01_PaddyDoctor_Scientific_Audit.ipynb`
3. `02_PaddyDoctor_Confounding_Leakage_Splits.ipynb`
4. `03_PaddyDoctor_NearDuplicates_MatchedProtocols.ipynb`
5. `04_PaddyDoctor_PhenologyMatched_Protocols.ipynb`
6. `05_v3_1_PaddyDoctor_P1_CNN_Transformer_Benchmark_30epochs_noEarlyStop_FIXED.ipynb`
7. `06_PaddyDoctor_P4_CrossCultivar_30epochs_noEarlyStop.ipynb`
8. `07_v1_1_PaddyDoctor_P5_ExactAgeMatched_30epochs_noEarlyStop_FIXED.ipynb`
9. `08_PaddyDoctor_Joint_Statistical_Inference_P1_P4_P5.ipynb`
10. `09_v1_1_PaddyDoctor_Calibration_Reliability_TemperatureScaling_FIXED.ipynb`
11. `10_PaddyDoctor_Explainability_Faithfulness.ipynb`
12. `11_PaddyDoctor_Computational_Efficiency_Benchmark.ipynb`
13. `12_PaddyDoctor_Final_Article_Tables_Figures.ipynb`

## Data

Primary images come from the public **Paddy Doctor / Paddy Disease Classification** dataset. The raw image dataset is **not redistributed in this repository**. Notebook `00_PaddyDoctor_Download_Initial_Audit.ipynb` downloads the official source and creates the project structure in Google Drive.

The original project path used in Colab is:

```text
/content/drive/MyDrive/PADDY_DOCTOR_Q1
```

For archival reproducibility, lightweight derived metadata and the frozen split/pair assignment files may be released separately when redistribution is appropriate. Raw images, checkpoints, credentials, and other large/private artifacts should remain outside GitHub.

## Main reproducibility safeguards

- Exact duplicate audit with SHA-256.
- Near-duplicate/perceptual audit with pHash and SSIM.
- Scientific exclusion of the official Kaggle test set when overlap is detected.
- Frozen protocol definitions before model comparison.
- Three pre-specified training seeds.
- Common optimization budget across architectures.
- Temperature scaling fitted only on source-domain validation and frozen for target test.
- Paired stratified bootstrap, exact McNemar with Holm correction, and GEE with robust standard errors.
- Explainability interpreted within architecture/method, not as a cross-method ranking.

## Training configuration

The manuscript reports a common training budget of 30 epochs, 224×224 input resolution, batch size 32, AdamW, initial learning rate 3×10⁻⁴, weight decay 0.05, two warm-up epochs, cosine annealing, ImageNet-1k initialization, full fine-tuning, and checkpoint selection by validation Macro-F1.

## Hardware/software reference

Experiments were executed on an NVIDIA L4 (22 GB). The manuscript records PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, and timm 1.0.28. P1 and P4/P5 were executed in different Colab Python runtimes; see `docs/REPRODUCIBILITY.md`.

Because Colab runtimes evolve, `requirements.txt` and `environment.yml` are convenience specifications rather than byte-for-byte reconstructions of every historical runtime. Exact environment snapshots should be archived before public release when available.

## Automated repository audit

A GitHub Actions workflow under `.github/workflows/repository-audit.yml` checks the expected notebook/script inventory, notebook JSON validity, Python syntax, misplaced notebooks, and accidentally tracked sensitive/large artifacts.

## Citation and archival release

A machine-readable citation file is provided in `CITATION.cff`. Before journal submission/public release, the authors should freeze a versioned release (recommended: `v1.0.0`), archive that release in a persistent repository such as Zenodo, and add the resulting DOI to this README and to the manuscript's Data/Code Availability statements. See `docs/ARCHIVING.md`.

The GitHub repository should remain the actively maintained code repository; the archived DOI should identify the immutable version supporting the submitted manuscript.

## License

No license has been assigned yet. **A code license must be selected by the authors before public release.** This should be resolved before making the repository public so that permitted reuse is explicit.

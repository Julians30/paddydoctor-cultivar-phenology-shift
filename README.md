# Paddy Doctor - Cultivar Shift and Phenological Confounding

Reproducibility repository for the study:

**Cambio de cultivar y confusión fenológica en la clasificación de enfermedades del arroz: evaluación controlada de robustez y confiabilidad de CNN y Vision Transformers**

## Scientific objective

This project evaluates whether high in-domain performance in rice disease classification remains reliable under cultivar/domain shift and whether the degradation persists after exact matching on the observed plant-age variable.

The study compares **ResNet-50, ConvNeXt-Tiny, ViT-S/16, and Swin-Tiny** using three pre-specified seeds (**2026, 2027, 2028**) and the staged experimental design **P1 → P4 → P5**.

- **P1:** leakage-audited IID reference.
- **P4:** bidirectional ADT45 ↔ OTHER domain shift, class-matched.
- **P5:** sensitivity analysis with exact disease + age matching.

P5 is a sensitivity/control analysis and is **not** interpreted as a pure causal effect of cultivar.

## Repository structure

```text
.
├── notebooks/          # Canonical Colab/Jupyter notebooks
├── scripts/            # Plain-Python exports of notebook code
├── data/README.md      # Dataset access and non-redistribution note
├── docs/
│   └── REPRODUCIBILITY.md
├── results/README.md   # Expected result folders/files
├── figures/README.md   # Figure generation note
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

## Citation

The manuscript is under preparation/revision. A DOI/archival citation should be added here after publication or repository archiving.

## License

No license has been assigned yet. Add the license selected by the authors before public release.

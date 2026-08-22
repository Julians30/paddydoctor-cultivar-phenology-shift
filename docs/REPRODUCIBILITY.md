# Reproducibility guide

## Recommended execution order

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

## Runtime

The notebooks are designed for Google Colab and mount Google Drive. The project root is:

`/content/drive/MyDrive/PADDY_DOCTOR_Q1`

The manuscript records:
- NVIDIA L4, 22 GB VRAM
- PyTorch 2.11.0+cu128
- torchvision 0.26.0+cu128
- timm 1.0.28
- Python 3.12.13 for P1
- Python 3.13.15 for P4/P5

Because Colab images evolve, exact future reproduction may require an archived environment or container. The repository therefore records both package requirements and the versions reported in the manuscript.

## Randomness

The pre-specified seeds are 2026, 2027, and 2028. Do not replace these with post-hoc seeds when reproducing the published comparisons.

## Data leakage

Do not use the official competition test split as the main scientific test set if the integrity audit detects exact/perceptual overlap with the labeled training data.

## Interpretation of protocols

P1, P4, and P5 answer different questions. P5 reduces observed age imbalance but does not identify a pure causal cultivar effect because OTHER pools multiple cultivars and unobserved acquisition factors may remain.

## Outputs

Each notebook writes permanent outputs to Google Drive. Generated models, caches, raw images, and large binary artifacts should remain outside GitHub. Small CSV tables, split indices, and final figures can be archived separately once the authors decide which supplementary artifacts will be public.

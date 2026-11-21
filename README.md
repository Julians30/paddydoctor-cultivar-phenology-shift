# Paddy Doctor model reliability under cultivar associated domain shift

Reproducibility materials for **Reliability of deep vision models under cultivar-associated domain shift in rice plant-health classification with exact age matching**.

**Authors:** Julián Coronel-Reyes, Bryan Iván Barahona-Montalván and Bryan Orlando Vélez-San Martín. Correspondence: jcoronel1@utmachala.edu.ec. Identifiers and affiliations: [CITATION.cff](CITATION.cff). Roles: [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).

## Research question and designs

Does predictive performance and confidence reliability deteriorate under cultivar-associated domain shift after exact matching on benchmark class and recorded age? How well do source-fitted calibration and confidence-based acceptance rules transfer?

ResNet-50, ConvNeXt-Tiny, ViT-S/16 and Swin-Tiny use seeds 2026, 2027 and 2028. Controls and transferred models predict **identical target images**. There are **228 completed neural fits**, each with a 30-epoch horizon; calibration fits are counted separately.

| Design | Comparison | Neural fits |
| --- | --- | ---: |
| P1 | Leakage-audited IID reference, ten classes | 12 |
| P4 | ADT45 versus OTHER, matched class composition, ten classes | 24 |
| Reference P5 | Exact class-and-recorded-age pairs, five classes | 24 |
| Repeated P5 | Five overlapping matching constructions, five classes | 120 |
| P6 | Four OTHER cultivars balanced across three classes and paired with ADT45 | 24 |
| P7 | AndraPonni versus REST excluding AndraPonni, three classes | 24 |

P2/P3 are intermediate software labels, not additional evaluated experiments. OTHER in P4/P5 pools nine cultivars. P6 and P7 retain blast, dead heart and hispa and differ in sample size from P5.

## Findings and scope

The 40 repeated-P5 mean losses were positive, with marginal image and seed-and-image intervals above zero. P6 retained losses of 0.1712–0.3723 after cultivar–class balancing; P7 retained losses of 0.2330–0.7255. All 16 paired image intervals were positive. Frozen-feature domain AUC was 0.9537 in P4 restricted to the same five classes as P5 and 0.9392 in P5. Source temperature scaling improved average probability scores without restoring same-domain reliability.

These are internal comparisons within one public collection. Recorded age is not validated phenological stage. Cultivar and unobserved acquisition conditions are not causally separated. Constructions overlap; three seeds and small P6/P7 tests do not establish independent farm or field generalization. The ANOVA is descriptive, without residual replication for a full-interaction F test. A shared epoch horizon does not equalize elapsed compute.

## Contents

| Location | Contents |
| --- | --- |
| [notebooks/](notebooks/) | Colab/Jupyter notebooks 00–19 |
| [scripts/](scripts/) | Python exports and repository audit |
| [reproducibility/](reproducibility/) | Frozen outputs, assignments, histories, numerical sources, environment records and checksum manifest |
| [results/manuscript_tables/](results/manuscript_tables/) | Nine main and 51 supplementary CSV tables |
| [figures/](figures/) | Cultivar-sensitivity and domain-separability figures, PNG/PDF |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Execution routes, environments and limitations |

## Reproduce numerical results without training

```bash
python -m pip install -r requirements-analysis.txt
cd reproducibility
python restore_large_files.py
python verify19.py --results 19_PaddyDoctor_RESULTADOS.zip --code notebook19_code --out reproduced19
```

This checks 48 completed fits, 96 evaluations, 960 result-file hashes, every 30-epoch history, 16 paired loss intervals and domain probabilities/AUC intervals. The feature cache is in the results archive. No original RGB images or GPU are required for this audit. Other numerical routes are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The 35 MB notebook-18 all-image calibrated-probability table is omitted because it can be regenerated; fitted calibrators, predictions, assignments and summary/interval tables are included.

## Reproduce training

Obtain the [Paddy Disease Classification RGB dataset](https://www.kaggle.com/competitions/paddy-disease-classification); it is not redistributed. The original Colab project location is `/content/drive/MyDrive/PADDY_DOCTOR_Q1`. Kaggle credentials stay in the user's runtime and must not be committed.

Notebooks 05–07 cover reference training, 15 covers repeated P5, and 19 covers P6/P7 and frozen features. Exact executed P6/P7 numerical sources are retained in `reproducibility/notebook19_code/` for source-signature checks. Pretraining lineages differ; explicit initialization and source hashes are retained for the 168 repeated-matching and cultivar-sensitivity fits, while some reference initialization records are incomplete.

Notebooks 10 and the equivalence component of 13 are exploratory tools and do not support quantitative claims in the current manuscript. Current endpoints and tables are identified in the supplementary material.

## Citation and license

Use [CITATION.cff](CITATION.cff). This release is permanently archived at Zenodo: **DOI [10.5281/zenodo.24424656](https://doi.org/10.5281/zenodo.24424656)**. A manuscript based on these materials has been prepared for submission to *SN Computer Science* (Topical Collection "AI for Agriculture"); the article DOI will be added here once it is available. [Archiving guidance](docs/ARCHIVING.md) explains how to identify a fixed snapshot.

Original code and associated repository documentation use the **[MIT License](LICENSE)**. Dataset material, pretrained weights and third-party components retain their respective terms; see [licensing scope](docs/LICENSING.md).

The three largest scientific files are distributed as binary parts to keep individual transfers manageable. `restore_large_files.py` reconstructs their exact original bytes and checks both part and full-file SHA-256 values before analysis.

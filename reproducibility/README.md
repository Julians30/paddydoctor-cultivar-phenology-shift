# Experimental records and numerical sources

Derived records support a study with 228 completed neural fits. Original RGB images, credentials and large trained rice-model checkpoint payloads are excluded.

| Location | Purpose |
| --- | --- |
| `inputs/p1.zip`, `p4.zip`, `p5.zip`, `cal.zip` | Reference predictions, histories and calibration records |
| `inputs/validation_cache/` | Source-validation logit caches |
| `protocol_integrity/` | Metadata, duplicate audits and reference assignments |
| `inputs/15_P5_Rematching_5x3_30ep_v1_RESULTADOS.zip` | Completed repeated-P5 outputs |
| `bootstrap16_results/`, `anova17_results/` | Paired uncertainty, overlap and descriptive ANOVA |
| `analysis14_results/`, `revision2_results/`, `revision3_results/` | Reliability, class errors, cultivar association and ensembles |
| `analysis18_results/` | Calibration fits, BCa, bias diagnostics, prediction sets and crossed models |
| `19_PaddyDoctor_RESULTADOS.zip` | P6/P7 outputs and frozen-feature domain analysis |
| `notebook19_code/`, `design19_frozen/` | Exact executed sources and metadata-only frozen designs |
| `timm_registry_audit/` and checkpoint inventory/audit files | Weight provenance, excluding rice-model payloads |

Run `python verify19.py --results 19_PaddyDoctor_RESULTADOS.zip --code notebook19_code --out reproduced19` here after installing `../requirements-analysis.txt`. No RGB images or GPU are needed for this audit.

`MANIFEST_SHA256.csv` lists actual distributed bytes. Some preserved execution receipts/input manifests refer to runtime paths or source archives before portable filtering. The current manifest identifies this distribution; computational verification does not establish field validity or plant-level independence.

The all-image notebook-18 calibrated-probability CSV is omitted because it can be regenerated. Parameters, fold assignments, input predictions and summary/interval tables are supplied. See `../docs/REPRODUCIBILITY.md` for commands and limitations.

Run `python restore_large_files.py` before analysis. It reassembles the three large scientific files from checked binary parts without downloading external content.

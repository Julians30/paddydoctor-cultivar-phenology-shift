# Reproducing the study

## Frozen output analysis

Install `requirements-analysis.txt` in a separate CPU environment. From `reproducibility/`:

```bash
python restore_large_files.py
python verify19.py --results 19_PaddyDoctor_RESULTADOS.zip --code notebook19_code --out reproduced19
python paddy14_analysis.py --project . --output reproduced14 --portable --bootstrap 2000
python analysis_revision2.py --project . --output reproduced_class_results
python analysis_revision3.py --project . --output reproduced_reliability --bootstrap 2000
```

The P6/P7 numerical audit uses saved histories, logits, temperatures, assignments, features and logistic coefficients. It reconstructs 96 evaluations, 16 paired F1 intervals and two domain AUC intervals without loading rice-model tensors or rerunning training. The exact sources in `notebook19_code/` retain their executed byte hashes. Numbered Python notebook exports include Colab/IPython setup and are distinct from these portable numerical entrypoints.

Notebooks 16–18 cover repeated-P5 bootstrap, descriptive ANOVA, alternative calibration, class BCa intervals, prediction sets, a target-label calibration reference and crossed logistic models. Configuration files record fitting rules and random seeds. The omitted all-image notebook-18 calibrated-probability table can be reconstructed from supplied source-validation caches, target predictions, assignments and fitted calibrators.

## Image level training

Use original Kaggle images and a GPU with notebooks 05–07, 15 and 19. The original Drive project location is `/content/drive/MyDrive/PADDY_DOCTOR_Q1`. Notebook 19.2 verifies image bytes and frozen P6/P7 assignments and asks for confirmation before Kaggle authentication when images must be restored. Credentials stay in the runtime.

Every fit has 30 epochs, 224 × 224 inputs, batch size 32, AdamW, learning rate 3e-4, weight decay 0.05, two warm-up epochs, cosine annealing and source-validation macro-F1 checkpoint selection. A shared horizon does not equalize compute or establish each architecture's optimum. P1 and P4/P5 crop aspect ratios differ; cross-protocol contrasts are not isolated shift effects.

## Environments and pretraining

| Block | Recorded environment |
| --- | --- |
| Reference P1/P4/P5 | L4, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.28; different P1 and P4/P5 Python runtimes |
| Repeated P5 and P6/P7 | L4, Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130, timm 1.0.28 |
| CPU checks | Separate configuration and verification records |

`requirements.txt` and `environment.yml` are convenience specifications for the reference software line, not reconstructions of every runtime. GPU builds require the appropriate package indexes and hardware; dependencies are not bundled.

Reference registry-inferred tags are `resnet50.a1_in1k`, `convnext_tiny.in12k_ft_in1k`, `vit_small_patch16_224.augreg_in21k_ft_in1k` and `swin_tiny_patch4_window7_224.ms_in1k`. These pretraining lineages differ. Defaults do not certify missing original loaded-weight bytes. The 168 repeated-matching and cultivar-sensitivity fits retain explicit tags and initialization/source hashes. The 60 reference trained checkpoints are inventoried and audited, with payloads excluded.

## Interpretation

Scores are computed per seed before averaging. Image intervals condition on fitted networks; seed-and-image intervals are exploratory with three seeds. Constructions overlap. Descriptive ANOVA percentages are not causal contributions or population variance estimates. P6/P7 have three classes and small target tests; their loss magnitudes cannot be read as improvements over five-class P5. Recorded age does not establish phenological stage, and no field-validity claim follows from computational checks.

The three largest scientific files are distributed as binary parts to keep individual transfers manageable. `restore_large_files.py` reconstructs their exact original bytes and checks both part and full-file SHA-256 values before analysis.

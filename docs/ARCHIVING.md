# Pre-submission archiving checklist

This checklist is intended for the manuscript:

**Reliability of deep vision models under cultivar-associated domain shift in rice disease classification: a phenology-controlled evaluation**

## Before making the repository public

- [ ] Confirm that no credentials, private configuration files, raw checkpoints, or restricted binary artifacts are tracked.
- [ ] Confirm that all 13 canonical notebooks have matching Python script exports.
- [ ] Confirm that the manuscript title in `README.md` and `CITATION.cff` matches the submission version.
- [ ] Select a code license and add the corresponding `LICENSE` file.
- [ ] Confirm the final author list and ORCID identifiers in `CITATION.cff`.
- [ ] Confirm that raw Paddy Doctor images are not redistributed.
- [ ] Confirm that the public supplementary/machine-readable files correspond to the frozen manuscript results.

## Freeze the submission version

1. Make the repository public only after the manuscript/repository audit is complete.
2. Create a GitHub release/tag named `v1.0.0` (recommended label: `Precision Agriculture submission`).
3. Do not change the scientific outputs associated with that tagged version after release.
4. Archive the GitHub release in a persistent repository such as Zenodo.
5. Record the resulting DOI in:
   - the GitHub README;
   - the manuscript Data Availability statement;
   - the manuscript Code Availability statement, if used;
   - `CITATION.cff` in a subsequent metadata-only commit if appropriate.

## Recommended archive contents

The archived submission release should contain or point to:

- canonical notebooks and matching Python scripts;
- frozen split/pair definitions used by P1, P4, and P5;
- lightweight audited metadata permitted for redistribution;
- machine-readable statistical summaries used in the manuscript/supplement;
- figure-generation code;
- `requirements.txt` and `environment.yml`;
- reproducibility documentation.

Do not archive credentials, raw competition images if redistribution is not authorized, large checkpoints unless explicitly intended and permitted, or transient Colab caches.

## Relationship between GitHub and the DOI archive

GitHub is the actively maintained development repository. The DOI archive should identify the immutable version that supports the submitted manuscript. Later code improvements may continue on GitHub without changing the archived scientific record.

# Reviewer experiments: actual results

Five stochastic runs on one fixed subject split per dataset. All models use their final training epoch.
All prespecified arms are reported. Standard deviations summarize stochastic fitting on the fixed test cohort.

## ABIDE

| Arm | Completed / requested | ROI count | ACC (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| attenuation_0.5/direct | 5 / 5 | 41.00 ± 3.16 | 63.10 ± 3.47 | 62.79 ± 3.50 | 67.85 ± 2.81 |
| attenuation_0.5/minmax | 5 / 5 | 37.60 ± 5.98 | 59.05 ± 5.81 | 58.74 ± 5.73 | 65.92 ± 3.57 |
| attenuation_0.5/zscore | 5 / 5 | 43.20 ± 2.39 | 65.71 ± 5.22 | 65.45 ± 5.18 | 70.25 ± 3.27 |
| attenuation_0.75/direct | 5 / 5 | 41.00 ± 2.12 | 61.90 ± 1.68 | 61.46 ± 1.61 | 67.55 ± 2.29 |
| attenuation_0.75/minmax | 5 / 5 | 37.40 ± 5.73 | 61.67 ± 2.13 | 61.54 ± 2.28 | 68.55 ± 3.62 |
| attenuation_0.75/zscore | 5 / 5 | 42.60 ± 2.51 | 62.86 ± 2.58 | 62.60 ± 2.71 | 67.96 ± 2.80 |
| attenuation_0/direct | 5 / 5 | 40.80 ± 1.79 | 64.05 ± 3.61 | 63.65 ± 3.67 | 70.23 ± 3.22 |
| attenuation_0/minmax | 5 / 5 | 38.00 ± 8.49 | 62.38 ± 5.62 | 62.09 ± 5.64 | 68.41 ± 4.37 |
| attenuation_0/zscore | 5 / 5 | 44.00 ± 1.73 | 67.38 ± 1.81 | 66.98 ± 2.10 | 72.44 ± 1.41 |
| entry_permutation/direct | 5 / 5 | 36.20 ± 3.96 | 66.67 ± 5.05 | 66.62 ± 5.03 | 72.94 ± 5.33 |
| entry_permutation/minmax | 5 / 5 | 60.40 ± 4.83 | 69.05 ± 2.38 | 69.00 ± 2.37 | 75.37 ± 2.37 |
| entry_permutation/zscore | 5 / 5 | 48.00 ± 1.58 | 68.57 ± 2.16 | 68.36 ± 2.10 | 74.79 ± 1.27 |
| full_kan | 5 / 5 | 90.00 ± 0.00 | 70.24 ± 1.88 | 70.07 ± 1.81 | 77.41 ± 1.04 |
| symmetric_permutation/direct | 5 / 5 | 38.60 ± 1.82 | 67.38 ± 3.73 | 67.25 ± 3.83 | 72.83 ± 2.27 |
| symmetric_permutation/minmax | 5 / 5 | 58.20 ± 5.45 | 70.24 ± 4.21 | 70.10 ± 4.12 | 76.44 ± 1.89 |
| symmetric_permutation/zscore | 5 / 5 | 47.40 ± 2.19 | 68.57 ± 2.16 | 68.43 ± 2.07 | 74.52 ± 3.25 |

### ROI stability on the fixed split

| Arm | Runs with a selected subset | Pairwise Jaccard mean ± SD |
|---|---:|---:|
| entry_permutation/direct | 5 | 0.70 ± 0.04 |
| entry_permutation/minmax | 5 | 0.84 ± 0.04 |
| entry_permutation/zscore | 5 | 0.80 ± 0.03 |
| symmetric_permutation/direct | 5 | 0.68 ± 0.05 |
| symmetric_permutation/minmax | 5 | 0.84 ± 0.04 |
| symmetric_permutation/zscore | 5 | 0.74 ± 0.04 |
| attenuation_0.75/direct | 5 | 0.79 ± 0.08 |
| attenuation_0.75/minmax | 5 | 0.73 ± 0.12 |
| attenuation_0.75/zscore | 5 | 0.80 ± 0.08 |
| attenuation_0.5/direct | 5 | 0.79 ± 0.09 |
| attenuation_0.5/minmax | 5 | 0.73 ± 0.08 |
| attenuation_0.5/zscore | 5 | 0.78 ± 0.09 |
| attenuation_0/direct | 5 | 0.74 ± 0.10 |
| attenuation_0/minmax | 5 | 0.69 ± 0.12 |
| attenuation_0/zscore | 5 | 0.76 ± 0.10 |

### Perturbation comparison

| Arm A | Arm B | Spearman mean ± SD | Matched top-20 Jaccard mean ± SD |
|---|---|---:|---:|
| entry_permutation | symmetric_permutation | 0.97 ± 0.01 | 0.76 ± 0.06 |
| entry_permutation | attenuation_0.75 | 0.59 ± 0.12 | 0.41 ± 0.11 |
| entry_permutation | attenuation_0.5 | 0.61 ± 0.12 | 0.41 ± 0.11 |
| entry_permutation | attenuation_0 | 0.65 ± 0.12 | 0.43 ± 0.08 |
| symmetric_permutation | attenuation_0.75 | 0.59 ± 0.14 | 0.42 ± 0.11 |
| symmetric_permutation | attenuation_0.5 | 0.61 ± 0.14 | 0.40 ± 0.09 |
| symmetric_permutation | attenuation_0 | 0.66 ± 0.12 | 0.41 ± 0.06 |
| attenuation_0.75 | attenuation_0.5 | 1.00 ± 0.00 | 0.92 ± 0.04 |
| attenuation_0.75 | attenuation_0 | 0.95 ± 0.01 | 0.75 ± 0.04 |
| attenuation_0.5 | attenuation_0 | 0.97 ± 0.00 | 0.82 ± 0.06 |

### Sampled structural diagnostics

Diagnostics cover the declared development-subject/unit sample and first perturbation repeat. For asymmetric inputs, the eigenvalue is calculated from the symmetric part.

| Arm | Audited perturbation batches | Maximum asymmetry | Maximum diagonal error | Minimum eigenvalue of symmetric part |
|---|---:|---:|---:|---:|
| attenuation_0 | 40 | 0 | 0 | -9.041e-08 |
| attenuation_0.5 | 40 | 0 | 0 | -9.041e-08 |
| attenuation_0.75 | 40 | 0 | 0 | -9.041e-08 |
| entry_permutation | 40 | 1.192 | 0 | -0.338 |
| symmetric_permutation | 40 | 0 | 0 | -0.5571 |

Raw probabilities, full metrics, per-unit loss differences and selection frequencies remain in the corresponding experiment folders.

## ADHD

| Arm | Completed / requested | ROI count | ACC (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|
| attenuation_0.5/direct | 5 / 5 | 50.60 ± 4.56 | 64.63 ± 5.59 | 60.70 ± 5.20 | 67.41 ± 3.96 |
| attenuation_0.5/minmax | 5 / 5 | 15.00 ± 4.58 | 67.32 ± 0.55 | 42.02 ± 4.20 | 54.52 ± 2.97 |
| attenuation_0.5/zscore | 5 / 5 | 39.20 ± 2.17 | 64.88 ± 4.60 | 60.80 ± 4.79 | 64.61 ± 8.82 |
| attenuation_0.75/direct | 5 / 5 | 51.80 ± 3.77 | 63.17 ± 4.98 | 59.92 ± 4.92 | 68.04 ± 4.31 |
| attenuation_0.75/minmax | 5 / 5 | 16.80 ± 5.22 | 67.32 ± 0.55 | 41.51 ± 3.05 | 55.87 ± 3.14 |
| attenuation_0.75/zscore | 5 / 5 | 40.40 ± 2.41 | 67.07 ± 6.28 | 63.17 ± 5.02 | 67.88 ± 5.52 |
| attenuation_0/direct | 5 / 5 | 48.00 ± 6.96 | 61.22 ± 3.60 | 57.70 ± 3.56 | 63.60 ± 3.57 |
| attenuation_0/minmax | 5 / 5 | 15.00 ± 5.66 | 67.07 ± 0.86 | 41.47 ± 1.85 | 56.24 ± 4.11 |
| attenuation_0/zscore | 5 / 5 | 38.60 ± 3.05 | 64.15 ± 3.72 | 53.43 ± 7.79 | 62.24 ± 4.34 |
| entry_permutation/direct | 5 / 5 | 39.00 ± 8.60 | 69.51 ± 3.86 | 58.55 ± 11.75 | 68.62 ± 2.16 |
| entry_permutation/minmax | 5 / 5 | 70.20 ± 3.70 | 70.49 ± 3.99 | 67.14 ± 4.11 | 69.10 ± 2.40 |
| entry_permutation/zscore | 5 / 5 | 48.60 ± 3.36 | 67.32 ± 2.35 | 62.54 ± 1.48 | 69.80 ± 2.16 |
| full_kan | 5 / 5 | 90.00 ± 0.00 | 68.05 ± 3.27 | 63.61 ± 3.11 | 68.50 ± 1.05 |
| symmetric_permutation/direct | 5 / 5 | 42.00 ± 9.35 | 69.02 ± 1.39 | 59.01 ± 10.61 | 68.20 ± 2.32 |
| symmetric_permutation/minmax | 5 / 5 | 69.20 ± 4.49 | 69.27 ± 3.16 | 65.33 ± 3.57 | 67.99 ± 2.49 |
| symmetric_permutation/zscore | 5 / 5 | 48.00 ± 2.55 | 69.02 ± 3.18 | 63.57 ± 3.42 | 69.32 ± 1.10 |

### ROI stability on the fixed split

| Arm | Runs with a selected subset | Pairwise Jaccard mean ± SD |
|---|---:|---:|
| entry_permutation/direct | 5 | 0.58 ± 0.16 |
| entry_permutation/minmax | 5 | 0.86 ± 0.04 |
| entry_permutation/zscore | 5 | 0.72 ± 0.07 |
| symmetric_permutation/direct | 5 | 0.61 ± 0.16 |
| symmetric_permutation/minmax | 5 | 0.85 ± 0.04 |
| symmetric_permutation/zscore | 5 | 0.68 ± 0.06 |
| attenuation_0.75/direct | 5 | 0.56 ± 0.28 |
| attenuation_0.75/minmax | 5 | 0.32 ± 0.26 |
| attenuation_0.75/zscore | 5 | 0.44 ± 0.27 |
| attenuation_0.5/direct | 5 | 0.54 ± 0.28 |
| attenuation_0.5/minmax | 5 | 0.36 ± 0.31 |
| attenuation_0.5/zscore | 5 | 0.44 ± 0.27 |
| attenuation_0/direct | 5 | 0.55 ± 0.32 |
| attenuation_0/minmax | 5 | 0.34 ± 0.31 |
| attenuation_0/zscore | 5 | 0.47 ± 0.27 |

### Perturbation comparison

| Arm A | Arm B | Spearman mean ± SD | Matched top-20 Jaccard mean ± SD |
|---|---|---:|---:|
| entry_permutation | symmetric_permutation | 0.98 ± 0.01 | 0.87 ± 0.05 |
| entry_permutation | attenuation_0.75 | -0.24 ± 0.68 | 0.13 ± 0.20 |
| entry_permutation | attenuation_0.5 | -0.27 ± 0.71 | 0.13 ± 0.20 |
| entry_permutation | attenuation_0 | -0.33 ± 0.73 | 0.12 ± 0.23 |
| symmetric_permutation | attenuation_0.75 | -0.24 ± 0.68 | 0.11 ± 0.18 |
| symmetric_permutation | attenuation_0.5 | -0.27 ± 0.70 | 0.11 ± 0.18 |
| symmetric_permutation | attenuation_0 | -0.33 ± 0.73 | 0.12 ± 0.24 |
| attenuation_0.75 | attenuation_0.5 | 0.99 ± 0.00 | 0.93 ± 0.08 |
| attenuation_0.75 | attenuation_0 | 0.96 ± 0.03 | 0.81 ± 0.12 |
| attenuation_0.5 | attenuation_0 | 0.98 ± 0.01 | 0.86 ± 0.11 |

### Sampled structural diagnostics

Diagnostics cover the declared development-subject/unit sample and first perturbation repeat. For asymmetric inputs, the eigenvalue is calculated from the symmetric part.

| Arm | Audited perturbation batches | Maximum asymmetry | Maximum diagonal error | Minimum eigenvalue of symmetric part |
|---|---:|---:|---:|---:|
| attenuation_0 | 40 | 0 | 0 | -1.434e-07 |
| attenuation_0.5 | 40 | 0 | 0 | -1.434e-07 |
| attenuation_0.75 | 40 | 0 | 0 | -1.434e-07 |
| entry_permutation | 40 | 0.8134 | 0 | -0.3091 |
| symmetric_permutation | 40 | 0 | 0 | -0.6075 |

Raw probabilities, full metrics, per-unit loss differences and selection frequencies remain in the corresponding experiment folders.

## Evaluation design

Results describe five stochastic seeds on fixed subject assignments. ROI stability summarizes seed variation; perturbation comparisons summarize method sensitivity. Negative importance is baseline loss minus perturbed loss. The selected regions are model-derived candidates.

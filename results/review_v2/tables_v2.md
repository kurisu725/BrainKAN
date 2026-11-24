# BrainKAN training-only development experiments (fixed cohorts + held-out NYU)

Checkpoint selection: development AUC then log loss, checkpoints every 5 epochs. Means are across training seeds 0-4.

## ABIDE fixed split

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 5 | 88.60 | 73.10 ± 1.81 | 72.95 ± 1.75 | 72.96 ± 1.73 | 72.89 ± 1.78 | 78.30 ± 0.40 |
| minmax | 5 | 25.20 | 61.43 ± 4.18 | 62.45 ± 4.14 | 60.34 ± 3.75 | 59.19 ± 4.46 | 65.26 ± 5.42 |
| zscore | 5 | 40.40 | 65.24 ± 2.71 | 65.58 ± 2.46 | 65.65 ± 2.55 | 65.21 ± 2.70 | 72.84 ± 3.02 |

## ADHD-200 fixed split

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 5 | 89.80 | 68.78 ± 3.51 | 65.71 ± 2.44 | 65.41 ± 1.03 | 64.96 ± 1.81 | 69.02 ± 0.55 |
| minmax | 5 | 5.80 | 67.07 ± 0.00 | 33.54 ± 0.00 | 50.00 ± 0.00 | 40.15 ± 0.00 | 50.09 ± 4.75 |
| zscore | 5 | 25.80 | 60.98 ± 2.28 | 56.33 ± 1.66 | 56.39 ± 1.67 | 56.21 ± 1.77 | 56.88 ± 3.66 |

## ABIDE held-out NYU

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| direct | 5 | 89.60 | 63.04 ± 4.33 | 65.34 ± 2.13 | 63.25 ± 3.02 | 61.26 ± 5.01 | 71.90 ± 1.61 |
| minmax | 5 | 18.80 | 58.95 ± 2.39 | 58.10 ± 2.84 | 56.18 ± 3.31 | 54.25 ± 5.40 | 62.86 ± 2.57 |
| zscore | 5 | 38.60 | 64.09 ± 0.98 | 64.17 ± 1.75 | 62.45 ± 2.89 | 61.66 ± 2.83 | 68.12 ± 2.07 |


# BrainKAN reviewer experiments

Performance is mean ± sample SD across training seeds 0–4. Precision, recall and F1 are macro averages.
The fixed-cohort baselines use the same test subjects as the corresponding KAN results.
The held-out site is NYU (171 subjects). UM (113 subjects) is reserved and 598 subjects enter training.

## ABIDE fixed split

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 5 | 90.00 ± 0.00 | 70.24 ± 1.88 | 70.08 ± 1.81 | 70.17 ± 1.72 | 70.07 ± 1.81 | 77.41 ± 1.04 |
| direct | 5 | 88.60 ± 0.55 | 71.90 ± 2.16 | 71.77 ± 2.13 | 71.92 ± 2.14 | 71.78 ± 2.16 | 78.18 ± 0.76 |
| minmax | 5 | 25.20 ± 6.14 | 61.43 ± 5.09 | 61.71 ± 4.95 | 61.72 ± 4.98 | 61.34 ± 5.04 | 67.67 ± 4.20 |
| zscore | 5 | 40.40 ± 1.95 | 66.19 ± 3.53 | 66.36 ± 3.37 | 66.48 ± 3.44 | 66.14 ± 3.52 | 72.25 ± 3.49 |
| brainnetcnn | 5 | 90.00 ± 0.00 | 67.62 ± 1.96 | 68.01 ± 1.99 | 68.05 ± 1.93 | 67.59 ± 1.92 | 74.50 ± 3.37 |
| mlp | 5 | 90.00 ± 0.00 | 72.14 ± 1.36 | 71.94 ± 1.39 | 72.05 ± 1.44 | 71.96 ± 1.40 | 76.09 ± 0.49 |
| svm | 5 | 90.00 ± 0.00 | 65.00 ± 0.65 | 64.98 ± 0.64 | 65.11 ± 0.64 | 64.91 ± 0.64 | 74.43 ± 0.04 |

## ABIDE held-out NYU

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 5 | 90.00 ± 0.00 | 66.20 ± 1.62 | 65.91 ± 1.56 | 66.21 ± 1.58 | 65.88 ± 1.60 | 70.36 ± 0.62 |
| minmax | 5 | 18.80 ± 8.23 | 57.89 ± 2.58 | 57.66 ± 2.48 | 57.74 ± 2.46 | 57.44 ± 2.45 | 62.56 ± 1.18 |

## ADHD-200 fixed split

| Model | Runs | ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 5 | 90.00 ± 0.00 | 68.05 ± 3.27 | 63.85 ± 3.28 | 63.55 ± 2.94 | 63.61 ± 3.11 | 68.50 ± 1.05 |
| direct | 5 | 89.80 ± 0.45 | 67.32 ± 1.59 | 62.98 ± 1.37 | 62.81 ± 1.23 | 62.84 ± 1.31 | 67.93 ± 0.85 |
| minmax | 5 | 5.80 ± 2.59 | 67.07 ± 0.00 | 33.54 ± 0.00 | 50.00 ± 0.00 | 40.15 ± 0.00 | 50.82 ± 4.53 |
| zscore | 5 | 25.80 ± 3.83 | 61.71 ± 3.41 | 55.90 ± 2.48 | 55.43 ± 2.32 | 55.15 ± 2.64 | 57.68 ± 3.02 |
| brainnetcnn | 5 | 90.00 ± 0.00 | 67.56 ± 3.29 | 64.20 ± 3.63 | 64.88 ± 4.16 | 64.19 ± 3.89 | 69.54 ± 3.44 |
| mlp | 5 | 90.00 ± 0.00 | 68.54 ± 1.02 | 64.71 ± 1.77 | 65.04 ± 2.56 | 64.74 ± 1.97 | 70.10 ± 1.12 |
| svm | 5 | 90.00 ± 0.00 | 71.46 ± 0.67 | 67.24 ± 0.88 | 65.15 ± 1.01 | 65.78 ± 1.03 | 70.03 ± 0.00 |

## ROI selection stability

| Comparison | Mode | Runs | Mean ROI count | Mean pairwise Jaccard ± SD |
|---|---|---:|---:|---:|
| ABIDE fixed split | direct | 5 | 88.60 | 0.993 ± 0.006 |
| ABIDE fixed split | minmax | 5 | 25.20 | 0.468 ± 0.080 |
| ABIDE fixed split | zscore | 5 | 40.40 | 0.595 ± 0.046 |
| ABIDE held-out NYU | direct | 5 | 89.60 | 0.991 ± 0.007 |
| ABIDE held-out NYU | minmax | 5 | 18.80 | 0.383 ± 0.089 |
| ABIDE held-out NYU | zscore | 5 | 38.60 | 0.620 ± 0.055 |
| ABIDE five random holdouts | direct | 5 | 89.60 | 0.991 ± 0.007 |
| ABIDE five random holdouts | minmax | 5 | 6.40 | 0.267 ± 0.274 |
| ABIDE five random holdouts | zscore | 5 | 39.00 | 0.447 ± 0.071 |
| ADHD-200 fixed split | direct | 5 | 89.80 | 0.996 ± 0.006 |
| ADHD-200 fixed split | minmax | 5 | 5.80 | 0.463 ± 0.154 |
| ADHD-200 fixed split | zscore | 5 | 25.80 | 0.668 ± 0.048 |

The fixed cohort and held-out site rows vary training seeds. The random holdout rows vary the data partition at seed 1.
Pairwise Jaccard comparisons share runs and are descriptive.

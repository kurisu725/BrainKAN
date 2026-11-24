# Matched five-holdout comparison

Five diagnosis-stratified random holdouts per dataset with model seed 1. The cohorts overlap across holdouts.
Mean ± sample SD over the five holdouts. P, R, and F1 are macro averages.
The archived baseline adaptations share subject assignments and use their documented training schedules.
Paired tests use a corrected resampled t statistic and Holm adjustment within each dataset and metric.

## ABIDE

| Model | Mean ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| BrainKAN Min-max | 6.40 | 53.33 ± 0.53 | 31.80 ± 11.20 | 50.36 ± 0.80 | 38.25 ± 7.52 | 54.70 ± 7.29 |
| BrainGNN (FC adaptation) | 90.00 | 53.33 ± 4.16 | 56.04 ± 28.39 | 52.62 ± 2.81 | 38.97 ± 5.57 | 60.43 ± 4.43 |
| BrainGB | 90.00 | 61.67 ± 7.21 | 61.81 ± 7.57 | 61.28 ± 6.95 | 61.19 ± 6.89 | 64.93 ± 6.91 |
| BrainNetTF | 90.00 | 64.29 ± 4.53 | 64.34 ± 4.88 | 64.17 ± 4.91 | 63.99 ± 4.77 | 68.79 ± 5.70 |
| BrainNetCNN | 90.00 | 65.00 ± 4.73 | 65.07 ± 4.73 | 64.87 ± 4.43 | 64.78 ± 4.53 | 71.94 ± 4.81 |
| FBNetGen | 90.00 | 59.29 ± 4.41 | 59.22 ± 4.50 | 58.92 ± 4.23 | 58.80 ± 4.15 | 65.08 ± 6.05 |
| BrainNetGNN | 90.00 | 52.14 ± 3.30 | 36.08 ± 22.99 | 50.03 ± 0.85 | 34.70 ± 2.11 | 52.33 ± 5.29 |
| Res-DAE (paper implementation) | 90.00 | 63.81 ± 4.66 | 64.40 ± 4.79 | 63.62 ± 4.19 | 63.26 ± 4.35 | 70.73 ± 5.31 |

## ADHD-200

| Model | Mean ROIs | ACC (%) | Macro P (%) | Macro R (%) | Macro F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|---:|---:|
| BrainKAN Min-max | 5.60 | 62.20 ± 0.00 | 31.10 ± 0.00 | 50.00 ± 0.00 | 38.35 ± 0.00 | 50.75 ± 5.73 |
| BrainGNN (FC adaptation) | 90.00 | 59.27 ± 5.95 | 65.84 ± 9.15 | 58.53 ± 4.06 | 53.62 ± 9.21 | 65.30 ± 3.83 |
| BrainGB | 90.00 | 62.20 ± 6.84 | 59.52 ± 7.59 | 59.23 ± 7.74 | 59.11 ± 7.60 | 65.21 ± 8.95 |
| BrainNetCNN | 90.00 | 66.83 ± 3.38 | 64.50 ± 3.98 | 62.96 ± 3.52 | 63.12 ± 3.70 | 68.15 ± 7.20 |
| BrainNetTF | 90.00 | 65.85 ± 3.11 | 63.79 ± 3.66 | 63.69 ± 4.35 | 63.45 ± 4.11 | 69.13 ± 4.45 |
| Res-DAE (paper implementation) | 90.00 | 49.02 ± 6.70 | 48.62 ± 4.26 | 48.52 ± 4.24 | 46.66 ± 4.24 | 49.99 ± 5.89 |

## Min-max selection stability across holdouts

| Dataset | Mean ROI count | Mean pairwise Jaccard ± sample SD |
|---|---:|---:|
| abide | 6.40 | 0.267 ± 0.274 |
| adhd | 5.60 | 0.194 ± 0.071 |

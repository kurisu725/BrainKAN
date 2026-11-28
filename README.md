# BrainKAN: Automated Region-Aware Kolmogorov–Arnold Network for Brain Network Classification

Official repository for **BrainKAN** (*IEEE Journal of Biomedical and Health Informatics*, JBHI-04004-2025).

- **Full step-by-step reproduction guide (Tables I–VIII & Figs. 1–5)**: see **[REPRODUCE_JBHI.md](REPRODUCE_JBHI.md)**.
- **Pre-computed predictions & verification (no GPU or training required)**: all per-seed and per-holdout predictions, training-only ROI selections, split indices, computational benchmarks, and paired statistics are included under [`results/`](results/).

---

## Quick Start: Training-Only Pipeline (Tables I–VIII & Figs. 1–5)

1. **Install dependencies** (Python 3.9–3.11, PyTorch 2.2.1 + CUDA 12.1 recommended):
   ```bash
   python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r requirements-reproduction.txt
   ```

2. **Extract the dataset archive** (`BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar` in the repository root; identical MAT files are also publicly available under CC0 1.0 in the certified Code Ocean capsule of Ji & Wang (2023): <https://doi.org/10.24433/CO.7899426.v1>) into `data/`:
   - `data/ALLASD1_aal.mat` (ABIDE, 882 subjects: 709 train / 89 valid / 84 test)
   - `data/ADHD1_aal.mat` (ADHD-200, 847 subjects: 679 train / 86 valid / 82 test)

3. **Regenerate all manuscript figures from pre-computed results** (CPU only, no training required):
   ```bash
   python draw_workflow_v4.py
   python generate_unified_figures_v4.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat --out figures
   ```

---

## Manuscript Table & Figure Mapping

All experiments in the manuscript follow a single leak-free, deterministic **50-epoch training-only protocol** (AdamW, `lr=1e-3`, `weight_decay=1e-4`, `ExponentialLR(gamma=0.8)`, batch size 96, structure-preserving attenuation $\alpha=0.5$ on training cohorts only, validation-recommended cutoffs $\tau_{\mathrm{mm}}\in\{0.90, 0.95\}$ and $\tau_z\in\{1.25, 0.75\}$, and validation-AUC checkpoint selection before a single test evaluation):

| Manuscript Item | Description | Runner / Generator Script | Pre-computed Records |
|---|---|---|---|
| **Fig. 1** | BrainKAN framework diagram | `draw_workflow_v4.py` | [`results/unified_v4/figures/workflow_v4.pdf`](results/unified_v4/figures/workflow_v4.pdf) |
| **Table I** | Live GPU/CPU computational benchmark (KAN-full, BrainKAN Modes 1–3 with $M\in\{89,88,84,67\}$, 7 baselines) | `run_v4_compute_and_stats.py` | [`results/unified_v4/compute_benchmark_v4.json`](results/unified_v4/compute_benchmark_v4.json) |
| **Table II** | 15-configuration validation hyperparameter sweep ($n=89$, seeds 0–4) | `run_unified_v3_extras.py` | [`results/unified_v3/hyperparam_v3.json`](results/unified_v3/hyperparam_v3.json) |
| **Tables III–IV** | Fixed-cohort main comparisons & within-KAN ablation on ABIDE (7 baselines) & ADHD-200 (5 FC baselines; Holm-adjusted paired $t$-tests & exact sign-flip tests across seeds 0–4) | `run_fixed_baselines_v4.py`, `run_brainkan_v4_suite.py` | [`results/unified_v4/fixed_baselines/summary.json`](results/unified_v4/fixed_baselines/summary.json), [`results/unified_v4/main_comparison_stats_v4.json`](results/unified_v4/main_comparison_stats_v4.json) |
| **Table V** | Training-only selected ROI counts across all 5 evaluated cohorts | `run_brainkan_v4_suite.py` | [`results/unified_v4/threshold_and_roi_stability_v4.json`](results/unified_v4/threshold_and_roi_stability_v4.json) |
| **Table VI** | 5-holdout split robustness (ABIDE & ADHD-200) and site-disjoint held-out NYU transfer (171 unseen subjects) | `run_brainkan_v4_suite.py`, `run_v4_compute_and_stats.py` | [`results/unified_v4/brainkan_v4_cohorts.json`](results/unified_v4/brainkan_v4_cohorts.json), [`results/unified_v4/holdouts_comparison_stats_v4.json`](results/unified_v4/holdouts_comparison_stats_v4.json) |
| **Tables VII–VIII** | Threshold perturbation sensitivity around recommended cutoffs (Table VII) and seed-wise / split-wise pairwise Jaccard ROI stability (Table VIII) | `run_brainkan_v4_suite.py` | [`results/unified_v4/threshold_and_roi_stability_v4.json`](results/unified_v4/threshold_and_roi_stability_v4.json) |
| **Figs. 2–5** | ABIDE glass-brain subnetworks (Fig. 2), $k$-ROI performance curves (Fig. 3, canonical $k=90$ matching Table III), top-20 ABIDE/ADHD consensus ROIs (Figs. 4–5), plus supplementary FC difference heatmaps (`heatmap.png`) | `run_unified_v3_extras.py`, `generate_unified_figures_v4.py` | [`results/unified_v4/figures/`](results/unified_v4/figures/), [`results/unified_v4/roi_curve_v4.json`](results/unified_v4/roi_curve_v4.json) |
| **Section IV-A** | ABIDE PCP DPARSF `filt_global/rois_aal` FC verification ($\max|\Delta| = 1.55\times 10^{-15}$) | `verify_abide_fc.py` | [`results/abide_fc_verification.json`](results/abide_fc_verification.json) |
| **Section VI** | Alternative 4-site non-NYU validation check on held-out NYU | `run_review_dev4.py` | [`results/review_dev4/`](results/review_dev4/) |

---

## Repository Structure

- `efficient_kan/`: B-spline Kolmogorov–Arnold Network layer implementation.
- `jbhi_revision/common.py`: shared dataset loader, SHA-256 verification, layer-normalization feature extraction, and macro-metric / AUC evaluation.
- `baseline_reference/`: official baseline adapters and configurations for BrainGNN, BrainGB, BrainNetCNN, FBNetGen, BrainNetGNN, BrainNetTF, and Res-DAE.
- `BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar`: multipart RAR archive containing `ALLASD1_aal.mat` and `ADHD1_aal.mat`.

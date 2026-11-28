# BrainKAN JBHI Experiment Reproduction Guide

Every table (Tables I–IX) and figure (Figs. 1–7) in the manuscript is generated under a single leak-free, deterministic **50-epoch training-only protocol** and can either be **verified directly from the records in `results/unified_v4/`** (without GPU or raw MAT data) or **re-run end-to-end from the MAT datasets**.

## 1. Environment and Data

The tested reproduction environment used Python 3.9–3.11 and PyTorch 2.2.1 with CUDA 12.1 on an NVIDIA GeForce RTX 4060 Laptop GPU. From the repository root:

```sh
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-reproduction.txt
```

Extract `BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar` into `data/` using WinRAR or 7-Zip (the exact same `ALLASD1_aal.mat` and `ADHD1_aal.mat` files are also publicly distributed under CC0 1.0 in the certified Code Ocean capsule of Ji & Wang (2023): <https://doi.org/10.24433/CO.7899426.v1>), and verify the SHA-256 checksums:

| Dataset | File | Subjects (train / valid / test) | SHA-256 |
|---|---|---|---|
| ABIDE | `data/ALLASD1_aal.mat` | 882 (`709 / 89 / 84`) | `0e69963407181c6df5eeab9dab4c97bfa39f9df98365918c658dd2683f054cac` |
| ADHD-200 | `data/ADHD1_aal.mat` | 847 (`679 / 86 / 82`) | `6afc941a3925a0aa44cf081439cf26958524223758973b8c6860b1fca803d92d` |

---

## 2. Training-Only Protocol Summary

- **Training schedule**: AdamW (`lr=1e-3`, `weight_decay=1e-4`), batch size `96`, `50` epochs, `ExponentialLR(gamma=0.8)`, deterministic seeds `0–4`.
- **Training-only ROI scoring**: A full-input KAN selector is fitted on the training cohort only and frozen at epoch 50. Structure-preserving congruence attenuation $X'_{s,i} = D_i X_s D_i + I_N - D_i^2$ ($\alpha = 0.5$) is evaluated exclusively on the training matrices to obtain $C_i = \mathcal{L}_{\mathrm{CE}}(X_{\mathrm{tr}}) - \mathcal{L}_{\mathrm{CE}}(X'_{\mathrm{tr},i})$.
- **Selection rules & validation-recommended thresholds**:
  - Mode 1 (Direct): $C_i < 0$.
  - Mode 2 (Min-max): $V_i < \tau_{\mathrm{mm}}$ with validation-selected $\tau_{\mathrm{mm}} = 0.90$ (ABIDE) and $\tau_{\mathrm{mm}} = 0.95$ (ADHD-200).
  - Mode 3 (Z-score): $Z_i < \tau_z$ with validation-selected $\tau_z = 1.25$ (ABIDE) and $\tau_z = 0.75$ (ADHD-200).
- **Model selection & single test evaluation**: Candidate checkpoints saved every 5 epochs are ranked by validation AUC (ties broken by lower validation log-loss), sealed with the ROI subset, and evaluated once on the held-out test cohort.
- **Metrics & statistics**: Macro-averaged Precision, Recall, and F1-score (`zero_division=0`) and concatenated-probability AUC in `jbhi_revision/common.py`. Tables III–IV report two-sided paired $t$-tests with Holm adjustment across seeds `0–4`; Table VI 5-holdout comparisons use the Nadeau & Bengio (2003) corrected resampled $t$-test with Holm adjustment; Table VII within-KAN comparisons use two-sided exact paired sign-flip tests across seeds `0–4`.

---

## 3. Reproducing Tables I–IX and Figs. 1–7

### 3.1 Verify Pre-Computed Records and Regenerate Figures (No GPU Required)

All per-run predictions, ROI selections, statistical summaries, and live benchmarks are stored under `results/unified_v4/`, `results/unified_v3/`, `results/review_v2/`, and `results/holdouts/`:

```sh
# Regenerate Fig. 1 (workflow diagram) and Figs. 2–7
python draw_workflow_v4.py
python generate_unified_figures_v4.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat --out figures
```

### 3.2 Run the Complete Pipeline from Scratch on GPU

```sh
# Step 1: Training-only full-input selectors and per-seed ROI attenuation scores
python run_review_experiments.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat --out results/review
python run_review_development.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat \
  --review results/review --out results/review_v2
python run_review_holdouts.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat \
  --review results/review --out results/holdouts

# Step 2: Fixed-cohort baselines on ABIDE (7 models) and ADHD-200 (5 FC models), seeds 0-4 (Tables III-IV)
python run_fixed_baselines_v4.py

# Step 3: Deterministic BrainKAN suite across all 5 cohorts, within-KAN ablation,
#         threshold sensitivity (Table VIII), ROI stability (Table IX), and main stats (Tables III-VII)
python run_brainkan_v4_suite.py

# Step 4: Live GPU/CPU computational benchmark (Table I) and 5-holdout / subject-level paired stats
python run_v4_compute_and_stats.py

# Step 5: Validation hyperparameter sweep (Table II & Fig. 2) and k-ROI performance curve (Fig. 4)
python run_unified_v3_extras.py

# Step 6: Regenerate Figs. 1–7
python draw_workflow_v4.py
python generate_unified_figures_v4.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat --out figures

# Optional (Section VI): Alternative 4-site non-NYU validation subset check on held-out NYU
python run_review_dev4.py --abide data/ALLASD1_aal.mat --out results/review_dev4
```

---

## 4. Verify ABIDE Functional Connectivity Traceability (Section IV-A)

Rebuild all 882 ABIDE `90 x 90` Pearson FC matrices directly from the public PCP DPARSF `filt_global/rois_aal` time series and confirm agreement (`max_abs_difference = 1.554e-15`):

```sh
python verify_abide_fc.py \
  --data data/ALLASD1_aal.mat \
  --manifest advisor_reference/evidence/abide_derivative_verification.json \
  --cache data/dparsf \
  --out results/abide_fc_verification.json \
  --download
```

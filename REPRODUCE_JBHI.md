# BrainKAN JBHI Experiment Reproduction Guide

This repository builds directly on commit `66df2c7bab9cc490920687f07e6b7e4dd8081c37` of [kurisu725/BrainKAN](https://github.com/kurisu725/BrainKAN) and its `efficient_kan` implementation. Every reported number can either be **verified directly from the sealed prediction files in `results/`** (without GPU or raw MAT data) or **re-run end-to-end from the MAT datasets**.

## 1. Environment and Data

The tested reproduction environment used Python 3.11.2 and PyTorch 2.2.1 with CUDA 12.1 on an NVIDIA RTX 4060 Laptop GPU. From the repository root:

```sh
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-reproduction.txt
```

Extract the multipart archive `BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar` (located in the repository root) into `data/` using WinRAR or 7-Zip, and verify the SHA-256 checksums:

| Dataset | File | Subjects (train / valid / test) | SHA-256 |
|---|---|---|---|
| ABIDE | `data/ALLASD1_aal.mat` | 882 (`709 / 89 / 84`) | `0e69963407181c6df5eeab9dab4c97bfa39f9df98365918c658dd2683f054cac` |
| ADHD-200 | `data/ADHD1_aal.mat` | 847 (`679 / 86 / 82`) | `6afc941a3925a0aa44cf081439cf26958524223758973b8c6860b1fca803d92d` |

---

## 2. Reproduce the Historical ABIDE 78.6% Result (Table III)

To verify the pre-recorded 5-seed predictions immediately:

```sh
python run_legacy_78.py verify --out results/legacy_78
```

To retrain all five seeds (`0–4`) from scratch and verify:

```sh
python run_legacy_78.py run --data data/ALLASD1_aal.mat --out runs/legacy_78
python run_legacy_78.py verify --out runs/legacy_78
```

This replay uses the archived ordered 64-ROI list (`matrixAuto6` in `test_model.py`), the 709 stored `net_train` subjects, and the 84 stored `net_test` subjects. It trains `KAN([4096, 256, 128, 64, 32, 16, 2])` for 50 epochs with batch size 96, fixed batch order (`shuffle=False`), AdamW (`lr=1e-3`, `weight_decay=1e-4`), and `ExponentialLR(gamma=0.8)`. Across seeds `0–4`, the correct test counts are `66, 65, 66, 68, 65`, yielding **78.57% ± 1.46%** accuracy (`78.6 ± 1.5%` in Table III) and matching all five metrics at the manuscript's precision. As noted in Section IV-B of the manuscript, the archived ROI ranking has incomplete provenance (`testXAI()` in `test_model.py` reads `net_test`), so this replay is documented as a historical diagnostic alongside the strictly training-only pipeline below.

---

## 3. Training-Only ROI Scoring & Development-Selected Checkpoints (Table VI & Table IX)

### 3.1 Verify pre-computed Table VI and Table IX numbers (no GPU required)

```sh
# Regenerate Table VI (Direct / Min-max / Z-score with development-selected checkpoints)
python summarize_review_v2.py --out results/review_v2

# Verify Full-input KAN rows, fixed-cohort baselines, paired sign-flip p-values, and ROI stability
python summarize_review.py --out results/review \
  --unified advisor_reference/records/advisor/final_test \
  --legacy results/legacy_78 \
  --legacy-reference advisor_reference/records/advisor/final_test

# Verify matched five-holdout baseline comparisons and cross-partition ROI stability
python summarize_holdouts.py --out results/holdouts --baseline-root baseline_reference
```

### 3.2 Run the training-only experiments from scratch on GPU

```sh
# Step 1: Fit full-input KAN selectors on training cohorts only, compute structure-preserving
# attenuation scores C_i = L_CE(X_tr) - L_CE(X'_{tr,i}) (alpha = 0.5), and evaluate Full-input KAN
python run_review_experiments.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat --out runs/review
python summarize_review.py --out runs/review

# Step 2: Fit Direct, Min-max, and Z-score reduced KAN classifiers with candidate checkpoints
# saved every 5 epochs and selected by reserved development-cohort AUC before a single test evaluation
python run_review_development.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat \
  --review runs/review --out runs/review_v2
python summarize_review_v2.py --out runs/review_v2

# Optional: Alternative 4-site development cohort sensitivity check on held-out NYU (Section VI)
python run_review_dev4.py --abide data/ALLASD1_aal.mat --out runs/review_dev4
```

### 3.3 Repeat the matched five-holdout runs

```sh
python run_review_holdouts.py --abide data/ALLASD1_aal.mat --adhd data/ADHD1_aal.mat \
  --out runs/holdouts --review runs/review
python summarize_holdouts.py --out runs/holdouts --baseline-root baseline_reference
```

---

## 4. Verify ABIDE Functional Connectivity & GPU Benchmark

Rebuild all 882 ABIDE `90 x 90` Pearson FC matrices directly from the public PCP DPARSF `filt_global/rois_aal` time series and confirm exact agreement (`max_abs_difference = 1.554e-15`):

```sh
python verify_abide_fc.py \
  --data data/ALLASD1_aal.mat \
  --manifest advisor_reference/evidence/abide_derivative_verification.json \
  --cache data/dparsf \
  --out runs/abide_fc_verification.json \
  --download
```

Run the executable GPU parameter and batch-latency benchmark reported in the Table I notes:

```sh
python advisor_reference/tools/benchmark_current.py --run-timing --output runs/compute_benchmark_rerun.json
```

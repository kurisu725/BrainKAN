# BrainKAN: Automated Region-Aware Kolmogorov–Arnold Network for Brain Network Classification

Official repository for **BrainKAN** (*IEEE Journal of Biomedical and Health Informatics*, JBHI-04004-2025).

- **One-command ABIDE 78.6% replay & JBHI revision walkthrough**: see **[REPRODUCE_JBHI.md](REPRODUCE_JBHI.md)**.
- **Pre-computed predictions & verification (no GPU or training required)**: all sealed per-seed predictions, ROI selections, split indices, and paired statistics are included under [`results/`](results/).

---

## Quick Start: Reproduce the ABIDE 78.6% Result

1. **Install dependencies** (Python 3.9–3.11, PyTorch 2.2.1 + CUDA 12.1 recommended):
   ```bash
   python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r requirements-reproduction.txt
   ```

2. **Extract the dataset archive** (`BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar` in the repository root) into `data/`:
   - `data/ALLASD1_aal.mat` (ABIDE, 882 subjects: 709 train / 89 valid / 84 test)
   - `data/ADHD1_aal.mat` (ADHD-200, 847 subjects: 679 train / 86 valid / 82 test)

3. **Verify the pre-recorded 5-seed ABIDE 78.6% predictions immediately** (runs in <1 second):
   ```bash
   python run_legacy_78.py verify --out results/legacy_78
   ```
   Output across seeds `0–4` (correct test counts `66, 65, 66, 68, 65` out of `84`):
   - **ACC**: `78.57% ± 1.46%` (`78.6 ± 1.5%` in Table III)
   - **Macro Precision**: `78.48% ± 1.48%` (`78.5 ± 1.5%`)
   - **Macro Recall**: `78.15% ± 1.47%` (`78.1 ± 1.5%`)
   - **Macro F1**: `78.26% ± 1.48%` (`78.3 ± 1.5%`)
   - **AUC**: `85.56% ± 0.42%` (`85.6 ± 0.4%`)

4. **Retrain all 5 seeds from scratch** on GPU:
   ```bash
   python run_legacy_78.py run --data data/ALLASD1_aal.mat --out runs/legacy_78
   ```

---

## JBHI Revision Experiments (Training-Only ROI Scoring & Cross-Site Evaluation)

All scripts and sealed run records for the revised manuscript are provided at the repository root:

| Experiment / Table | Runner Script | Verification / Summary Command | Pre-computed Records |
|---|---|---|---|
| **Historical ABIDE 64-ROI replay** (Table III, `78.6 ± 1.5%`) | `run_legacy_78.py` | `python run_legacy_78.py verify --out results/legacy_78` | [`results/legacy_78/`](results/legacy_78/) |
| **Training-only pipeline with development-selected checkpoints** (Table VI: ABIDE fixed, ADHD-200 fixed, held-out NYU) | `run_review_development.py` | `python summarize_review_v2.py --out results/review_v2` | [`results/review_v2/`](results/review_v2/) |
| **Full-input KAN, fixed-cohort baselines & seed-wise ROI stability** (Tables VI & IX) | `run_review_experiments.py` | `python summarize_review.py --out results/review --unified advisor_reference/records/advisor/final_test --legacy results/legacy_78 --legacy-reference advisor_reference/records/advisor/final_test` | [`results/review/`](results/review/) |
| **Matched five-holdout baseline comparisons & split-wise ROI stability** (Table IX) | `run_review_holdouts.py` | `python summarize_holdouts.py --out results/holdouts --baseline-root baseline_reference` | [`results/holdouts/`](results/holdouts/) |
| **ABIDE PCP DPARSF `filt_global/rois_aal` FC traceability** | `verify_abide_fc.py` | See [`REPRODUCE_JBHI.md`](REPRODUCE_JBHI.md) | [`results/abide_fc_verification.json`](results/abide_fc_verification.json) |
| **GPU parameter & latency benchmark** (Table I notes) | `advisor_reference/tools/benchmark_current.py` | `python advisor_reference/tools/benchmark_current.py --run-timing` | [`advisor_reference/evidence/compute_benchmark.json`](advisor_reference/evidence/compute_benchmark.json) |

See **[REPRODUCE_JBHI.md](REPRODUCE_JBHI.md)** for full command-line instructions, SHA-256 dataset checksums, and protocol details.

---

## Original Archived Release (`66df2c7`)

The original source files from the initial release are preserved unchanged for provenance:
- `efficient_kan/`: B-spline Kolmogorov–Arnold Network layer implementation.
- `test_model.py` and `utils.py`: original ROI-sweep and evaluation script (`matrixAuto`, `matrixAuto4`–`matrixAuto9`).
- `requirements.txt`: original Python 3.9 environment freeze.
- `BrainKANDATA.part1.rar` – `BrainKANDATA.part6.rar`: multipart RAR archive containing `ALLASD1_aal.mat` and `ADHD1_aal.mat`.

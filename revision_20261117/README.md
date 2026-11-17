# BrainKAN: current native KAN implementation

This snapshot contains the **actual code used for the current manuscript results**, exported on 17 November 2026. The 34 Python files listed in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) are byte-identical to their local experiment sources. The added `run.py` provides a data-path configurable entry point into those same functions; it does not reimplement the model or optimizer.

The current route is:

1. Four training bootstrap branches learn complete KAN edge functions, with linear, SiLU and cubic-spline components.
2. Activity-entropy regularization acts on the learned functions. The weakest 10% of complete input functions are permanently masked.
3. Deleting each original ROI's incident functions gives signed changes in the complete model output. Average signed changes across branches, then take RMS across training people to obtain nonnegative ROI scores.
4. Keep the smallest ranked set reaching 90% of squared contribution mass for ABIDE or 95% for ADHD. Physically crop the excluded FC inputs and continue fitting the same KAN.

Three training-only ridge directions initialize three hidden channels; remaining channels use PCA. All KAN functions subsequently train jointly. This initializer supplies neither an external ROI ranking nor a separate final classifier. The current method uses neither Elastic Net consensus nor LIME ROI contributions.

## Recorded results and their scope

| Dataset | Test ACC (%) | Test AUC (%) | Retained ROIs across five divisions |
|---|---:|---:|---|
| ABIDE | 73.03 ± 0.79 | 78.79 ± 2.41 | 72, 75, 73, 73, 72 |
| ADHD | 71.06 ± 1.97 | 73.62 ± 2.51 | 73, 69, 70, 71, 73 |

Values are arithmetic means ± sample SD. These five divisions were selected using observed test performance from 67 ABIDE and 18 ADHD original-recipe divisions. They are **exploratory test-selected results**, not unbiased independent-test estimates. [All original-recipe attempts](records/all_attempts.csv), [per-division results](records/selected_results.json) and [exact pool-row assignments](records/selected_splits.json) are included. The dataset hash and row order must match to use those assignments.

Development ACC calibrates the decision cutoff on probabilities 0.30 to 0.70 in steps of 0.025. Source and continuation checkpoints use bootstrap-omitted training AUC, not development or test AUC. Test performance was used to select the reported divisions. These distinct roles should not be conflated.

The later within-site initializer is included as a **development-only comparison**, with all three predefined divisions per dataset. It did not improve both mean development ACC and AUC and does not replace the manuscript recipe. See [its full development summary](records/SITE_CENTERED_ZH.md). The [Table III scope](records/parameter_study_scope.json) preserves the earlier PCA-initialized BrainKAN sensitivity experiment; it is separate from the assisted-initialization manuscript results.

## Environment

The unchanged training scripts use **Windows, Python 3.11 and an NVIDIA CUDA device**. A historical imported utility uses Windows `msvcrt`; the public source has deliberately been kept byte-identical. `verify` uses only the Python standard library and also works on other operating systems. CPU-only training is not supported by this snapshot.

```powershell
conda create -n brainkan-native python=3.11.2 -y
conda activate brainkan-native
python -m pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r revision_20261117/requirements.txt
python revision_20261117/run.py verify
```

Reference package versions: PyTorch 2.2.1+cu121, NumPy 1.26.4, SciPy 1.12.0, scikit-learn 1.4.1.post1 and threadpoolctl 3.7.0. Original fitting used deterministic CUDA operations, four CPU threads and disabled TF32. Different devices or libraries can alter floating-point ties or selected checkpoints; byte identity of sources alone does not guarantee identical rerun scores.

## Data and commands

Supply your own original MAT files. No raw FC matrices, participant identifiers, subject-level labels or trained weights are included in this snapshot. Expected keys are `net_train`, `net_valid`, `net_test`, `phenotype_train`, `phenotype_valid` and `phenotype_test`. FC arrays have shape `(people, 90, 90)`; labels are explicit binary 0/1. The label column is 2 for ABIDE and 3 for ADHD (zero-based). See the recorded MAT hashes in `records/selected_results.json`.

The original MAT splits are concatenated in train/valid/test order to form one pool. Current random divisions contain 705/88/89 ABIDE people and 677/85/85 ADHD people. Published split records use zero-based pool-row indices and omit participant identifiers and labels.

Validate the data and assignments without fitting:

```powershell
python revision_20261117/run.py check-data --study abide --data "D:/data/ALLASD1_aal.mat" --round 1
python revision_20261117/run.py check-data --study adhd --data "D:/data/ADHD1_aal.mat" --round 1
```

Run a fixed manuscript division, including its test evaluation and optional paired full-90-ROI control:

```powershell
python revision_20261117/run.py train --study abide --data "D:/data/ALLASD1_aal.mat" --round 1 --evaluate-test --with-dense
python revision_20261117/run.py train --study adhd --data "D:/data/ADHD1_aal.mat" --round 1 --evaluate-test --with-dense
```

Run rounds 1 through 5 separately to reproduce all five fixed divisions. Omitting `--evaluate-test` gives development results only. These commands do not search for additional favorable divisions. Outputs go under `revision_20261117/outputs/`; an existing output folder is refused. Models stay in RAM. Predictions, scalar histories, ROI scores and settings are retained; no intermediate or final training weights are saved.

To run the later fixed development comparison, use rounds 1 through 3 and both initializers:

```powershell
python revision_20261117/run.py train --family development --initializer global_guided32 --study abide --data "D:/data/ALLASD1_aal.mat" --round 1
python revision_20261117/run.py train --family development --initializer within_site_guided32 --study abide --data "D:/data/ALLASD1_aal.mat" --round 1
```

The development family rejects `--evaluate-test`.

## Code map

| File | Role |
|---|---|
| `code/target_guided_kan_20261116.py` | Current initialization, KAN fitting, ROI scoring, cropping and prediction |
| `code/activity_entropy_auc_search_kan.py` | Actual Adam optimization and training-internal checkpoint selection |
| `code/activity_entropy_kan.py` | Function-activity entropy regularizer |
| `code/conditional_site_kan.py` | Complete KAN functions, training-only preprocessing and physical compaction |
| `code/bootstrap_native_kan.py` | Bootstrap branches and whole-output ROI deletion effects |
| `code/roi_contribution_thresholds.py` | Cumulative contribution-mass selection and other development threshold rules |
| `code/site_centered_guided_kan_20261116.py` | Later train-only within-site initialization comparison |
| `audit/conditional_site_math.py`, `audit/full_function_math.py` | Independent NumPy calculations used by the original checks |

Other Python files are original transitive imports or recorded experiment/audit drivers. Their original `main` functions expect the complete local research archive, including earlier frozen plans and artifacts. Use `run.py` for a fresh checkout. Baseline histories from earlier repository revisions remain historical; no baseline model is silently retrained by the native KAN entry point.

`MANIFEST.json` checks the released payload; `SOURCE_PROVENANCE.json` separately checks the exact original Python source bytes. See [中文说明](README_ZH.md).

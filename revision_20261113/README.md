# BrainKAN: training-resample Elastic Net rank consensus

This revision publishes the EN-consensus implementation and records corresponding to the November 2026 manuscript. The executable path is **FC matrices → training-half Elastic Net fits → mean ROI ranks → 75-ROI subnetwork → newly trained KAN → final classification**. The [earlier permutation-importance revision](../revision_20261008/README.md) is retained separately.

中文：本版是训练集内部互补重采样的 EN 共识选脑区。100个半样本排名取平均，自动确定75个脑区的身份，再训练 KAN。它与早期置换重要性和 Min-max/Z-score 阈值版本分别保留。

## Install and verify the published results

Use Python 3.11. From this directory:

```bash
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -r requirements.txt
python run.py verify --output verification.json
python -m unittest discover -s tests -v
```

The recorded experiments used PyTorch 2.2.1+cu121, NumPy 1.26.4, SciPy 1.12.0 and scikit-learn 1.4.1.post1. Install the matching CUDA build of PyTorch if using an NVIDIA GPU. `verify` works on a CPU and does not need the MAT data or model weights. It checks file hashes, all **110 final prediction records**, 2,000 half-selector rankings, 1,000 complementary pairs, cohort assignments, the chosen ROI sets and the reported metrics.

## Method

For each outer training cohort, the saved plan specifies 50 complementary pairs of half-samples. When the training size is odd, one randomly omitted subject is recorded for each pair. Each half fits its own StandardScaler and Elastic Net logistic regression on the 4,005 upper-triangle FC edges, excluding diagonal and duplicate edges. Only the subjects in that training half fit either component.

The selector uses `C=0.1`, `l1_ratio=0.5`, `solver=saga`, `tol=1e-5`, `max_iter=20000`, and `random_state=1`. The score of an ROI is the L2 norm of its incident standardized-edge coefficients. Within each half, scores become average ranks (ties receive average ranks). Their mean across the 100 half-samples determines the consensus ranking. Retain the top 75, break ties by zero-based ROI ID, then supply the ROI submatrix in ascending ROI ID order.

The retained KAN applies per-subject LayerNorm **after** extracting the ROI submatrix, then flattens its full 75×75 matrix. It uses the original local efficient-KAN implementation with a fixed spline grid. ABIDE and NYU use hidden widths `[64,32]`, grid size 5, spline order 3, AdamW, initial learning rate 0.001, weight decay 0.0001, batch size 96, `shuffle=True`, and an exponential learning-rate multiplier of **0.8 after every epoch**. The last prescribed epoch (50) is evaluated. The legacy configuration dictionary contains a `dropout` field shared with other architectures; the fixed-grid KAN constructor does not apply dropout.

ADHD uses each partition's recorded Full KAN development-selected configuration, transferred unchanged to its EN75 classifier. `--development-search` repeats the 12-candidate Full KAN search rather than replaying those saved choices. The choice rule is development ACC, then AUC, then lower loss, fewer parameters and candidate ID. The final test does not enter this choice. Model/data/source settings are sealed before the final evaluation.

This is **EN coefficient-rank consensus**. It does not apply the CPSS frequency threshold and does not use KAN permutation scores to rank these ROIs.

## Data and exact subject assignments

Supply the retained `ALLASD1_aal.mat` (ABIDE) or `ADHD1_aal.mat` file. Raw data and neural checkpoints are not stored in Git. The MAT keys are:

```text
net_train, net_valid, net_test                 # (subjects, 90, 90), finite FC values
phenotype_train, phenotype_valid, phenotype_test
```

The loader concatenates these three arrays in that order. `records/studies/*.json` contains exact **pool indices** into this concatenation, dataset SHA256, per-half subject assignments, classifier configurations and ROI selections. ABIDE labels use phenotype column 2; ADHD labels use column 3 (zero-based, explicit 0/1). The filename fields `net_valid` and `net_test` do not override these saved outer partitions.

| Study | Training / development / final test | Repetition unit |
|---|---:|---|
| ABIDE | 709 / 89 / 84 | All five recorded random holdouts, IDs 0–4 |
| ADHD-200 | 679 / 86 / 82 | All ten recorded random holdouts, IDs 0–9 |
| NYU | 598 / 113 / 171 | One site-disjoint assignment, five stochastic seeds 0–4 |

The random holdouts mix acquisition sites and overlap between repetitions; they are not five-fold cross-validation or five disjoint cohorts. For NYU, UM is development, NYU is final test, and the remaining sites train the model. No site appears in more than one role in that site experiment.

## Refit the complete EN → KAN pipeline

Use a new output directory. These commands fit the selectors again rather than importing a shared cross-partition ROI intersection:

```bash
python run.py run --study abide --data /path/to/ALLASD1_aal.mat --output /path/to/new_abide --device cuda
python run.py run --study adhd --data /path/to/ADHD1_aal.mat --output /path/to/new_adhd --device cuda --development-search
python run.py run --study nyu --data /path/to/ALLASD1_aal.mat --output /path/to/new_nyu --device cuda
```

`--device cpu` is supported. `--partitions 0` runs one recorded partition (or NYU repeat) for inspection. By default, every recorded partition is included. For ADHD, omitting `--development-search` replays the saved classifier configuration for each partition and still refits EN and KAN. Repeating the development search trains all candidates on training subjects and chooses by development metrics before fitting EN75; it does not select the best test score or best test epoch.

Safe pause in PowerShell, replacing the output path:

```powershell
New-Item -ItemType File -Path 'D:\results\new_abide\PAUSE' -Force
Get-Content 'D:\results\new_abide\status.json'
# Wait for stage=paused and safe_to_restart=true before restarting the computer.
Remove-Item -LiteralPath 'D:\results\new_abide\PAUSE'
# Repeat the exact original run command to resume.
```

The current EN half-fit finishes before pause; KAN pauses at an epoch boundary and saves optimizer, scheduler and random-number states. Storage keeps one rolling checkpoint, final EN75 weights and one leader per development search, together with lightweight histories/predictions. It does not retain a separate network for every epoch/candidate. All requested development jobs finish and are sealed before any final predictions are evaluated. The workflow creates a new result folder and does not overwrite the published records.

## Published results and table mapping

`python run.py verify --output verification.json` reconstructs the following values from the saved subject-level probabilities. P, R and F1 use macro averaging; SD is sample SD (`ddof=1`).

| BrainKAN scope | ACC (%) | AUC (%) |
|---|---:|---:|
| ABIDE, all five random holdouts | 66.67 ± 6.63 | 69.31 ± 4.61 |
| ADHD, all ten random holdouts | 66.46 ± 4.57 | 69.56 ± 5.01 |
| ADHD, manuscript five-partition subset | 69.76 ± 4.43 | 72.57 ± 4.85 |
| NYU, five stochastic repeats | 65.96 ± 0.96 | 70.95 ± 0.51 |

The ADHD manuscript subset is IDs **1, 2, 5, 6, 8**, selected by the author after inspecting all ten completed results. Its descriptive summary is reported alongside the complete ten-partition results. The ABIDE EN runs are exploratory method development on previously inspected holdouts. Within each fit, EN scaling and ranking use training subjects only; the published records also retain this study-level development history.

| Manuscript material | Published records |
|---|---|
| ABIDE baseline table | `records/predictions/abide/`: BrainGNN, BrainGB, BrainNetCNN, FBNetGen, BrainNetGNN, BrainNetTF, Res-DAE, and BrainKAN; 40 predictions |
| Full KAN parameter table | `records/reference_tables/tables.json`, `hyperparameter_table.tex`, and the candidate-level `records/development/` histories, configurations and development predictions |
| ADHD table and complete supplement | `records/predictions/adhd/`: five FC-input baselines and transferred EN75; all 60 predictions across ten partitions |
| NYU EN comparison | `records/predictions/nyu/`: EN75 and matched Full KAN; all 10 predictions |
| ROI choices and stability | `records/studies/*.json`, `records/*_half_scores.npz`; ranks and memberships are checked by `verify` |

The ADHD `brainkan` label here denotes the **Full90-development-selected → EN75 transfer** used in the manuscript. A separate direct-EN75 development search was also explored locally; its records are not substituted for the transferred configuration in these tables. NYU compares the same `[64,32]` KAN with 75 versus 90 ROIs.

## Files and provenance

- `code/`: portable workflow; numerical EN, KAN, feature, metric and training routines copied from the recorded implementations.
- `records/`: exact assignments, compact half-selector scores, configurations, predictions, reference summaries and development search records.
- `snapshots/`: frozen experiment and baseline implementations for source inspection. Their orchestration scripts retain original local study paths; use `run.py` for portable EN training. Baseline adapters and training budgets are identified by the saved code/records rather than represented as fresh official benchmark scores.
- `SOURCE_PROVENANCE.json`: relative original filenames and SHA256 values; the local studies remain unchanged.
- `MANIFEST.sha256.json`: hashes of this public revision.
- `PREPROCESSING.md`: verified input derivative/procedure and preprocessing references.

The efficient-KAN source license is included as `code/efficient-kan-LICENSE`. Third-party baseline licenses are retained beside their snapshots. No additional license is asserted for third-party sources.

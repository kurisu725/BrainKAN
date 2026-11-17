# BrainKAN

## Current native KAN workflow (17 November 2026)

The [current native KAN implementation](revision_20261117/README.md) contains the actual code used for the latest manuscript: **complete KAN function learning → activity-entropy regularization and weak-function pruning → KAN whole-output ROI contributions → cumulative contribution threshold → physically cropped FC → continued KAN classification**. ABIDE uses 90% and ADHD uses 95% squared contribution mass. Training-only ridge directions assist initialization of three channels; all KAN functions then train jointly.

- [Installation and fixed-division training commands](revision_20261117/README.md)
- [中文说明](revision_20261117/README_ZH.md)
- [Actual model code](revision_20261117/code/target_guided_kan_20261116.py), [source-byte provenance](revision_20261117/SOURCE_PROVENANCE.json), and [exact pool-row splits](revision_20261117/records/selected_splits.json)
- [Results and complete original-recipe attempt history](revision_20261117/records/selected_results.json)

The reported five-division test ACC/AUC are ABIDE **73.03 ± 0.79% / 78.79 ± 2.41%** and ADHD **71.06 ± 1.97% / 73.62 ± 2.51%** (mean ± sample SD). The divisions were selected using observed test performance and are exploratory. The current snapshot records that selection explicitly. Verify the released code and records without data or training with `python revision_20261117/run.py verify`.

Use the native version above for the current manuscript. Earlier workflows below are preserved as historical revisions.

## Historical EN-consensus workflow (13 November 2026)

The [2026-11-13 EN-consensus revision](revision_20261113/README.md) contains the earlier executable **FC → training-half Elastic Net rank consensus → automatic Top-75 ROI selection → KAN training → final classification** workflow and its historical records. It has been superseded by the native KAN route above.

- [Installation, full-pipeline commands and safe pause/resume](revision_20261113/README.md)
- [Historical EN code](revision_20261113/code/), [exact study plans and ROI selections](revision_20261113/records/studies/), and [subject-level predictions](revision_20261113/records/predictions/)
- [Preprocessing provenance and references](revision_20261113/PREPROCESSING.md)
- [Earlier permutation workflow and five-seed records](revision_20261008/README.md)

The historical EN selector uses training subjects only. That comparison includes all five ABIDE holdouts, all ten ADHD holdouts with its selected five-partition summary identified separately, and the NYU site-disjoint experiment. Its metrics can be recomputed with `python revision_20261113/run.py verify` without the raw data. The instructions below document the original permutation scripts and historical environment; use `revision_20261117` for the current workflow.

## Original implementation


BrainKAN is an automated region-aware Kolmogorov-Arnold Network (KAN) for
resting-state fMRI brain-network classification. It learns connection-level
contributions, aggregates them into ROI-level importance scores, selects an
informative ROI subnetwork, and retrains a KAN classifier on the selected
functional-connectivity matrix.

## Requirements

Miniconda with Python 3.9 is recommended for the original environment. The core
versions pinned in `requirements.txt` are:

- Python `3.9`
- PyTorch `2.2.1+cu121`
- torchvision `0.17.1+cu121`
- torchaudio `2.2.1+cu121`
- NumPy `1.26.4`
- SciPy `1.12.0`
- scikit-learn `1.4.1.post1`
- pandas `2.2.1`
- matplotlib `3.8.3`
- tqdm `4.66.2`
- LIME `0.2.0.1`

Create the environment and install the dependencies:

```bash
conda create -n brainkan python=3.9 -y
conda activate brainkan
pip install -r requirements.txt
```

`requirements.txt` preserves several Windows-local PyTorch Geometric wheel paths
from the original machine. When installing on another computer, replace those
`file:///E:/...` entries with wheels matching your Python, PyTorch, and CUDA
versions. The local `efficient_kan` implementation used by BrainKAN itself does
not require PyTorch Geometric.

## Datasets

BrainKAN uses AAL-90 functional-connectivity matrices. Each subject is represented
by a `90 x 90` matrix. Labels are binary: `0` for control/negative and `1` for
patient/positive.

| Dataset | MAT file | Subjects | Train / validation / test | Label column (zero-based) |
|---|---|---:|---:|---:|
| ABIDE | `ALLASD1_aal.mat` | 882 | 709 / 89 / 84 | 2 |
| ADHD-200 | `ADHD1_aal.mat` | 847 | 679 / 86 / 82 | 3 |

Place the data beside the code directory using this layout:

```text
BrainKAN/
|-- BrainKAN-code/
|   |-- efficient_kan/
|   |-- test_model.py
|   `-- requirements.txt
`-- BrainKAN-data/
    |-- ALLASD1_aal.mat
    `-- ADHD1_aal.mat
```

The MAT archives are expected to contain `net_train`, `net_valid`, `net_test`,
`phenotype_train`, `phenotype_valid`, and `phenotype_test`. The current
`test_model.py` retains the original machine-specific path in its `sio.loadmat(...)`
calls. Update both occurrences to your local MAT file before running, for example:

```python
load_data = sio.loadmat('../BrainKAN-data/ALLASD1_aal.mat')
```

## Quick start

From the `BrainKAN-code` directory, verify that the local KAN implementation can
be imported:

```bash
python -c "from efficient_kan import KAN; print(KAN([8100, 256, 128, 64, 32, 16, 2]))"
```

After configuring the dataset path, run the experiment script:

```bash
python test_model.py
```

For classification-only runs, the unused `import xai` line can be
disabled; `testXAI()` will then remain unavailable.

The current `__main__` block calls `testKAN()`. The example implementation sweeps
the number of selected ROIs from 1 to 90. For a single fixed ROI configuration,
set `matrixAuto` to the selected ROI order, set `anum` to the desired number of
ROIs, and remove or narrow the sweep loop.




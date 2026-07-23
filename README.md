# BrainKAN

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




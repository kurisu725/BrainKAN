# BrainKAN completed reviewer experiments

This is a copy of the executable source used for the completed October 2026 evaluation. Source hashes match both frozen experiment plans. The evaluation completed 160 model fits (two datasets, five seeds, 16 configurations), all using the final 50th epoch.

Use Python 3.11 and the pinned dependencies. The recorded training environment is in `../results/runtime.json`; GPU execution used PyTorch 2.2.1+cu121. The FC inputs are `ALLASD1_aal.mat` and `ADHD1_aal.mat` in the existing BrainKAN data package. Their hashes and exact row assignments are in `../results/abide/plan.json` and `../results/adhd/plan.json`.

From this code directory, generate plans with local data paths, then run the development and evaluation stages into new directories:

```powershell
python scripts/reviewer_experiments.py plan --dataset abide --data data/ALLASD1_aal.mat --output abide_plan.json --symmetric
python scripts/reviewer_experiments.py plan --dataset adhd --data data/ADHD1_aal.mat --output adhd_plan.json --symmetric
python scripts/reviewer_experiments.py run --plan abide_plan.json --output outputs/abide --device auto
python scripts/reviewer_experiments.py run --plan adhd_plan.json --output outputs/adhd --device auto
python scripts/reviewer_experiments.py evaluate --experiment outputs/abide --device auto
python scripts/reviewer_experiments.py evaluate --experiment outputs/adhd --device auto
python scripts/summarize_reviewer_run.py outputs
```

Check generated plans against the recorded assignments, hashes, seeds and configuration before training. The recorded plans retain their original machine paths for provenance. ROI contributions use development subjects; final evaluation runs after development artifacts have been frozen.

The result package contains contribution arrays, per-unit loss differences, ROI lists, training histories and predictions. Large model checkpoint files and input MAT files remain in the existing local experiment/data directories and are omitted from this document-and-results delivery. Fresh training regenerates checkpoints. Running `evaluate` against the copied result folder requires the original checkpoints; the commands above use newly generated output directories.

The main CLI in `run.py` also supports classification of subsequent unlabeled FC matrices using saved checkpoints and ROI indices; see `python run.py --help` for its commands.

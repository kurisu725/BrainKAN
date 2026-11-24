# ADHD original-paper FC baselines

Five fixed diagnosis-stratified mixed-site random holdouts: 679 training / 86 development / 82 test; same subjects as the ADHD EN75 parameter study. Not five-fold cross-validation.

Prespecified final epoch: BrainGNN 100, BrainGB 100, BrainNetTF 200, BrainNetCNN 50.

Training seed 1. Model-specific learning rates, regularizers and mixup follow the already-audited ABIDE implementations. All 20 development jobs are sealed before final test evaluation. Main comparison includes one BrainKAN EN75 row, using the existing per-partition development-selected parameters. All candidate and preprocessing provenance remains separate from historical paper results.

PAUSE.cmd requests an epoch-boundary safe pause. STATUS.cmd reports safe_to_restart. RESUME.cmd resumes with optimizer/scheduler/RNG states. One rolling checkpoint and one final model per method per partition are retained.

# ADHD Res-DAE reproduction

Same five ADHD 679/86/82 mixed-site random holdouts. Same paper-based Res-DAE implementation as the ABIDE reproduction. Four width/lr candidates per partition; development AUC selects candidate, development Youden selects threshold.

Training-only ComBat with age/sex; Fisher top 2002 edges; training-only EROS-SMOTE; train-fitted feature standardization.

200 joint-training epochs then 50 classifier-finetuning epochs; use epoch 250 for each candidate.

Age and sex were linked from the public PCP/NITRC metadata by all 847 subject IDs. Subject 10044 has missing sex in both the consolidated table and original NYU phenotype file. Each partition fits the sex mode on observed training subjects only; ties choose the smaller encoded value. The same fitted value transforms missing development/test metadata. No subjects are dropped.

One rolling checkpoint and one chosen model per partition are retained; all four candidates keep light histories and development predictions. The controller waits until the four FC baselines finish and validate. No test set enters candidate selection, threshold selection or ComBat fitting.

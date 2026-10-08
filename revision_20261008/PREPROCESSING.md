# Preprocessing source

BrainKAN uses preprocessed AAL-90 Pearson-correlation functional-connectivity matrices. Image preprocessing follows the procedure cited in the manuscript:

Junzhong Ji and Xingyu Wang (2023). *Convolutional architecture search based on particle swarm algorithm for functional brain network classification*. Applied Soft Computing 149, 111049. [Original article](https://doi.org/10.1016/j.asoc.2023.111049), **Section 4.1, page 6**.

The original description states:

> (1) removing the first five time frames of the original rs-fMRI and performing slice timing correction. (2) eliminating head motion data with a horizontal displacement of more than 2 millimeters and a rotation angle of more than 2 degrees for head motion correction. (3) smoothing, registration and filtering operations are performed on the images in order.

**For the preprocessing procedure, please refer to the original article cited above (Section 4.1, p. 6).**

**预处理流程请参考上述原文第 4.1 节、第 6 页。**

The same section describes AAL parcellation with 90 regions and Pearson-correlation FC construction. BrainKAN performs ROI selection and classification from the resulting FC matrices.

| Setting | Information in the cited passage |
|---|---|
| Software | DPARSF |
| Initial frames | Remove the first five frames |
| Slice timing | Slice-timing correction |
| Gross-motion exclusion | Displacement above 2 mm or rotation above 2 degrees |
| Smoothing, registration, filtering | Processing steps stated |
| Filter passband, smoothing FWHM | Numerical settings require the configuration associated with the input data |
| Nuisance/motion regressors, global signal handling | Require the configuration associated with the input data |
| Input representation | AAL-90, Pearson correlation |

The new experiments preserve the train/validation/test assignments stored in the MAT files and use seeds 0-4 on those fixed assignments. Diagnosis uses zero-based phenotype column 2 for ABIDE and column 3 for ADHD. Checksums and exact row assignments are recorded in `results/abide/plan.json` and `results/adhd/plan.json`.

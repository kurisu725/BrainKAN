# BrainKAN
Environment Setup
Refer to requirements.txt for necessary packages.

Dataset

The dataset used is the ABIDE dataset, with inputs being 90x90 matrices of AAL brain regions. Labels are binary: 0 (non-patient, negative) and 1 (patient, positive).

Usage
Using KAN:
Run the following to use the KAN model:
testKAN()

Automatic Extraction of High-impact Brain Regions:
To extract the brain regions that contribute significantly, perform the following steps:
Execute testKAN() to generate a model.
Use the model with testXAI() as follows:testXAI()
testXAI() will produce a 90x90 matrix of contributions. Summing these will yield the individual contributions of the 90 brain regions. According to the methods described in the paper, process the results to retain regions with a contribution less than zero.
Modify the input matrix for testKAN() to include only these extracted brain regions.
These steps will help you identify and focus on the most impactful areas of the brain according to the model's analysis.

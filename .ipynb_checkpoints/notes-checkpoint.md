# Notes - Trial Outcome Prediction
Documenting research progress through the semester.

## Week 1-2

- Reading papers
    - HINT (SOTA model): [HINT: Hierarchical interaction network for clinical trial-outcome predictions](https://www.cell.com/patterns/pdf/S2666-3899(22)00018-6.pdf)
    - PyTrial (Benchmark tool): [PyTrial: Machine Learning Software and Benchmark for Clinical Trial Applications](https://arxiv.org/abs/2306.04018)
    - AnyPredict / MediTab: [MediTab: Scaling Medical Tabular Data Predictors via Data Consolidation, Enrichment, and Refinement](https://arxiv.org/abs/2305.12081) - The paper for PyTrial claimed that this model outperformed HINT. Their model is not available, thus the results cannot be verified or reproduced. The described approach does not seem very promising either.
    - Factors of clinical trial outcomes: [Factors Affecting Success of New Drug Clinical Trials](https://link.springer.com/article/10.1007/s43441-023-00509-1)

- Implementing GPT Baseline
    Evaluated `GPT-4` and `GPT-3.5` on the `Trial Outcome Prediction` datasets with various prompting strategies, including Chain of Thought, and breaking down the challenge into subtasks. When tested for binary classification with 30-50 uniformly distributed samples, the models showed very poor performance.

## Week 3
- Testing HINT
    Reproducing HINT's results on a virtual machine and analyzing which features could be improved.

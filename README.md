# Progressive Diagnosis-Specific Representation Learning for Multi-Label ECG Classification

This repository contains the official implementation of the paper:

**Progressive Diagnosis-Specific Representation Learning via Diagnostic Dependency Modeling for Multi-Label ECG Classification**

## Overview

This work proposes a progressive diagnosis-specific representation learning framework for multi-label ECG classification. The framework consists of:

- Diagnosis-Specific Representation (DSR) learning
- Diagnostic Dependency Modeling (DDM)
- Intrinsic diagnostic dependency attention for model interpretability

The framework is evaluated on the PTB-XL benchmark dataset using patient-independent data partitioning.

## Repository Structure

```
src/            Source code
notebooks/      Training and evaluation notebooks
tests/          Unit tests
```

## Dataset

The experiments use the publicly available PTB-XL dataset available from PhysioNet:

https://physionet.org/content/ptb-xl/1.0.3/

The dataset is **not included** in this repository.

## Reproducibility

The repository contains the implementation used to generate the experimental results reported in the accompanying manuscript.

## License

This repository is provided for research purposes.

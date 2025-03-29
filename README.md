# README: Modeling H3 Variant

## Overview
This project focuses on modeling the inheritance and distribution of **H3 histone variants (H3.1 and H3.3)** in genomic regions. The workflow involves data preprocessing, probabilistic modeling, and computational analysis to infer patterns of histone inheritance across generations.

## Project Workflow
### 1. Data Preprocessing
- Input **ChIP-seq** datasets for H3.1 and H3.3 are processed.
- Data is converted into **5 Mb bins** with presence (1) or absence (0) indicators for each histone variant.

### 2. Sequence Generation
- Sequences of H3.1 and H3.3 occurrences are generated.
- Binary encoding is used: `1` for presence and `0` for absence.

### 3. Probabilistic Modeling
- The project implements a **Viterbi-based decoding** algorithm to reconstruct the mother sequence from the observed daughter sequences.
- Transition probabilities are computed using `mu1` and `mu2`, reflecting persistence probabilities.

### 4. Filtering and Post-Processing
- Sequences are filtered based on empirical thresholds (`mu1`, `mu2`).
- Further refinements are applied to ensure consistent classification.

## Installation & Requirements
### Dependencies:
Ensure you have the following installed:
- **Python 3.8+**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `joblib`
- `nucpossimulator`
- `samtools`
- `sratoolkit`
- `bedtools`

## Usage
### Running Simulations
The folder **H3_decode_sim** contains all simulation scripts:
- **`Variant_sim.py`**: Contains all functions required for data generation, Viterbi algorithm correction, etc.
- **`Run_variant_sim.py`**: The driver script to execute simulations.

#### Example Run Command:
```sh
python3 Run_variant_sim.py --batch_name "TestBatch" \
  --alpha1_eval "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]" \
  --beta1_eval "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]" \
  --alpha2 0.2 --beta2 0.2 --mu1 0.6 --mu2 0.4 --rho 0.5 \
  --seed 42 --n_samples 10 --seq_length 100 --verbose_level 5
```

### Experimental Validation
The **Expt_Valid** directory contains all scripts related to experimental validation, sequentially numbered for easy reference.

## Output
- Simulated sequences and probability metrics stored as `.csv` files.
- Processed data with histone variant assignments.
- Visualization plots for sequence distributions and inheritance patterns.

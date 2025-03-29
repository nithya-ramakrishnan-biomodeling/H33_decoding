# README: Modeling H3 Variant

## Overview
This project focuses on modeling the inheritance and distribution of **H3 histone variants (H3.1 and H3.3)** in genomic regions. The workflow involves data preprocessing, probabilistic modeling, and computational analysis to infer patterns of histone inheritance across generations.

## Project Workflow
1. **Data Preprocessing:**
   - Input **ChIP-seq** datasets for H3.1 and H3.3 are processed.
   - Data is converted into **5 kb bins** with presence (`1`) or absence (`0`) indicators for each histone variant.
   
2. **Sequence Generation:**
   - Sequences of H3.1 and H3.3 occurrences are generated.
   - Binary encoding is used: `1` for presence and `0` for absence.
   
3. **Probabilistic Modeling:**
   - The project implements a **Viterbi-based decoding** algorithm to reconstruct the mother sequence from the observed daughter sequences.
   - Transition probabilities are computed using key parameters:
     - **alpha1**: Probability of H3.1, `1` followed by `1`.
     - **beta1**: Probability of H3.1, `0` followed by `0`.
     - **alpha2**: Probability of H3.3, `1` followed by `1`.
     - **beta2**: Probability of H3.3, `0` followed by `0`.
   - Additional parameters:
     - **mu1**: probability of staying in h3.1.
     - **mu2**: Probability of staying in h3.3.
     - **rho**: Probability of a random flip (noise in sequencing, loss at replication).
   
4. **Filtering and Post-Processing:**
   - Sequences are filtered based on empirical thresholds (`mu1`, `mu2`).
   - Further refinements are applied to ensure consistent classification.
   
## Installation & Requirements
### Dependencies:
Ensure you have the following installed:
- **Python 3.8+**
- pandas
- numpy
- matplotlib
- seaborn
- joblib
- samtools
- bedtools

## Usage
### Running the Simulation
To generate simulated data and decode sequences, run:
```bash
python3 run_variant_sim.py \
  --batch_name "TestBatch" \
  --alpha1_eval "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]" \
  --beta1_eval "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]" \
  --alpha2 0.2 \
  --beta2 0.2 \
  --mu1 0.6 \
  --mu2 0.4 \
  --rho 0.5 \
  --seed 42 \
  --n_samples 10 \
  --seq_length 100 \
  --verbose_level 5
```

### Processing Experimental Data
Run the following script to process experimental data:
```bash
python3 process_experimental_data.py --input_dir /path/to/input --output_dir /path/to/output
```

## Output Files
- **Filtered Sequences:** Processed and filtered sequences based on probabilistic modeling.
- **Histone Variant Inheritance Predictions:** CSV files with inferred histone inheritance states.

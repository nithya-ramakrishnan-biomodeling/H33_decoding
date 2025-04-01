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
- sratoolkit
- Nucpossimulator: [NucPos Quick Start Guide](https://bioinformatics.hochschule-stralsund.de/nucpos/quick_start.html)
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
The scripts in experimental data validations are numbered just have to follow along.
- **Number 0:** PDF with guide to download and process data to .bed format.
- **Number 1:** Subsetting the bed files into 5mb regions and running **Nucpossimulator**.
- **Number 2:** Descretising the nuspossimulator output and rounding off to 200.
- **Number 3:** Combining both H3.1 and H3.3, descretised files.
- **Number 4:** Correcting invalid rows based on random number generation or referring to nuspossimulator result.bed.
- **Number 5:** Median filtering (which is optional step)
- **Number 6:** Conflict resolve check the invalid and update based on max appeared value in that window.
- **Number 7:** Generating source- seq with information of if the nucleosome is H3.1 or H3.3.
- **Number 8:** Calculating probability values, mu1 and mu2.
- **Number 9:** Filtering source as per mu1, mu2 values.
- **Number 10:** Experimental validation- generating mother as per source and running algorithm.

## Output Files
- **Filtered Sequences:** Processed and filtered sequences based on probabilistic modeling.
- **Histone Variant Inheritance:** CSV files with inferred histone variant.
- **Ouput Csv:** with alpha1, beta1, alpha2, beta2, mu1 and mu2 with BERs.
- **Violin plot:** with mu1=mu2, grouped at low, mid and high with BER distribution.

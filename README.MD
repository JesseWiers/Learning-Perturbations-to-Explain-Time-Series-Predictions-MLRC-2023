# Reproducibility Study of Learning Perturbations to Explain Time Series Predictions

## Overview
This project aims to investigate and reproduce the findings related to [Learning Perturbations to Explain Time Series Predictions](https://doi.org/10.48550/arXiv.2305.18840) (Joseph Engehuhard 2023). . Our study includes a range of experiments with Hidden Markov Models (HMM), MIMIC-III dataset analysis, and our contribution involving an LSTM model.

## Installation

### Prerequisites
- Python (Version 3.11 recommended)
- pip (Python package installer)

### Installing Dependencies
To install the required Python packages, navigate to the project directory and run:

```bash
pip install -r requirements.txt
```

## Getting Started

### Examples
The `examples/` directory contains Jupyter notebooks that demonstrate how to execute the experiments. These notebooks serve as a practical guide to understanding the implementation details and usage of the models.

### Key Experiments
- HMM Experiments: Located in `experiments/hmm/`, these files detail the work work with Hidden Markov Models.
- MIMIC-III Data Analysis: Files related to MIMIC-III are in `experiments/mimic3/`.


## Data Preparation

### MIMIC-III Dataset
The MIMIC-III dataset is a critical component of our study. To use this dataset:

1. **Access the Data**: Obtain the MIMIC-III dataset from [PhysioNet](https://physionet.org/content/mimiciii/1.4/).
2. **Database Setup**: Host a PostgreSQL database instance with the downloaded data.
3. **Data Processing**: Run the script `tint/datasets/mimic3.py` to process and convert the data into `.pkl` format suitable for our experiments.

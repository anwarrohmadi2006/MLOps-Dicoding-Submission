# 🏥 End-to-End MLOps Pipeline: Experimentation Phase (K1)
**Project Title**: House Price Prediction Analysis & Automated Preprocessing  
**Author**: Anwar Rohmadi  
**Organization**: Dicoding Academy - Membangun Sistem Machine Learning

---

## 📋 Project Overview
This repository contains the first phase (K1) of the MLOps lifecycle for the Ames Housing dataset. The primary objectives are to perform high-quality Exploratory Data Analysis (EDA) and establish a robust, automated data preprocessing pipeline.

## 📁 Repository Structure
The project follows a modular structure to ensure maintainability and reproducibility:

```text
Eksperimen_SML_Anwar-Rohmadi/
├── .github/workflows/       # Automated CI Pipeline
│   └── preprocess.yml       # Triggers preprocessing on data changes
├── house_prices_raw/        # Source Dataset (Raw)
│   ├── train.csv            # Training features and targets
│   └── test.csv             # Unseen test data
├── preprocessing/           # Preprocessing & Analysis Core
│   ├── Eksperimen_Anwar-Rohmadi.ipynb  # Comprehensive EDA Report
│   ├── automate_Anwar-Rohmadi.py       # Production-ready automation script
│   └── house_prices_preprocessing/     # Validated preprocessed artifacts
├── requirements.txt         # Technical environment specifications
└── README.md                # Project documentation
```

## 📊 Phase 1: Exploratory Data Analysis (EDA)
The analysis documented in `Eksperimen_Anwar-Rohmadi.ipynb` includes:
- **Descriptive Statistics**: Thorough evaluation of feature distributions and scales.
- **Multivariate Analysis**: Correlation matrices and scatter plots identifying key drivers of property value.
- **Quality Audit**: Identification of missing values, outliers, and data inconsistencies.

## 🛠️ Phase 2: Automated Preprocessing (Skilled/Advance)
The `automate_Anwar-Rohmadi.py` script implements an industrial-standard preprocessing pipeline:
1. **Missing Data Imputation**: Calculated based on training set statistics (Median for numeric, Mode for categorical).
2. **Feature Engineering**: Derivation of high-signal features such as `TotalSF` (Square Footage) and `HouseAge`.
3. **Categorical Encoding**: Implementation of robust Label Encoding.
4. **Feature Scaling**: Standardization to ensure convergence in gradient-based algorithms.

## 🚀 Execution Instructions

### Deployment Environment
```bash
pip install -r requirements.txt
```

### Manual Trigger
```bash
cd preprocessing
python automate_Anwar-Rohmadi.py
```

### Automation (CI)
The system is integrated with **GitHub Actions**. Every update to the preprocessing logic or raw data triggers an automated validation run, ensuring the data integrity of the downstream model training phase.

---
*This project is submitted as part of the MLOps certification at Dicoding Academy.*

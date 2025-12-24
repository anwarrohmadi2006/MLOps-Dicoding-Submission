"""
Run Notebook Cells - Generate outputs for Eksperimen_Anwar-Rohmadi.ipynb
This script executes key preprocessing steps and saves outputs.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
RAW_DIR = "../house_prices_raw"
PROCESSED_DIR = "../house_prices_preprocessing"
TARGET_COL = "SalePrice"
ID_COL = "Id"

print("=" * 60)
print("EKSPERIMEN MACHINE LEARNING - HOUSE PRICES PREDICTION")
print("Author: Anwar Rohmadi")
print("=" * 60)

# 1. Load Dataset
print("\n[1/6] Loading Dataset...")
train_df = pd.read_csv(f"{RAW_DIR}/train.csv")
test_df = pd.read_csv(f"{RAW_DIR}/test.csv")
print(f"Train dataset shape: {train_df.shape}")
print(f"Test dataset shape: {test_df.shape}")

# 2. EDA - Target Statistics
print("\n[2/6] Exploratory Data Analysis...")
print(f"\n{'='*50}")
print("STATISTIK TARGET (SalePrice)")
print(f"{'='*50}")
print(f"Mean: ${train_df[TARGET_COL].mean():,.2f}")
print(f"Median: ${train_df[TARGET_COL].median():,.2f}")
print(f"Std: ${train_df[TARGET_COL].std():,.2f}")
print(f"Min: ${train_df[TARGET_COL].min():,.2f}")
print(f"Max: ${train_df[TARGET_COL].max():,.2f}")
print(f"Skewness: {train_df[TARGET_COL].skew():.4f}")
print(f"Kurtosis: {train_df[TARGET_COL].kurtosis():.4f}")

# Missing Values
missing = train_df.isnull().sum()
missing_pct = (missing / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)
missing_df = missing_df[missing_df['Missing Count'] > 0]
print(f"\nJumlah kolom dengan missing values: {len(missing_df)}")
print("\nTop 5 kolom dengan missing values:")
print(missing_df.head().to_string())

# Correlations
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
correlations = train_df[numeric_cols].corr()[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)
print("\n\nTop 10 fitur berkorelasi POSITIF dengan SalePrice:")
for feat, corr in correlations.head(10).items():
    print(f"  {feat}: {corr:.4f}")

# 3. Handle Missing Values
print("\n[3/6] Handling Missing Values...")
def handle_missing_values(df):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()
            df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'None', inplace=True)
    return df

train_processed = handle_missing_values(train_df)
print(f"Missing values setelah handling: {train_processed.isnull().sum().sum()}")

# 4. Feature Engineering
print("\n[4/6] Feature Engineering...")
def feature_engineering(df):
    df = df.copy()
    if all(col in df.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    return df

train_processed = feature_engineering(train_processed)
new_features = ['TotalSF', 'TotalBath', 'HouseAge', 'RemodAge']
print(f"New features created: {[f for f in new_features if f in train_processed.columns]}")

# 5. Encoding
print("\n[5/6] Encoding Categorical Variables...")
y = train_processed[TARGET_COL].copy()
X = train_processed.drop(columns=[ID_COL, TARGET_COL])

cat_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Encoded {len(cat_cols)} categorical columns")
print(f"Shape after encoding: {X.shape}")

# Train/Val Split
TEST_SIZE = 0.2
RANDOM_STATE = 42
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"\nTraining set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")

# 6. Scaling and Save
print("\n[6/6] Scaling and Saving Preprocessed Data...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)

os.makedirs(PROCESSED_DIR, exist_ok=True)
X_train_scaled.to_csv(f"{PROCESSED_DIR}/X_train.csv", index=False)
X_val_scaled.to_csv(f"{PROCESSED_DIR}/X_val.csv", index=False)
y_train.to_csv(f"{PROCESSED_DIR}/y_train.csv", index=False)
y_val.to_csv(f"{PROCESSED_DIR}/y_val.csv", index=False)

metadata = {
    'feature_cols': X_train_scaled.columns.tolist(),
    'n_train': len(X_train_scaled),
    'n_val': len(X_val_scaled),
    'n_features': len(X_train_scaled.columns),
    'target_stats': {
        'mean': float(y.mean()),
        'median': float(y.median()),
        'std': float(y.std())
    }
}
with open(f"{PROCESSED_DIR}/metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nFiles saved to: {PROCESSED_DIR}/")
print(f"  - X_train.csv ({len(X_train_scaled)} samples)")
print(f"  - X_val.csv ({len(X_val_scaled)} samples)")
print(f"  - y_train.csv, y_val.csv")
print(f"  - metadata.json")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE!")
print(f"Total features: {metadata['n_features']}")
print(f"Train/Val split: {metadata['n_train']}/{metadata['n_val']}")
print("=" * 60)

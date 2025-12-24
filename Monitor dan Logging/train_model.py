"""
Train and Save Model for Serving
Author: Anwar-Rohmadi
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
import os

print("=" * 60)
print("TRAINING MODEL FOR SERVING")
print("=" * 60)

# Config
RAW_DIR = "../Eksperimen_SML_Anwar-Rohmadi/house_prices_raw"
OUTPUT_DIR = "."
TARGET_COL = "SalePrice"
ID_COL = "Id"

# Load data
print("\n[1/5] Loading data...")
train_df = pd.read_csv(f"{RAW_DIR}/train.csv")
print(f"Loaded {len(train_df)} samples")

# Handle missing values
print("\n[2/5] Preprocessing...")
def preprocess(df):
    df = df.copy()
    # Numeric: fill with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    # Categorical: fill with mode and encode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None', inplace=True)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

train_processed = preprocess(train_df)

# Feature engineering
if all(col in train_processed.columns for col in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
    train_processed['TotalSF'] = train_processed['TotalBsmtSF'] + train_processed['1stFlrSF'] + train_processed['2ndFlrSF']

y = train_processed[TARGET_COL]
X = train_processed.drop(columns=[ID_COL, TARGET_COL])

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Val: {len(X_val)}")

# Train with best hyperparameters from Optuna
print("\n[3/5] Training HistGradientBoostingRegressor...")
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_iter=300,
    min_samples_leaf=15,
    l2_regularization=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("\n[4/5] Evaluating...")
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R2: {r2:.4f}")

# Save model
print("\n[5/5] Saving model...")
model_path = f"{OUTPUT_DIR}/model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

# Save feature names
feature_names = X.columns.tolist()
with open(f"{OUTPUT_DIR}/feature_names.json", 'w') as f:
    json.dump({'features': feature_names, 'n_features': len(feature_names)}, f, indent=2)
print(f"Feature names saved ({len(feature_names)} features)")

# Save model info
model_info = {
    'algorithm': 'HistGradientBoostingRegressor',
    'n_features': len(feature_names),
    'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
    'hyperparameters': {
        'learning_rate': 0.05,
        'max_depth': 8,
        'max_iter': 300,
        'min_samples_leaf': 15
    }
}
with open(f"{OUTPUT_DIR}/model_info.json", 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print(f"Files: model.pkl, feature_names.json, model_info.json")
print("=" * 60)

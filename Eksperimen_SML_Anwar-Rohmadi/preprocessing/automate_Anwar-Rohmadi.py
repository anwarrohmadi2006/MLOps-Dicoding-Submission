"""
Automated Preprocessing Script for House Prices Dataset
Author: Anwar-Rohmadi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
from datetime import datetime

# Configuration
RAW_DIR = "../house_prices_raw"
PROCESSED_DIR = "../house_prices_preprocessing"
TARGET_COL = "SalePrice"
ID_COL = "Id"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_data():
    """Load raw train and test data"""
    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    print(f"Train shape: {train_df.shape}")
    if test_df is not None:
        print(f"Test shape: {test_df.shape}")
    
    return train_df, test_df

def handle_missing_values(df):
    """Handle missing values in dataset"""
    df = df.copy()
    
    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Categorical columns: fill with mode or 'None'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'None', inplace=True)
    
    return df

def encode_categorical(df, label_encoders=None):
    """Encode categorical variables"""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object']).columns
    
    if label_encoders is None:
        label_encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    else:
        for col in cat_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df[col] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
    
    return df, label_encoders

def feature_engineering(df):
    """Create new features"""
    df = df.copy()
    
    # Total square footage
    if 'TotalBsmtSF' in df.columns and '1stFlrSF' in df.columns and '2ndFlrSF' in df.columns:
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    # Total bathrooms
    bath_cols = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
    if all(col in df.columns for col in bath_cols):
        df['TotalBath'] = df['FullBath'] + 0.5*df['HalfBath'] + df['BsmtFullBath'] + 0.5*df['BsmtHalfBath']
    
    # House age
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    
    # Remodel age
    if 'YearRemodAdd' in df.columns and 'YrSold' in df.columns:
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    
    return df

def preprocess_pipeline(train_df, test_df=None):
    """Main preprocessing pipeline"""
    print("Starting preprocessing pipeline...")
    
    # Store IDs
    train_ids = train_df[ID_COL] if ID_COL in train_df.columns else None
    test_ids = test_df[ID_COL] if test_df is not None and ID_COL in test_df.columns else None
    
    # Separate target
    y = train_df[TARGET_COL].copy() if TARGET_COL in train_df.columns else None
    
    # Drop ID and target from features
    drop_cols = [col for col in [ID_COL, TARGET_COL] if col in train_df.columns]
    X = train_df.drop(columns=drop_cols)
    
    if test_df is not None:
        X_test = test_df.drop(columns=[ID_COL] if ID_COL in test_df.columns else [])
    
    # Handle missing values
    print("Handling missing values...")
    X = handle_missing_values(X)
    if test_df is not None:
        X_test = handle_missing_values(X_test)
    
    # Feature engineering
    print("Feature engineering...")
    X = feature_engineering(X)
    if test_df is not None:
        X_test = feature_engineering(X_test)
    
    # Encode categorical
    print("Encoding categorical variables...")
    X, label_encoders = encode_categorical(X)
    if test_df is not None:
        X_test, _ = encode_categorical(X_test, label_encoders)
    
    # Split train/validation
    print("Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    feature_cols = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    
    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'label_encoders': label_encoders
    }

def save_processed_data(processed_data):
    """Save processed data to files"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Save datasets
    processed_data['X_train'].to_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'), index=False)
    processed_data['X_val'].to_csv(os.path.join(PROCESSED_DIR, 'X_val.csv'), index=False)
    processed_data['y_train'].to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
    processed_data['y_val'].to_csv(os.path.join(PROCESSED_DIR, 'y_val.csv'), index=False)
    
    # Save metadata
    metadata = {
        'feature_cols': processed_data['feature_cols'],
        'n_train': len(processed_data['X_train']),
        'n_val': len(processed_data['X_val']),
        'n_features': len(processed_data['feature_cols']),
        'processed_at': datetime.now().isoformat()
    }
    
    with open(os.path.join(PROCESSED_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Data saved to {PROCESSED_DIR}/")
    print(f"Train samples: {metadata['n_train']}, Val samples: {metadata['n_val']}")
    print(f"Features: {metadata['n_features']}")

def main():
    """Main execution"""
    print("="*50)
    print("AUTOMATED PREPROCESSING - House Prices Dataset")
    print("="*50)
    
    # Load data
    train_df, test_df = load_data()
    
    # Preprocess
    processed_data = preprocess_pipeline(train_df, test_df)
    
    # Save
    save_processed_data(processed_data)
    
    print("="*50)
    print("PREPROCESSING COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()

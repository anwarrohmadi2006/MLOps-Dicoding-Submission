"""
Baseline Modelling with MLflow Autolog
Author: Anwar-Rohmadi
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn

# Configuration
DATA_DIR = "house_prices_preprocessing"

def load_processed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv").values.ravel()
    return X_train, X_val, y_train, y_val

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}

def main():
    # Load data
    X_train, X_val, y_train, y_val = load_processed_data()
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Enable autolog
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="baseline_rf"):
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred)
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R2: {metrics['r2']:.4f}")
        
        print("Baseline model training complete!")

if __name__ == "__main__":
    main()

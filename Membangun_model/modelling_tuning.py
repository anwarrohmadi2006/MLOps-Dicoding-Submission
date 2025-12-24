"""
Advanced Modelling with Optuna Tuning + Manual Logging + DagsHub
Author: Anwar-Rohmadi
K2 ADVANCE: Manual logging, DagsHub online tracking, 2 extra artifacts
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub

# ============ CONFIGURATION ============
DATA_DIR = "house_prices_preprocessing"
DAGSHUB_REPO_OWNER = "anwarrohmadi2006"  # Actual DagsHub username
DAGSHUB_REPO_NAME = "Membangun_model"
N_TRIALS = 30
CV_FOLDS = 5

# ============ SETUP DAGSHUB ============
def setup_dagshub():
    """Initialize DagsHub MLflow tracking"""
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ============ DATA LOADING ============
def load_processed_data():
    """Load preprocessed data"""
    X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
    X_val = pd.read_csv(f"{DATA_DIR}/X_val.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_val = pd.read_csv(f"{DATA_DIR}/y_val.csv").values.ravel()
    return X_train, X_val, y_train, y_val

# ============ METRICS ============
def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoiding division by zero)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}

# ============ OPTUNA OBJECTIVE ============
def objective(trial, X_train, y_train):
    """Optuna objective function for hyperparameter tuning"""
    params = {
        'max_iter': trial.suggest_int('max_iter', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 32, 255),
        'random_state': 42
    }
    
    model = HistGradientBoostingRegressor(**params)
    
    # Cross-validation with RMSE
    scores = cross_val_score(
        model, X_train, y_train, 
        cv=CV_FOLDS, 
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    
    return -scores.mean()

# ============ EXTRA ARTIFACTS ============
def create_metrics_json(metrics, best_params, dataset_info):
    """Create metrics.json artifact"""
    artifact_data = {
        'metrics': metrics,
        'best_hyperparameters': best_params,
        'dataset_info': dataset_info,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'HistGradientBoostingRegressor',
        'tuning_method': 'Optuna',
        'cv_folds': CV_FOLDS,
        'n_trials': N_TRIALS
    }
    
    os.makedirs('artifacts', exist_ok=True)
    filepath = 'artifacts/metrics.json'
    with open(filepath, 'w') as f:
        json.dump(artifact_data, f, indent=2)
    
    return filepath

def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot artifact"""
    # Get permutation importance approximation from model
    importances = np.abs(model.feature_importances_) if hasattr(model, 'feature_importances_') else np.zeros(len(feature_names))
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    
    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], align='center', color='steelblue')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs('artifacts', exist_ok=True)
    filepath = 'artifacts/feature_importance.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    return filepath

# ============ MAIN TRAINING ============
def main():
    print("="*60)
    print("ADVANCED MODELLING WITH OPTUNA + DAGSHUB")
    print("="*60)
    
    # Setup DagsHub
    setup_dagshub()
    
    # Load data
    X_train, X_val, y_train, y_val = load_processed_data()
    feature_names = X_train.columns.tolist()
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize', study_name='house_prices_tuning')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_TRIALS, show_progress_bar=True)
    
    print(f"\nBest trial RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    # Train final model with best params
    best_params = study.best_params
    best_params['random_state'] = 42
    
    with mlflow.start_run(run_name="optuna_tuned_hgbr"):
        # MANUAL LOGGING - Parameters
        mlflow.log_param("model_type", "HistGradientBoostingRegressor")
        mlflow.log_param("tuning_method", "Optuna")
        mlflow.log_param("n_trials", N_TRIALS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train final model
        final_model = HistGradientBoostingRegressor(**best_params)
        final_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = final_model.predict(X_train)
        y_pred_val = final_model.predict(X_val)
        
        # Calculate metrics
        train_metrics = calculate_metrics(y_train, y_pred_train)
        val_metrics = calculate_metrics(y_val, y_pred_val)
        
        # MANUAL LOGGING - Metrics
        mlflow.log_metric("train_rmse", train_metrics['rmse'])
        mlflow.log_metric("train_mae", train_metrics['mae'])
        mlflow.log_metric("train_r2", train_metrics['r2'])
        mlflow.log_metric("train_mape", train_metrics['mape'])
        
        mlflow.log_metric("val_rmse", val_metrics['rmse'])
        mlflow.log_metric("val_mae", val_metrics['mae'])
        mlflow.log_metric("val_r2", val_metrics['r2'])
        mlflow.log_metric("val_mape", val_metrics['mape'])
        
        print(f"\nValidation Metrics:")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  MAE:  {val_metrics['mae']:.4f}")
        print(f"  R2:   {val_metrics['r2']:.4f}")
        print(f"  MAPE: {val_metrics['mape']:.2f}%")
        
        # Log model
        mlflow.sklearn.log_model(final_model, "model")
        
        # CREATE & LOG EXTRA ARTIFACTS
        dataset_info = {
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_features': len(feature_names)
        }
        
        # Artifact 1: metrics.json
        metrics_path = create_metrics_json(val_metrics, best_params, dataset_info)
        mlflow.log_artifact(metrics_path)
        print(f"Logged artifact: {metrics_path}")
        
        # Artifact 2: feature_importance.png
        importance_path = create_feature_importance_plot(final_model, feature_names)
        mlflow.log_artifact(importance_path)
        print(f"Logged artifact: {importance_path}")
        
        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow Run ID: {run_id}")
    
    print("="*60)
    print("TRAINING COMPLETE! Check DagsHub for results.")
    print("="*60)

if __name__ == "__main__":
    main()

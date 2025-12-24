# Membangun_model

## 📋 Deskripsi
Repository untuk **K2 - Membangun Model Machine Learning** pada submission Dicoding MLOps. Berisi training model dengan hyperparameter tuning menggunakan Optuna dan tracking dengan MLflow + DagsHub.

## 📁 Struktur Folder
```
Membangun_model/
├── modelling.py              # Basic model training
├── modelling_tuning.py       # Optuna hyperparameter tuning
├── house_prices_preprocessing/  # Preprocessed dataset
├── artifacts/
│   ├── feature_importance.png  # Feature importance chart
│   └── metrics.json            # Model metrics
├── DagsHub.txt               # Link ke DagsHub
├── requirements.txt          # Dependencies
└── screenshots/              # Bukti screenshots
```

## 🚀 Cara Menjalankan

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training dengan Optuna
```bash
python modelling_tuning.py
```

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| **Algorithm** | HistGradientBoostingRegressor |
| **RMSE** | $30,050 |
| **MAE** | $16,447 |
| **R²** | 0.88 |
| **Optuna Trials** | 30 |

## 🔧 Hyperparameters
```python
{
    'learning_rate': 0.05,
    'max_depth': 8,
    'max_iter': 300,
    'min_samples_leaf': 15,
    'l2_regularization': 0.1
}
```

## 📈 MLflow Tracking
Experiment tracking menggunakan DagsHub sebagai remote MLflow server.

## 👤 Author
**Anwar Rohmadi**

## 🔗 Links
- [DagsHub Experiments](https://dagshub.com/anwarrohmadi2006/Membangun_model/experiments)

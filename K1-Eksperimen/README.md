# Eksperimen_SML_Anwar-Rohmadi

## 📋 Deskripsi
Repository untuk **K1 - Eksperimentasi Data Science** pada submission Dicoding MLOps. Berisi pipeline preprocessing data untuk dataset House Prices.

## 📁 Struktur Folder
```
Eksperimen_SML_Anwar-Rohmadi/
├── .github/
│   └── workflows/
│       └── preprocess.yml       # GitHub Actions workflow
├── preprocessing/
│   ├── Eksperimen_Anwar-Rohmadi.ipynb  # EDA Notebook
│   ├── automate_Anwar-Rohmadi.py       # Automated preprocessing
│   └── run_notebook.py                  # Executable script
├── house_prices_raw/            # Raw dataset (train.csv, test.csv)
├── house_prices_preprocessing/  # Preprocessed output
├── config.py                    # Configuration
└── requirements.txt             # Dependencies
```

## 🚀 Cara Menjalankan

### Local
```bash
pip install -r requirements.txt
cd preprocessing
python automate_Anwar-Rohmadi.py
```

### GitHub Actions
Push ke repository untuk trigger workflow otomatis.

## 📊 Fitur Preprocessing
- Missing value handling (numeric: median, categorical: mode)
- Feature engineering (TotalSF, HouseAge, TotalBathrooms, dll)
- Outlier detection & removal
- Label encoding untuk categorical
- Train/validation split (80/20)

## 👤 Author
**Anwar Rohmadi**

## 🔗 Links
- [GitHub Repository](https://github.com/anwarrohmadi2006/Eksperimen_SML_Anwar-Rohmadi)

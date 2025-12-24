# Monitoring dan Logging

## 📋 Deskripsi
Repository untuk **K4 - Monitoring dan Logging** pada submission Dicoding MLOps. Berisi stack monitoring lengkap dengan Prometheus, Grafana, dan model serving menggunakan **model asli** (HistGradientBoostingRegressor).

## 📁 Struktur Folder
```
Monitoring dan Logging/
├── 1.bukti_serving/           # Screenshots health check
├── 2.prometheus.yml           # Prometheus configuration
├── 3.prometheus_exporter.py   # Model serving API + metrics
├── 4.bukti monitoring Prometheus/  # Prometheus screenshots
├── 5.bukti monitoring Grafana/     # Grafana screenshots
├── 6.bukti alerting Grafana/       # Alert rules screenshots
├── 7.Inference.py             # Load test script
├── grafana/
│   └── provisioning/
│       ├── dashboards/
│       │   └── model-monitoring.json  # Dashboard config
│       ├── datasources/
│       │   └── prometheus.yml
│       └── alerting/
│           └── alert-rules.yml  # Alert rules
├── docker-compose.yml         # Docker orchestration
├── Dockerfile.exporter        # Exporter Docker image
├── model.pkl                  # Trained HGBR model (703KB)
├── feature_names.json         # 80 feature names
├── model_info.json            # Model metrics
└── train_model.py             # Model training script
```

## 🚀 Cara Menjalankan

### 1. Start Docker Stack
```bash
docker-compose up -d
```

### 2. Akses Services
| Service | URL | Credentials |
|---------|-----|-------------|
| Model API | http://localhost:8000 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |

### 3. Run Load Test
```bash
python 7.Inference.py --requests 100 --workers 10
```

## 🤖 Model Serving API

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/health` | GET | Health check |
| `/predict` | POST | Predict house price |
| `/metrics` | GET | Prometheus metrics |
| `/model-info` | GET | Model information |

### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [60, 3, 68, 8450, 1, 0, 3, 0, 0, 0, 0, 5, 0, 0, 0, 1, 7, 5, 2003, 2003, 1, 0, 3, 3, 0, 196, 2, 1, 2, 2, 1, 0, 0, 706, 0, 0, 150, 856, 1, 2, 1, 4, 856, 854, 0, 1710, 1, 0, 2, 1, 3, 1, 2, 8, 0, 0, 0, 1, 2003, 2, 2, 548, 2, 2, 0, 0, 61, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2008, 0, 0, 2566]}'
```

## 📊 Grafana Dashboard
- **Dashboard Name:** anwarrohmadi111
- **Panels:** 12 panels
- **Metrics:**
  - Request Rate
  - Error Rate
  - Latency P50/P95
  - Prediction Statistics
  - Container Resources

## 🚨 Alert Rules
| Alert | Condition | Severity |
|-------|-----------|----------|
| High Latency P95 | >500ms | warning |
| High Error Rate | >5% | critical |
| High CPU Usage | >80% | warning |

## 🔧 Model Details
| Metric | Value |
|--------|-------|
| **Algorithm** | HistGradientBoostingRegressor |
| **Features** | 80 |
| **R²** | 0.8875 |
| **RMSE** | $28,963 |
| **MAE** | $16,447 |

## ⚠️ Penting
- **TIDAK ADA SIMULASI** - Semua prediksi menggunakan model asli
- Model file `model.pkl` harus ada untuk serving berjalan
- `7.Inference.py` menggunakan realistic house features

## 👤 Author
**Anwar Rohmadi**

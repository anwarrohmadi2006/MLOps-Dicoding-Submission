# 🏗️ House Price Prediction: End-to-End MLOps Ecosystem
**Author**: Anwar Rohmadi  
**Organization**: Dicoding Academy - Membangun Sistem Machine Learning  
**Project Objective**: Implementation of a production-grade MLOps pipeline for the Ames Housing dataset, covering the full lifecycle from data engineering to automated monitoring.

---

## 📋 Comprehensive Project Overview
This repository serves as the unified orchestration hub for a complete MLOps ecosystem. It demonstrates highly automated workflows, industrial-standard experiment tracking, and proactive performance monitoring.

## 📁 Repository Portfolio
The system is decomposed into four specialized modules, each representing a critical phase of the MLOps lifecycle:

| Component | Technical Scope | Focus Area |
|-----------|------------------|------------|
| [**Eksperimen_SML_Anwar-Rohmadi**](./Eksperimen_SML_Anwar-Rohmadi/) | K1 | Advanced EDA & Automated Preprocessing |
| [**Membangun_model**](./Membangun_model/) | K2 | Hyperparameter Optimization & MLflow Tracking |
| [**Workflow-CI**](./Workflow-CI/) | K3 | Continuous Integration & Containerization |
| [**Monitoring dan Logging**](./Monitoring%20dan%20Logging/) | K4 | Real-time Metrics & Proactive Alerting |

---

## 🛠️ Module Deep Dive

### 🔬 K1: Data Engineering & Analysis
- **EDA**: Multidimensional analysis identifying house price drivers (e.g., `GrLivArea`, `OverallQual`).
- **Automation**: Production-ready script `automate_Anwar-Rohmadi.py` for consistent data transformation.
- **CI**: GitHub Actions integrated to validate data pipeline integrity on every update.

### 🧪 K2: Model Development (Skilled)
- **Tracking**: Complete integration with **MLflow** and **DagsHub** for experiment auditability.
- **Algorithm**: `HistGradientBoostingRegressor` optimized via **Optuna** for superior accuracy.
- **Artifacts**: Systematic logging of parameters, metrics (RMSE/R²), and serialized model binaries.

### 🐳 K3: Continuous Integration (Advance)
- **Containerization**: Automatic transformation of MLflow models into **Docker images**.
- **DevOps**: Fully automated pipeline pushing validated versions to **Docker Hub**.
- **Registry**: [anwarrohmadi111784/mlops-dicoding-model](https://hub.docker.com/r/anwarrohmadi111784/mlops-dicoding-model)

### 📈 K4: Production Monitoring (Advance)
- **Serving**: Model deployed as a REST API with Prometheus-compatible metrics endpoints.
- **Observability**: **Grafana** dashboard featuring 12 operational and model-performance panels.
- **Alerting**: Automated triggers for high latency and error rate anomalies.
- **Scaling**: Ready for online monitoring via **Grafana Cloud** (see `MONITORING_ONLINE.md`).

---

## 🚀 Deployment & Operations
To initialize the monitoring stack locally:

```bash
cd "Monitoring dan Logging"
docker-compose up -d
```

**System Endpoints**:
- **Monitoring Portal**: `http://localhost:3000` (Grafana)
- **Metrics Explorer**: `http://localhost:9090` (Prometheus)
- **Prediction Engine**: `localhost:8000/predict` (Model API)

---
*This ecosystem demonstrates a complete, automated, and observable machine learning lifecycle, fulfilling all "Advance" criteria for the Dicoding MLOps certification.*

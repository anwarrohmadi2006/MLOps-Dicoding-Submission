# MLOps Dicoding Submission
**Author: Anwar-Rohmadi**

Proyek lengkap untuk submission kelas MLOps Dicoding.

## Struktur Folder

| Folder | Kriteria | Deskripsi |
|--------|----------|-----------|
| `K1-Eksperimen` | K1 | Eksperimen ML dengan preprocessing pipeline |
| `K2-Membangun_model` | K2 | Training model dengan MLflow & DagsHub |
| `K3-Workflow-CI` | K3 | CI/CD Pipeline dengan GitHub Actions & Docker |
| `K4-Monitoring` | K4 | Monitoring & Logging dengan Prometheus & Grafana |

## K1 - Eksperimen
- Preprocessing data House Prices
- GitHub Actions workflow (`preprocess.yml`)
- Download data dari Kaggle

## K2 - Membangun Model
- Model training dengan scikit-learn
- Experiment tracking dengan MLflow
- DagsHub integration

## K3 - Workflow CI/CD
- GitHub Actions untuk training otomatis
- Docker image build dan push ke DockerHub
- MLproject file untuk MLflow

## K4 - Monitoring
- Model serving dengan FastAPI
- Prometheus metrics exporter
- Grafana dashboard & alerting

## Links
- **DockerHub**: https://hub.docker.com/r/anwarrohmadi111784/house-prices-model
- **DagsHub**: https://dagshub.com/anwarrohmadi2006/Membangun_model

# 🏥 MLOps Complete Submission - House Price Prediction
**Author: Anwar-Rohmadi**

Selamat datang di repositori submission lengkap untuk proyek MLOps Dicoding. Repositori ini mencakup seluruh siklus hidup Machine Learning, mulai dari eksperimen data, pelacakan model, otomatisasi CI/CD, hingga pemantauan performa model di produksi.

---

## 📂 Struktur Proyek

| Folder | Kriteria | Deskripsi |
| :--- | :--- | :--- |
| [**`K1-Eksperimen/`**](./K1-Eksperimen) | **K1** | EDA Komprehensif dan Pipeline Preprocessing Otomatis. |
| [**`K2-Membangun_model/`**](./K2-Membangun_model) | **K2** | Training model dengan Pelacakan Eksperimen (MLflow & DagsHub). |
| [**`K3-Workflow-CI/`**](./K3-Workflow-CI) | **K3** | Otomatisasi Training & Build Gambar Docker (GitHub Actions). |
| [**`K4-Monitoring/`**](./K4-Monitoring) | **K4** | Serving Model, Prometheus Exporter, Grafana Dashboard & Alerting. |

---

## 🚀 Fitur Utama (Level Advance)

### 1. Preprocessing Otomatis (K1)
Kami menyediakan notebook EDA mendalam dan script `automate_Anwar-Rohmadi.py` yang secara otomatis membersihkan data, melakukan feature engineering, dan menyiapkan data train/val.

### 2. Experiment Tracking (K2)
Seluruh proses training dilacak menggunakan **MLflow** yang terintegrasi dengan **DagsHub**. Anda dapat melihat parameter, metrik (RMSE, MAE, R2), dan artifak model secara online.
- **DagsHub Dashboard:** [Link DagsHub](https://dagshub.com/anwarrohmadi2006/MLOps-Dicoding-Submission)

### 3. CI/CD Pipeline (K3)
Menggunakan GitHub Actions (`mlops-pipeline.yml`) untuk:
- Menjalankan preprocessing otomatis.
- Melatih model baru setiap ada perubahan kode.
- Membangun dan mengunggah image Docker secara otomatis ke Docker Hub.
- **Docker Image:** `anwarrohmadi111784/mlops-dicoding-model:latest`

### 4. Monitoring & Alerting (K4)
Model dijalankan menggunakan FastAPI dengan metrik Prometheus yang mencakup:
- Request latency (P95).
- Error rate.
- Prediksi statistik (Mean/Std).
- Pemanfaatan sumber daya sistem (CPU/Memory).
- **Online Monitoring:** Panduan lengkap tersedia [di sini](./K4-Monitoring/MONITORING_ONLINE.md) untuk menggunakan Grafana Cloud.

---

## 🛠️ Cara Menjalankan

### Persiapan
Clone repositori ini:
```bash
git clone https://github.com/anwarrohmadi2006/MLOps-Dicoding-Submission.git
cd MLOps-Dicoding-Submission
```

### Menjalankan Monitoring Lokal
Gunakan Docker Compose untuk menjalankan seluruh stack monitoring:
```bash
cd K4-Monitoring
docker-compose up -d
```
Akses Dashboard:
- **Grafana:** `http://localhost:3000` (User/Pass: admin/admin)
- **Prometheus:** `http://localhost:9090`
- **Model API:** `http://localhost:8000`

---

## 👨‍💻 Author
**Anwar-Rohmadi**  
Proyek ini dikembangkan sebagai syarat kelulusan kelas MLOps di Dicoding Academy.

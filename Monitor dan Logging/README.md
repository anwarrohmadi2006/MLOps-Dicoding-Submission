# K4 - Monitoring dan Logging
**Author: Anwar-Rohmadi**

Folder ini berisi sistem penyajian model (Model Serving) dan infrastruktur pemantauan performa.

## Isi Folder
- `3.prometheus_exporter.py`: API Server (FastAPI) yang mengekspos metrik Prometheus.
- `2.prometheus.yml`: Konfigurasi pengumpulan data Prometheus.
- `docker-compose.yml`: Orkestrasi Docker untuk Exporter, Prometheus, Grafana, dan cAdvisor.
- `grafana/`: Folder konfigurasi dashboard dan alerting untuk Grafana.
- `7.inference.py`: Script untuk melakukan simulasi beban (Load Testing).
- `MONITORING_ONLINE.md`: Panduan setup monitoring online di awan.

## Layanan & Port
| Layanan | Port | Deskripsi |
| :--- | :--- | :--- |
| **Model API** | `8000` | Endpoint `/predict` & `/metrics` |
| **Prometheus** | `9090` | GUI Query Metrik |
| **Grafana** | `3000` | Visualisasi Visual |
| **cAdvisor** | `8080` | Metrik Container Docker |

# K3 - Workflow CI/CD
**Author: Anwar-Rohmadi**

Folder ini berisi file-file yang diperlukan untuk menjalankan alur kerja integrasi dan pengiriman mandiri (CI/CD).

## Isi Folder
- `MLProject/MLproject`: File konfigurasi standard MLflow untuk reproduksibilitas.
- `MLProject/conda.yaml`: Definisi environment dependensi.
- `.github/workflows/mlops-pipeline.yml`: (Berada di root repositori) Definisi alur kerja GitHub Actions.
- `DockerHub.txt`: Tautan ke image model di Docker Hub.

## Otomatisasi
Setiap push ke branch `master` akan memicu:
1. Preprocessing Data.
2. Pelatihan Model.
3. Build Docker Image.
4. Push Image ke Docker Hub.

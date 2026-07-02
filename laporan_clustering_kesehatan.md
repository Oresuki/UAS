# Laporan Tugas Akhir: Preprocessing dan Unsupervised Learning (Clustering)
## Segmentasi Risiko Pasien Berdasarkan Data Kesehatan



## 1. Deskripsi Dataset

**Nama Dataset:** Stroke Prediction Dataset
**Sumber Asli:** Kaggle (dipublikasikan oleh fedesoriano), diakses melalui mirror publik di GitHub
**Jumlah Data Mentah:** 5.110 baris, 12 kolom

Dataset ini berisi data klinis pasien yang digunakan untuk memprediksi risiko stroke, mencakup informasi demografi (usia, jenis kelamin), riwayat medis (hipertensi, penyakit jantung), gaya hidup (status merokok, jenis pekerjaan), serta indikator kesehatan (kadar glukosa darah, BMI).

Dataset dipilih karena:
1. Relevan untuk kasus pengelompokan (segmentasi pasien berdasarkan profil risiko kesehatan)
2. Memiliki kecacatan data nyata yang perlu dibersihkan (dijelaskan di bagian 2)

## 2. Tahap Cleaning Data

### 2.1 Eksplorasi Awal (Sebelum Cleaning)
Sebelum membersihkan data, dilakukan pengecekan menyeluruh dan ditemukan:

| Temuan | Detail |
|---|---|
| Missing value | Kolom `bmi` memiliki nilai kosong (tertulis sebagai `N/A`) |
| Duplikat | Setelah dicek dengan `.duplicated()`, **tidak ditemukan** baris duplikat (0 baris) |
| Format tidak konsisten | Nama kolom `Residence_type` menggunakan huruf kapital, berbeda dari kolom lain yang huruf kecil semua |
| Kategori semu-missing | Kolom `smoking_status` memiliki kategori `"Unknown"` yang secara semantik adalah data tidak tersedia |
| Kategori tidak representatif | Kolom `gender` memiliki kategori `"Other"` dengan hanya 1 baris data |

> **Catatan kejujuran metodologis:** Dataset ini tidak mengandung baris duplikat. Proses pengecekan duplikat tetap dijalankan sebagai bagian standar dari pipeline cleaning (sesuai ketentuan tugas), dan hasilnya (nihil) dilaporkan apa adanya, bukan direkayasa.

### 2.2 Langkah Cleaning yang Dilakukan

| No | Langkah | Metode | Alasan |
|---|---|---|---|
| 1 | Menyeragamkan nama kolom | Mengubah semua nama kolom menjadi huruf kecil | Konsistensi format |
| 2 | Menangani missing value pada `bmi` | Imputasi dengan **median** | Median lebih tahan terhadap outlier dibanding mean |
| 3 | Menghapus baris duplikat | `drop_duplicates()` | Standar praktik cleaning (hasil: 0 baris terhapus) |
| 4 | Menghapus kategori `gender = "Other"` | Filtering | Hanya 1 baris, tidak representatif dan berpotensi menjadi outlier saat clustering |
| 5 | Menghapus kolom `id` | Drop kolom | Hanya nomor identitas, tidak relevan untuk analisis |
| 6 | Encoding kolom kategorikal (`smoking_status`) | Label Encoding | K-Means membutuhkan input numerik (distance-based algorithm) |
| 7 | Scaling seluruh fitur numerik | StandardScaler (mean=0, std=1) | Menyamakan skala antar fitur (usia 0–82 vs glukosa puluhan–ratusan) agar tidak bias |


## 3. Implementasi Clustering

**Algoritma yang digunakan:** K-Means Clustering

**Alasan pemilihan:**
- Mudah diimplementasikan dan diinterpretasikan (cocok untuk pemahaman dasar unsupervised learning)
- Efisien untuk dataset berukuran menengah (~5.000 baris)
- Cocok untuk fitur numerik hasil scaling yang digunakan dalam kasus ini

**Fitur yang digunakan untuk clustering:**
`age`, `hypertension`, `heart_disease`, `avg_glucose_level`, `bmi`, `smoking_status` (setelah encoding)

Kolom `stroke` **sengaja tidak digunakan** dalam proses clustering karena merupakan label/target yang seharusnya tidak diketahui pada skema unsupervised learning murni. Kolom ini baru digunakan setelah proses clustering selesai, sebagai validasi eksternal terhadap hasil pengelompokan.


## 4. Evaluasi dan Visualisasi

### 4.1 Penentuan Jumlah Cluster Optimal

| K | WCSS (Inertia) | Silhouette Score |
|---|---|---|
| 2 | 24265.29 | **0.3861** (tertinggi) |
| 3 | 19529.11 | 0.2290 |
| 4 | 15081.09 | 0.2925 |
| 5 | 12525.84 | 0.3230 |
| 6 | 10588.62 | 0.3225 |
| 7 | 9401.22 | 0.3262 |
| 8 | 8574.12 | 0.3137 |

**Pembahasan pemilihan K:**
Secara murni matematis, K=2 memiliki Silhouette Score tertinggi. Namun K=2 hanya memisahkan pasien menjadi dua kelompok sangat umum ("risiko rendah" vs "risiko tinggi"), yang kurang informatif untuk kebutuhan analisis segmentasi kesehatan yang lebih detail.

Grafik Elbow Method menunjukkan penurunan WCSS mulai melandai di sekitar **K=4**, dan Silhouette Score pada titik ini (0.29) masih tergolong wajar (skor di atas 0.25 umumnya menunjukkan struktur cluster yang cukup jelas). Oleh karena itu, dipilih **K=4** sebagai kompromi antara kualitas statistik cluster dan kegunaan praktis hasil segmentasi.

Silhouette Score akhir untuk K=4: **0.2925**

### 4.2 Visualisasi
Notebook menyediakan dua jenis visualisasi:
1. **Scatter plot 2D** antara `age` dan `avg_glucose_level`, diwarnai berdasarkan cluster
2. **Visualisasi PCA** yang mereduksi 6 fitur menjadi 2 komponen utama untuk melihat pemisahan cluster secara keseluruhan

### 4.3 Karakteristik Tiap Cluster

| Cluster | Jumlah Pasien | Usia Rata-rata | Hipertensi | Penyakit Jantung | Glukosa Rata-rata | BMI Rata-rata | % Riwayat Stroke |
|---|---|---|---|---|---|---|---|
| **0** | 2.986 | 50.0 | 0% | 0% | 106.9 | 31.1 | 4.86% |
| **1** | 1.413 | 18.5 | 0% | 0% | 92.1 | 22.6 | 0.28% |
| **2** | 434 | 61.0 | 100% | 0% | 127.1 | 32.8 | 12.21% |
| **3** | 276 | 68.2 | 23% | 100% | 136.8 | 30.1 | 17.03% |

**Interpretasi tiap cluster:**
- **Cluster 1 — Usia Muda, Risiko Sangat Rendah:** Rata-rata usia 18,5 tahun, tidak ada riwayat hipertensi/penyakit jantung, BMI normal. Kelompok dengan persentase stroke terendah (0,28%).
- **Cluster 0 — Dewasa, Risiko Rendah-Sedang:** Usia rata-rata 50 tahun, indikator kesehatan relatif normal, namun BMI sedikit di atas normal. Persentase stroke 4,86%.
- **Cluster 2 — Lansia dengan Hipertensi:** Seluruh pasien di cluster ini memiliki hipertensi, usia rata-rata 61 tahun, glukosa dan BMI lebih tinggi. Persentase stroke naik menjadi 12,21%.
- **Cluster 3 — Risiko Tertinggi:** Usia rata-rata tertua (68 tahun), seluruh pasien memiliki penyakit jantung, glukosa rata-rata tertinggi. Persentase stroke tertinggi di antara semua cluster (17,03%).

### 4.4 Validasi Hasil Clustering
Meskipun kolom `stroke` tidak digunakan dalam proses clustering, terdapat pola yang **konsisten dan meningkat secara bertahap** dari Cluster 1 → 0 → 2 → 3 (0,28% → 4,86% → 12,21% → 17,03%). Pola ini selaras dengan pengetahuan medis umum bahwa usia, hipertensi, penyakit jantung, dan kadar glukosa tinggi merupakan faktor risiko stroke. Konsistensi ini menjadi validasi bahwa hasil clustering yang dihasilkan **masuk akal secara klinis**, bukan sekadar pengelompokan acak secara statistik.



## 5. Kesimpulan

1. Proses cleaning berhasil menangani missing value (imputasi median pada `bmi`), memverifikasi ketiadaan duplikat, menyeragamkan format nama kolom, dan membersihkan kategori tidak representatif.
2. Algoritma K-Means dengan K=4 menghasilkan 4 segmen pasien dengan karakteristik risiko kesehatan yang berbeda secara jelas.
3. Segmentasi yang dihasilkan **selaras dengan pengetahuan medis** dan tervalidasi oleh pola persentase riwayat stroke yang meningkat secara konsisten antar cluster.
4. Hasil ini dapat dimanfaatkan misalnya oleh fasilitas kesehatan untuk memprioritaskan program skrining atau edukasi kesehatan pada kelompok berisiko tinggi (Cluster 2 dan 3).

### Keterbatasan
- Jumlah kasus stroke dalam dataset relatif sedikit dan tidak seimbang (imbalanced) terhadap kasus non-stroke, sehingga persentase pada cluster kecil (misal Cluster 3 dengan 276 pasien) perlu dibaca secara hati-hati.
- Pemilihan K=4 melibatkan pertimbangan interpretatif, bukan murni angka statistik tertinggi, sehingga nilai K alternatif (misalnya K=2 atau K=5) tetap valid untuk dieksplorasi lebih lanjut.
- Dataset bersifat cross-sectional (potret satu waktu), sehingga tidak menangkap perubahan kondisi kesehatan pasien dari waktu ke waktu.




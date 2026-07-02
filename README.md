# Remote Work Mental Health Analyzer 🏠

Ini adalah project UAS Machine Learning yang mencoba melihat gimana pola kerja remote bisa berhubungan sama kondisi mental seseorang. Daripada cuma bikin model terus taruh di notebook, kelompok kami bikin dashboard interaktifnya juga pakai Streamlit biar hasilnya bisa langsung dicoba, nggak cuma dibaca.

Intinya, dari beberapa data yang diisi user (umur, jam kerja, posisi, dukungan kantor, dll), sistem ini bakal memprediksi tiga hal sekaligus:

- Kualitas tidur
- Tingkat stres
- Kondisi kesehatan mental secara umum

Ketiganya diprediksi bareng dalam satu kali proses, karena model yang dipakai adalah Multi-Output Classification — jadi bukan tiga model terpisah, tapi satu model yang bisa ngeluarin tiga output berbeda.

## Coba Sendiri

App-nya bisa langsung diakses di:
👉 [project-uas-machine-learning.streamlit.app](https://project-uas-machine-learning.streamlit.app)

## Apa yang Bisa Dilakukan

Ada 3 tab utama di dashboard:

**Formulir Prediksi** — di sini user isi data-data terkait dirinya (jam kerja per minggu, pengalaman kerja, work-life balance, isolasi sosial, jumlah meeting virtual, dan lain-lain), lalu sistem langsung kasih hasil prediksi tiga variabel di atas.

**Analitik Data** — nampilin eksplorasi dari dataset yang dipakai buat training, mulai dari distribusi stress level, heatmap korelasi antar variabel numerik, sampai tabel data mentahnya yang bisa difilter dan di-download jadi CSV.

**Dokumentasi Sistem** — penjelasan singkat soal arsitektur model dan info dataset, buat yang penasaran cara kerjanya di balik layar.

## Cara Kerjanya

Alurnya cukup standar untuk aplikasi ML berbasis Streamlit:

1. User isi form di dashboard
2. Data yang masuk divalidasi dan diproses dulu (encoding + scaling) pakai encoder dan scaler yang sama persis dengan yang dipakai waktu training
3. Data yang sudah diproses masuk ke model buat diprediksi
4. Hasil prediksi tiga variabel ditampilkan ke user

## Parameter Input

Data yang perlu diisi user dibagi jadi beberapa kelompok:

*Demografi & pekerjaan* — umur, pengalaman kerja, jam kerja per minggu, posisi kerja

*Kesejahteraan* — work-life balance, tingkat isolasi sosial, dukungan perusahaan, jumlah virtual meeting

*Lainnya* — akses ke fasilitas kesehatan mental, perubahan produktivitas, kepuasan kerja remote

## Tech Stack

- Python
- Streamlit (buat dashboard-nya)
- Scikit-learn (training model)
- Pandas & NumPy (olah data)
- Plotly (visualisasi di tab Analitik Data)
- Joblib (nyimpen model, encoder, scaler)

## Struktur Project

```
├── dashboard.py                              # aplikasi Streamlit
├── projek_kel1_ML.ipynb                      # notebook eksplorasi & training model
├── model_multi_output.pkl                    # model hasil training
├── encoder.pkl                               # encoder buat fitur kategorikal
├── scaler.pkl                                # scaler buat fitur numerik
├── Impact_of_Remote_Work_on_Mental_Health.csv # dataset
└── requirements.txt
```

## Menjalankan di Lokal

```bash
git clone <repo-ini>
cd project-uas-machine-learning
pip install -r requirements.txt
streamlit run dashboard.py
```

## Catatan

Project ini dibuat untuk tujuan pembelajaran (tugas UAS mata kuliah Machine Learning), jadi hasil prediksinya sebaiknya nggak dijadikan acuan medis atau psikologis yang serius. Kalau memang lagi ngalamin masalah kesehatan mental, tetap lebih baik konsultasi ke profesional ya.

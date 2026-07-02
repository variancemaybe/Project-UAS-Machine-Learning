🏠 Remote Work Mental Health Analyzer

Remote Work Mental Health Analyzer merupakan aplikasi berbasis Machine Learning yang dirancang untuk menganalisis dampak sistem kerja jarak jauh (remote work) terhadap kondisi kesehatan mental seseorang. 
Aplikasi ini membantu pengguna memprediksi tiga aspek utama kesejahteraan berdasarkan data demografi, pola kerja, dan faktor pendukung lainnya.

Model yang digunakan merupakan Multi-Output Classification, sehingga dalam satu kali proses prediksi aplikasi dapat menghasilkan beberapa output secara bersamaan, yaitu:

😴 Kualitas Tidur (Sleep Quality)
😟 Tingkat Stres (Stress Level)
🧠 Kondisi Mental (Mental Health Condition)

Pengguna hanya perlu mengisi beberapa parameter seperti usia, pengalaman kerja, jam kerja mingguan, posisi pekerjaan, keseimbangan kehidupan dan pekerjaan (Work-Life Balance), tingkat isolasi sosial, dukungan perusahaan, jumlah rapat virtual, akses terhadap fasilitas kesehatan mental, perubahan produktivitas, serta tingkat kepuasan bekerja secara remote. Selanjutnya sistem akan melakukan proses prediksi menggunakan model Machine Learning yang telah dilatih dari dataset.

✨ Fitur Utama
📊 Prediksi tiga kondisi kesehatan secara bersamaan menggunakan Multi-Output Classification.
👤 Form input interaktif untuk data demografi dan pekerjaan.
⚖️ Analisis faktor kesejahteraan seperti Work-Life Balance, Social Isolation, dan Company Support.
🏢 Evaluasi faktor tambahan seperti akses layanan kesehatan mental, perubahan produktivitas, dan kepuasan kerja remote.
📈 Tampilan antarmuka modern berbasis Streamlit.
📚 Dokumentasi sistem yang menjelaskan cara penggunaan aplikasi.


🛠️ Teknologi yang Digunakan
Python
Streamlit
Scikit-Learn
Pandas
NumPy
Joblib
Machine Learning (Multi-Output Classification)


⚙️ Cara Kerja Sistem
Pengguna mengisi seluruh data pada formulir.
Data divalidasi dan diproses sesuai format yang dibutuhkan model.
Data dipreprocessing menggunakan encoder dan scaler yang telah disimpan saat proses pelatihan model.
Model Machine Learning melakukan prediksi terhadap tiga target sekaligus.
Hasil analisis ditampilkan kepada pengguna sebagai informasi kondisi kesehatan mental berdasarkan karakteristik dan kebiasaan kerja remote.


📌 Parameter yang Digunakan
👤 Demografi & Pekerjaan
Umur
Pengalaman Kerja
Jam Kerja per Minggu
Posisi Pekerjaan
⚖️ Metrik Kesejahteraan
Work-Life Balance
Tingkat Isolasi Sosial
Dukungan Perusahaan
Jumlah Virtual Meeting
🏥 Faktor Tambahan
Akses Fasilitas Kesehatan Mental
Perubahan Produktivitas
Kepuasan terhadap Remote Work


🎯 Tujuan Pengembangan

Aplikasi ini dikembangkan sebagai implementasi Machine Learning untuk membantu mengidentifikasi potensi dampak kerja jarak jauh terhadap kesejahteraan mental. Hasil prediksi diharapkan dapat menjadi informasi awal bagi individu maupun organisasi dalam meningkatkan kualitas lingkungan kerja, keseimbangan hidup, serta dukungan terhadap kesehatan mental karyawan.

💡 Keunggulan Aplikasi
Prediksi tiga variabel sekaligus dalam satu proses menggunakan Multi-Output Classification.
Antarmuka sederhana, responsif, dan mudah digunakan.
Proses analisis berlangsung secara cepat melalui model yang telah dilatih sebelumnya.
Memanfaatkan berbagai faktor yang berpengaruh terhadap kesejahteraan pekerja remote sehingga hasil prediksi lebih komprehensif.
Cocok sebagai implementasi nyata penerapan Machine Learning pada bidang kesehatan mental dan analisis sumber daya manusia (Human Resources Analytics).

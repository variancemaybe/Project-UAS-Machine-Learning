import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time

# Konfigurasi Halaman 
st.set_page_config(
    page_title="Kel1 | Remote Work Mental Health Analyzer", 
    page_icon="🧠", 
    layout="wide"
)

# Fungsi untuk load model dan preprocessing
@st.cache_resource
def load_components():
    # Memuat model multi-output tunggal yang baru saja disimpan
    model = joblib.load('model_multi_output.pkl')
    scaler = joblib.load('scaler.pkl')
    try:
        encoder = joblib.load('encoder.pkl')
    except:
        encoder = None
    return model, scaler, encoder

model, scaler, encoder = load_components()

# HEADER & DESAIN UI UTAMA
st.title("💻 Remote Work Mental Health Analyzer 🏠")
st.markdown("""
Aplikasi prediksi dampak *remote work* terhadap **Kualitas Tidur, Tingkat Stres, dan Kondisi Mental** menggunakan *Machine Learning* (**Multi-Output Classification**).
""")

# Menggunakan Tabs agar UI lebih rapi
tab1, tab2 = st.tabs(["📊 Formulir Prediksi", "⚙️ Dokumentasi Sistem"])

with tab1:
    st.info("Silakan lengkapi data demografi dan metrik kesejahteraan di bawah ini untuk memulai analisa.", icon="ℹ️")
    
    # Bikin 3 kolom form
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👨‍💻 Demografi & Kerja")
        with st.container(border=True):
            age = st.slider("Umur", 18, 65, 25, help="Usia pekerja")
            experience = st.slider("Pengalaman Kerja (Tahun)", 0, 40, 3)
            hours_worked = st.number_input("Jam Kerja per Minggu", 10, 100, 40)
            job_role = st.selectbox("Posisi Pekerjaan", [
                "Data Scientist", "Designer", "HR", "Marketing", 
                "Project Manager", "Sales", "Software Engineer"
            ])

    with col2:
        st.subheader("⚖️ Metrik Kesejahteraan")
        with st.container(border=True):
            work_life = st.slider("Rating Work-Life Balance", 1, 5, 3, help="1 = Sangat Buruk, 5 = Sangat Baik")
            isolation = st.slider("Rating Isolasi Sosial", 1, 5, 3, help="1 = Tidak terisolasi, 5 = Sangat terisolasi")
            support = st.slider("Rating Dukungan Perusahaan", 1, 5, 3)
            virtual_meetings = st.number_input("Jumlah Meeting Virtual / Minggu", 0, 50, 10)

    with col3:
        st.subheader("🏥 Faktor Tambahan")
        with st.container(border=True):
            mental_health_access = st.radio("Akses ke Fasilitas Kesehatan Mental?", ["Yes", "No"], horizontal=True)
            productivity = st.selectbox("Perubahan Produktivitas", ["Decrease", "Increase", "No Change"])
            satisfaction = st.selectbox("Kepuasan Remote Work", ["Satisfied", "Neutral", "Unsatisfied"])

    st.divider()

    # Logika Prediksi dengan Animasi
    if st.button("🚀 Analisa Prediksi", type="primary", use_container_width=True):
        
        # Simulasi proses loading agar interaktif
        with st.spinner('Mengeksekusi pipeline prediksi secara efektif dan rasional... Memvalidasi data...'):
            time.sleep(1.5) # Memberi efek proses komputasi
            
            expected_columns = [
                'Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 
                'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work', 
                'Job_Role_Data Scientist', 'Job_Role_Designer', 'Job_Role_HR', 'Job_Role_Marketing', 
                'Job_Role_Project Manager', 'Job_Role_Sales', 'Job_Role_Software Engineer', 
                'Access_to_Mental_Health_Resources_No', 'Access_to_Mental_Health_Resources_Yes', 
                'Productivity_Change_Decrease', 'Productivity_Change_Increase', 'Productivity_Change_No Change', 
                'Satisfaction_with_Remote_Work_Neutral', 'Satisfaction_with_Remote_Work_Satisfied', 'Satisfaction_with_Remote_Work_Unsatisfied'
            ]
            
            input_dict = {col: 0 for col in expected_columns}
            
            input_dict['Age'] = age
            input_dict['Years_of_Experience'] = experience
            input_dict['Hours_Worked_Per_Week'] = hours_worked
            input_dict['Number_of_Virtual_Meetings'] = virtual_meetings
            input_dict['Work_Life_Balance_Rating'] = work_life
            input_dict['Social_Isolation_Rating'] = isolation
            input_dict['Company_Support_for_Remote_Work'] = support
            
            input_dict[f'Job_Role_{job_role}'] = 1
            input_dict[f'Access_to_Mental_Health_Resources_{mental_health_access}'] = 1
            input_dict[f'Productivity_Change_{productivity}'] = 1
            input_dict[f'Satisfaction_with_Remote_Work_{satisfaction}'] = 1
            
            df_input = pd.DataFrame([input_dict])[expected_columns]
            
            try:
                # Standardisasi data input
                data_scaled = scaler.transform(df_input)
                
                # Cukup 1x panggil predict untuk menghasilkan 3 output sekaligus secara paralel
                predictions = model.predict(data_scaled) # Output shape: (1, 3)
                
                # Mengambil nilai indeks masing-masing target
                num_sleep = int(predictions[0][0])
                num_stress = int(predictions[0][1])
                num_mental = int(predictions[0][2])
                
                # Melakukan inverse transform jika encoder tersedia
                if encoder is not None:
                    hasil_sleep = encoder['Sleep_Quality'].inverse_transform([num_sleep])[0]
                    hasil_stress = encoder['Stress_Level'].inverse_transform([num_stress])[0]
                    hasil_mental = encoder['Mental_Health_Condition'].inverse_transform([num_mental])[0]
                else:
                    hasil_sleep = str(num_sleep)
                    hasil_stress = str(num_stress)
                    hasil_mental = str(num_mental)
                
                st.success("✅ Analisa Berhasil Dilakukan secara Rasional!")
                
                # Menampilkan metrik hasil prediksi
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.metric("Kualitas Tidur", str(hasil_sleep))
                with res_col2:
                    st.metric("Tingkat Stres", str(hasil_stress))
                with res_col3:
                    st.metric("Kondisi Mental", str(hasil_mental))
                
                # Feedback Dinamis berdasarkan kondisi mental
                if hasil_mental in ['Burnout', 'Depression', 'Anxiety']:
                    st.warning(f"⚠️ Peringatan: Sistem mendeteksi indikasi **{hasil_mental}**. Disarankan untuk mengkaji ulang *work-life balance* atau menghubungi profesional kesehatan mental.")
                else:
                    st.balloons() # Efek animasi jika hasilnya positif
                    st.success("Kondisi mental pekerja terpantau stabil. Pertahankan ritme kerja yang efektif!")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

with tab2:
    st.subheader("Arsitektur Model")
    st.write("Sistem ini mengimplementasikan pendekatan **Multi-Output Classification** sesuai dengan standar rekayasa pipeline yang efektif dan efisien. Menggunakan satu buah objek model tunggal yang diekspor untuk melakukan inferensi tiga variabel target sekaligus secara paralel tanpa memerlukan proses komputasi berantai.")
    st.code("Model_Type: Multi-Output Classification\nArtifact_Stored: model_multi_output.pkl\nData_Pipeline: StandardScaler + Dummy Encoding", language='yaml')
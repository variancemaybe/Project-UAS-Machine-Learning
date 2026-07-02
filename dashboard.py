import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Kel1 | Remote Work Mental Health Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS — TAMPILAN MODERN
st.markdown("""
<style>
    .main { background-color: #0e1117; }

    .hero {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #DB2777 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(124,58,237,0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: rgba(255,255,255,0.9);
        font-size: 1.02rem;
        margin: 0;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2130, #262b3d);
        border: 1px solid rgba(124,58,237,0.25);
        padding: 1rem 1rem 0.6rem 1rem;
        border-radius: 14px;
    }
    div[data-testid="stMetricLabel"] { font-weight: 600; }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
        border-left: 5px solid #7C3AED;
        padding-left: 0.6rem;
    }

    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        height: 3em;
    }

    div[data-testid="stTabs"] button {
        font-size: 1.02rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL & DATA (CACHED)
@st.cache_resource
def load_components():
    model = joblib.load('model_multi_output.pkl')
    scaler = joblib.load('scaler.pkl')
    try:
        encoder = joblib.load('encoder.pkl')
    except Exception:
        encoder = None
    return model, scaler, encoder

@st.cache_data
def load_data():
    df = pd.read_csv('Impact_of_Remote_Work_on_Mental_Health.csv')
    return df

model, scaler, encoder = load_components()
df = load_data()

PALETTE = ["#7C3AED", "#4F46E5", "#DB2777", "#059669", "#F59E0B", "#0EA5E9"]

# HERO HEADER
st.markdown("""
<div class="hero">
    <h1>💻 Remote Work Mental Health Analyzer 🏠</h1>
    <p>Aplikasi prediksi dampak <b>remote work</b> terhadap Kualitas Tidur, Tingkat Stres, dan Kondisi
    Mental menggunakan Machine Learning (Multi-Output Classification), dilengkapi dashboard analitik data.</p>
</div>
""", unsafe_allow_html=True)

# TABS
tab1, tab2, tab3 = st.tabs(["📊 Formulir Prediksi", "📈 Analitik Data", "⚙️ Dokumentasi Sistem"])

# TAB 1 — FORMULIR PREDIKSI
with tab1:
    st.info("Silakan lengkapi data demografi dan metrik kesejahteraan di bawah ini untuk memulai analisa.", icon="ℹ️")

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

    if st.button("🚀 Analisa Prediksi", type="primary", use_container_width=True):

        with st.spinner('Mengeksekusi pipeline prediksi... Memvalidasi data...'):
            time.sleep(1.2)

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
                data_scaled = pd.DataFrame(scaler.transform(df_input), columns=expected_columns)
                predictions = model.predict(data_scaled)

                num_sleep = int(predictions[0][0])
                num_stress = int(predictions[0][1])
                num_mental = int(predictions[0][2])

                if encoder is not None:
                    hasil_sleep = encoder['Sleep_Quality'].inverse_transform([num_sleep])[0]
                    hasil_stress = encoder['Stress_Level'].inverse_transform([num_stress])[0]
                    hasil_mental = encoder['Mental_Health_Condition'].inverse_transform([num_mental])[0]
                else:
                    hasil_sleep = str(num_sleep)
                    hasil_stress = str(num_stress)
                    hasil_mental = str(num_mental)

                st.success("✅ Analisa Berhasil Dilakukan!")

                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("😴 Kualitas Tidur", str(hasil_sleep))
                with res_col2:
                    st.metric("😟 Tingkat Stres", str(hasil_stress))
                with res_col3:
                    st.metric("🧠 Kondisi Mental", str(hasil_mental))

                if hasil_mental in ['Burnout', 'Depression', 'Anxiety']:
                    st.warning(f"⚠️ Peringatan: Sistem mendeteksi indikasi **{hasil_mental}**. Disarankan untuk mengkaji ulang *work-life balance* atau menghubungi profesional kesehatan mental.")
                else:
                    st.balloons()
                    st.success("Kondisi mental pekerja terpantau stabil. Pertahankan ritme kerja yang efektif!")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses data: {e}")

# TAB 2 — ANALITIK DATA
with tab2:
    st.markdown('<div class="section-title">📌 Ringkasan Dataset</div>', unsafe_allow_html=True)

    n_rows, n_cols = df.shape
    total_missing = int(df.isna().sum().sum())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jumlah Data", f"{n_rows:,}")
    m2.metric("Jumlah Kolom", f"{n_cols}")
    m3.metric("Rata-rata Umur", f"{df['Age'].mean():.1f} th")
    m4.metric("Data Kosong (Missing)", f"{total_missing:,}")

    st.divider()

    # ---------- FILTER ----------
    st.markdown('<div class="section-title">🔎 Filter Data</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        filter_region = st.multiselect("Region", sorted(df['Region'].dropna().unique()))
    with f2:
        filter_role = st.multiselect("Posisi Pekerjaan", sorted(df['Job_Role'].dropna().unique()))
    with f3:
        filter_location = st.multiselect("Lokasi Kerja", sorted(df['Work_Location'].dropna().unique()))
    with f4:
        filter_stress = st.multiselect("Tingkat Stres", sorted(df['Stress_Level'].dropna().unique()))

    df_filtered = df.copy()
    if filter_region:
        df_filtered = df_filtered[df_filtered['Region'].isin(filter_region)]
    if filter_role:
        df_filtered = df_filtered[df_filtered['Job_Role'].isin(filter_role)]
    if filter_location:
        df_filtered = df_filtered[df_filtered['Work_Location'].isin(filter_location)]
    if filter_stress:
        df_filtered = df_filtered[df_filtered['Stress_Level'].isin(filter_stress)]

    st.caption(f"Menampilkan **{len(df_filtered):,}** dari **{n_rows:,}** total data sesuai filter yang dipilih.")

    st.divider()

    # ---------- DISTRIBUSI TARGET UTAMA ----------
    st.markdown('<div class="section-title">🧠 Distribusi Kondisi Kesehatan Mental</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        vc = df_filtered['Stress_Level'].value_counts().reset_index()
        vc.columns = ['Stress_Level', 'Jumlah']
        fig = px.pie(vc, names='Stress_Level', values='Jumlah', hole=0.5,
                     title="Tingkat Stres", color_discrete_sequence=PALETTE)
        fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, width='stretch')

    with c2:
        vc = df_filtered['Sleep_Quality'].value_counts().reset_index()
        vc.columns = ['Sleep_Quality', 'Jumlah']
        fig = px.pie(vc, names='Sleep_Quality', values='Jumlah', hole=0.5,
                     title="Kualitas Tidur", color_discrete_sequence=PALETTE)
        fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, width='stretch')

    with c3:
        vc = df_filtered['Mental_Health_Condition'].fillna('Tidak Ada').value_counts().reset_index()
        vc.columns = ['Mental_Health_Condition', 'Jumlah']
        fig = px.pie(vc, names='Mental_Health_Condition', values='Jumlah', hole=0.5,
                     title="Kondisi Mental", color_discrete_sequence=PALETTE)
        fig.update_layout(margin=dict(t=50, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig, width='stretch')

    # ---------- DEMOGRAFI & PEKERJAAN ----------
    st.markdown('<div class="section-title">👥 Demografi & Karakteristik Pekerjaan</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)

    with d1:
        fig = px.histogram(df_filtered, x='Age', nbins=25, title="Distribusi Umur Pekerja",
                            color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=350, bargap=0.05)
        st.plotly_chart(fig, width='stretch')

    with d2:
        vc = df_filtered['Job_Role'].value_counts().reset_index()
        vc.columns = ['Job_Role', 'Jumlah']
        fig = px.bar(vc.sort_values('Jumlah'), x='Jumlah', y='Job_Role', orientation='h',
                     title="Jumlah Data per Posisi Pekerjaan", color='Jumlah',
                     color_continuous_scale=PALETTE)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')

    d3, d4 = st.columns(2)
    with d3:
        vc = df_filtered['Work_Location'].value_counts().reset_index()
        vc.columns = ['Work_Location', 'Jumlah']
        fig = px.bar(vc, x='Work_Location', y='Jumlah', title="Distribusi Lokasi Kerja",
                     color='Work_Location', color_discrete_sequence=PALETTE)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with d4:
        vc = df_filtered['Region'].value_counts().reset_index()
        vc.columns = ['Region', 'Jumlah']
        fig = px.bar(vc.sort_values('Jumlah'), x='Jumlah', y='Region', orientation='h',
                     title="Jumlah Data per Region", color='Jumlah',
                     color_continuous_scale=PALETTE)
        fig.update_layout(height=350)
        st.plotly_chart(fig, width='stretch')

    # ---------- HUBUNGAN ANTAR VARIABEL ----------
    st.markdown('<div class="section-title">📊 Hubungan Work-Life Balance & Isolasi Sosial dengan Stres</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        agg = df_filtered.groupby('Stress_Level')['Work_Life_Balance_Rating'].mean().reset_index()
        fig = px.bar(agg, x='Stress_Level', y='Work_Life_Balance_Rating',
                     title="Rata-rata Work-Life Balance per Tingkat Stres",
                     color='Stress_Level', color_discrete_sequence=PALETTE)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with e2:
        agg = df_filtered.groupby('Stress_Level')['Social_Isolation_Rating'].mean().reset_index()
        fig = px.bar(agg, x='Stress_Level', y='Social_Isolation_Rating',
                     title="Rata-rata Isolasi Sosial per Tingkat Stres",
                     color='Stress_Level', color_discrete_sequence=PALETTE)
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # ---------- HEATMAP KORELASI NUMERIK ----------
    st.markdown('<div class="section-title">🔗 Korelasi Antar Variabel Numerik</div>', unsafe_allow_html=True)
    numeric_cols = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings',
                    'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work']
    corr = df_filtered[numeric_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='Purples', aspect='auto',
                     title="Heatmap Korelasi")
    fig.update_layout(height=450)
    st.plotly_chart(fig, width='stretch')

    # ---------- TABEL DATA ----------
    st.markdown('<div class="section-title">📋 Tampilan Data</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered, use_container_width=True, height=380)

    csv_download = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Unduh Data (CSV)",
        data=csv_download,
        file_name="remote_work_mental_health_filtered.csv",
        mime="text/csv",
        use_container_width=True
    )

# TAB 3 — DOKUMENTASI SISTEM
with tab3:
    st.subheader("Arsitektur Model")
    st.write("Sistem ini mengimplementasikan pendekatan **Multi-Output Classification** menggunakan satu buah objek model tunggal yang diekspor untuk melakukan inferensi tiga variabel target sekaligus secara paralel.")
    st.code("Model_Type: Multi-Output Classification\nArtifact_Stored: model_multi_output.pkl\nData_Pipeline: StandardScaler + Dummy Encoding", language='yaml')

    st.divider()
    st.subheader("Tentang Dataset")
    st.write(f"Dataset yang digunakan berisi **{df.shape[0]:,} baris** dan **{df.shape[1]} kolom**, mencakup data demografi, pola kerja, dan faktor kesejahteraan dari pekerja remote di berbagai region.")
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ['Kolom', 'Tipe Data']
    st.dataframe(dtypes_df, use_container_width=True)
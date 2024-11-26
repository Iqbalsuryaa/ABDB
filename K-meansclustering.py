import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf

# Fungsi untuk setiap menu
def home():
    # Menampilkan Header/Banner
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Header Banner" width="800">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Judul dan Deskripsi
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")    

    # Menampilkan Abstrak
    st.subheader("Abstrak")
    st.write("""
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
    """)

    # Menampilkan Gambar Arsitektur Sistem
    st.subheader("Arsitektur Sistem")
    # Menampilkan gambar menggunakan HTML
    st.markdown(
        """
        <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Gambar Hider" width="700">
        """,
        unsafe_allow_html=True,
    )

    # Penjelasan Arsitektur Sistem
    st.write("""
        Arsitektur sistem ini menggambarkan alur kerja aplikasi dari pengambilan data,
        preprocessing, hingga analisis. Data curah hujan diolah menggunakan beberapa
        metode untuk menghasilkan prediksi yang akurat. Komponen utama meliputi:
        - **Pengumpulan Data:** Data cuaca harian dari BMKG atau citra awan.
        - **Preprocessing:** Normalisasi data, augmentasi gambar, dan transformasi fitur.
        - **Model Analitik:** Penggunaan algoritma ARIMA untuk data waktu, CNN untuk klasifikasi gambar,
          dan clustering dengan K-Means untuk pengelompokan data.
        - **Output:** Prediksi cuaca atau rekomendasi tindakan untuk sektor pertanian.
    """)

# Fungsi lain tetap sama seperti sebelumnya...
def prediksi_arima():
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini akan memproses data curah hujan menggunakan model ARIMA.")
    # Data Dummy
    data = {"Bulan": ["Jan", "Feb", "Mar", "Apr", "Mei"],
            "Curah Hujan": [100, 150, 200, 250, 300]}
    df = pd.DataFrame(data)
    st.dataframe(df)

def klasifikasi_cnn():
    st.title("Klasifikasi Citra Awan dengan Metode CNN")
    uploaded_file = st.file_uploader("Unggah Gambar Awan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra yang diunggah.", use_column_width=True)

def klasifikasi_decision_tree():
    st.title("Klasifikasi Cuaca menggunakan Decision Trees")
    suhu = st.number_input("Suhu (°C)", min_value=0, max_value=50, value=25)
    kelembaban = st.number_input("Kelembaban (%)", min_value=0, max_value=100, value=75)
    angin = st.number_input("Kecepatan Angin (km/jam)", min_value=0, max_value=200, value=15)
    if st.button("Prediksi Cuaca"):
        st.write("**Kategori Cuaca: Berawan**")

def clustering_kmeans():
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    data = np.random.rand(100, 2) * 100
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    labels = kmeans.labels_
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    ax.set_title("Clustering Curah Hujan")
    st.pyplot(fig)

# Sidebar Menu
st.sidebar.title("Main Menu")
menu = st.sidebar.radio(
    "Pilih Menu:",
    (
        "Home",
        "Prediksi Curah Hujan dengan Metode ARIMA",
        "Klasifikasi Citra Awan Curah Hujan dengan Metode CNN",
        "Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees",
        "Clustering Curah Hujan dengan Metode K-Means",
    )
)

# Menentukan menu yang dipilih
if menu == "Home":
    home()
elif menu == "Prediksi Curah Hujan dengan Metode ARIMA":
    prediksi_arima()
elif menu == "Klasifikasi Citra Awan Curah Hujan dengan Metode CNN":
    klasifikasi_cnn()
elif menu == "Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees":
    klasifikasi_decision_tree()
elif menu == "Clustering Curah Hujan dengan Metode K-Means":
    clustering_kmeans()


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
    st.title("Home")
    st.write("Selamat datang di aplikasi prediksi dan analisis curah hujan!")
    st.image(
        "https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png",
        caption="Gambar Hider",
        use_column_width=True,
    )

# Prediksi Curah Hujan dengan Metode ARIMA
def prediksi_arima():
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini akan memproses data curah hujan menggunakan model ARIMA.")
    
    # Contoh data dummy
    st.write("**Data Curah Hujan**:")
    data = {"Bulan": ["Jan", "Feb", "Mar", "Apr", "Mei"],
            "Curah Hujan": [100, 150, 200, 250, 300]}
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    st.write("**Model ARIMA akan diterapkan di sini.**")
    st.info("Anda dapat menambahkan model ARIMA lengkap sesuai dataset.")

# Klasifikasi Citra Awan untuk Prediksi Curah Hujan dengan Metode CNN
def klasifikasi_cnn():
    st.title("Klasifikasi Citra Awan dengan Metode CNN")
    st.write("Unggah citra awan untuk melakukan prediksi jenis curah hujan.")
    
    uploaded_file = st.file_uploader("Unggah Gambar Awan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra yang diunggah.", use_column_width=True)
        
        # Contoh prediksi dummy
        st.write("**Hasil Prediksi:**")
        st.write("Jenis Awan: **Cumulonimbus**")
        st.write("Potensi Curah Hujan: **Tinggi**")
        st.info("Implementasikan model CNN untuk prediksi citra awan.")

# Klasifikasi Cuaca menggunakan Decision Trees
def klasifikasi_decision_tree():
    st.title("Klasifikasi Cuaca menggunakan Decision Trees")
    st.write("Masukkan data fitur untuk melakukan klasifikasi cuaca.")
    
    # Contoh data input
    suhu = st.number_input("Suhu (°C)", min_value=0, max_value=50, value=25)
    kelembaban = st.number_input("Kelembaban (%)", min_value=0, max_value=100, value=75)
    angin = st.number_input("Kecepatan Angin (km/jam)", min_value=0, max_value=200, value=15)
    
    if st.button("Prediksi Cuaca"):
        # Contoh prediksi dummy
        st.write("**Hasil Prediksi:**")
        st.write("Kategori Cuaca: **Berawan**")
        st.info("Tambahkan Decision Tree untuk prediksi cuaca.")

# Clustering Curah Hujan dengan Metode K-Means
def clustering_kmeans():
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Lakukan clustering pada data curah hujan.")
    
    # Contoh data dummy
    data = np.random.rand(100, 2) * 100
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    labels = kmeans.labels_
    
    # Plot hasil clustering
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    ax.set_title("Clustering Curah Hujan")
    st.pyplot(fig)
    st.info("Implementasikan clustering curah hujan dengan dataset nyata.")

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

# Navigasi Menu
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

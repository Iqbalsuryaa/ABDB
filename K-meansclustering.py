# Import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Streamlit setup
st.title("K-Means Clustering dengan Dataset Cuaca")

# File uploader
uploaded_file = st.file_uploader("Upload file CSV Anda", type="csv")

# Periksa apakah file diupload
if uploaded_file is not None:
    try:
        # Membaca dataset
        data = pd.read_csv(uploaded_file)
        st.write("Dataset berhasil diupload!")

        # Menampilkan deskripsi fitur data
        st.subheader("Deskripsi Fitur Dataset")
        deskripsi_fitur = {
            "Tn": "Temperatur Minimum (°C)",
            "Tx": "Temperatur Maksimum (°C)",
            "Tavg": "Temperatur Rata-rata (°C)",
            "RH_avg": "Kelembaban Rata-rata (%)",
            "RR": "Curah Hujan (mm)",
            "ss": "Lamanya Penyinaran Matahari (jam)",
            "ff_x": "Kecepatan Angin Maksimum (m/s)",
            "ddd_x": "Arah Angin Saat Kecepatan Maksimum",
            "ff_avg": "Kecepatan Angin Rata-rata (m/s)",
            "ddd_car": "Arah Angin Terbanyak"
        }
        for fitur, deskripsi in deskripsi_fitur.items():
            st.write(f"{fitur}: {deskripsi}")

        # Menampilkan dataset
        st.subheader("Lima Baris Pertama Dataset")
        st.write(data.head())

        # Heatmap korelasi
        st.subheader("Heatmap Korelasi Antar Fitur")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        st.pyplot(plt)

        # Standarisasi data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

        # Menentukan jumlah cluster dengan metode Elbow
        st.subheader("Metode Elbow untuk Menentukan Cluster Optimal")
        inertia = []
        K = range(1, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data_scaled)
            inertia.append(kmeans.inertia_)

        # Plot Elbow
        plt.figure(figsize=(8, 5))
        plt.plot(K, inertia, 'bx-')
        plt.xlabel('Jumlah Cluster')
        plt.ylabel('Inertia')
        plt.title('Metode Elbow')
        st.pyplot(plt)

        # Clustering menggunakan K-Means
        st.subheader("Hasil Clustering")
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(data_scaled)  # Perbaikan di sini
        st.write("Rata-rata nilai tiap cluster:")
        st.write(data.groupby('Cluster').mean())

        # Visualisasi cluster
        st.subheader("Visualisasi Cluster")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            x=data['Tavg'], y=data['RH_avg'], hue=data['Cluster'], palette='viridis'
        )
        plt.title('Visualisasi Cluster')
        plt.xlabel('Temperatur Rata-rata (°C)')
        plt.ylabel('Kelembaban Rata-rata (%)')
        plt.legend(title='Cluster')
        st.pyplot(plt)

        # Menampilkan hasil per cluster
        for cluster in sorted(data['Cluster'].unique()):
            st.subheader(f"Cluster {cluster}")
            st.write(data[data['Cluster'] == cluster])

    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca dataset: {e}")
else:
    st.info("Silakan upload file CSV untuk memulai analisis.")

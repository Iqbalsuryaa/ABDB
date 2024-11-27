import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Judul aplikasi
st.title('Aplikasi Clustering Curah Hujan')

# Fitur upload file
st.sidebar.header('Unggah File Hasil Clustering')
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=['csv'])

if uploaded_file is not None:
    # Membaca file yang diunggah
    df = pd.read_csv(uploaded_file)

    # Menampilkan data frame
    st.write("Data yang diunggah:")
    st.write(df.head())

    # Pastikan data memiliki kolom yang diperlukan untuk clustering dan analisis heatmap
    if 'Tavg' in df.columns and 'RH_avg' in df.columns and 'RR' in df.columns:
        # Menampilkan informasi mengenai dataset
        st.write("Informasi Dataset:")
        st.write(df.describe())

        # Menyusun data untuk heatmap
        correlation = df[['Tavg', 'RH_avg', 'RR']].corr()

        # Membuat figure dan ax untuk plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Membuat heatmap
        sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)

        # Menampilkan plot dengan Streamlit
        st.pyplot(fig)

        # Melakukan clustering dengan KMeans
        st.sidebar.header('Pengaturan Clustering')
        n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=3)

        # Menggunakan StandardScaler untuk normalisasi
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[['Tavg', 'RH_avg', 'RR']])

        # Melakukan clustering menggunakan KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Menampilkan hasil clustering
        st.write(f"Hasil Clustering dengan {n_clusters} Cluster:")
        st.write(df.head())

        # Menampilkan plot clustering
        st.write("Visualisasi Hasil Clustering:")
        fig2, ax2 = plt.subplots(figsize=(10, 6))

        # Plot hasil clustering
        scatter = ax2.scatter(df['Tavg'], df['RH_avg'], c=df['Cluster'], cmap='viridis')
        ax2.set_xlabel('Temperatur Rata-rata (Tavg)')
        ax2.set_ylabel('Kelembaban Rata-rata (RH_avg)')
        fig2.colorbar(scatter, ax=ax2, label='Cluster')

        # Menampilkan plot clustering
        st.pyplot(fig2)
        
    else:
        st.error("File tidak memiliki kolom yang diperlukan ('Tavg', 'RH_avg', 'RR').")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")

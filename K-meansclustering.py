import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Data
data_file = st.file_uploader("Upload Data", type=["csv", "xlsx"])
if data_file is not None:
    if data_file.name.endswith('csv'):
        df = pd.read_csv(data_file)
    elif data_file.name.endswith('xlsx'):
        df = pd.read_excel(data_file)

    # Menampilkan data untuk melihat struktur
    st.write(df.head())

    # Pastikan tidak ada kolom yang memiliki tipe data string selain label
    # Menangani kolom kategorikal
    # Misalkan kolom yang mengandung string seperti 'Kategori' perlu diubah menjadi numerik
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))

    # Menangani missing values
    # Mengganti NaN dengan nilai rata-rata untuk kolom numerik
    df.fillna(df.mean(), inplace=True)

    # Menampilkan data yang sudah dibersihkan
    st.write("Data setelah membersihkan kolom:")
    st.write(df.head())

    # Pilih fitur untuk clustering
    features = st.multiselect("Pilih fitur untuk clustering", df.columns)

    # Jika sudah memilih fitur dan menekan tombol
    if st.button('Tampilkan Hasil Clustering K-Means'):
        if len(features) > 0:
            # Standardisasi data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df[features])

            # KMeans Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            df['cluster'] = kmeans.fit_predict(df_scaled)

            # Menampilkan hasil
            st.write("Hasil Clustering:")
            st.write(df.head())
            
            # Visualisasi Clustering
            st.subheader("Visualisasi Clustering")
            st.scatter_chart(df[['cluster']].head())  # Gantilah ini sesuai visualisasi yang sesuai
        else:
            st.warning("Pilih fitur terlebih dahulu untuk clustering")

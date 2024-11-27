import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan grafik Elbow
def elbow_method(data):
    wcss = []
    for i in range(1, 11):  # Coba k = 1 sampai 10
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    # Menampilkan grafik Elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss)
    plt.title('Metode Elbow untuk Pemilihan Jumlah Klaster')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.show()

# Upload file CSV
st.title('K-Means Clustering dengan Streamlit')

uploaded_file = st.file_uploader("Pilih file CSV untuk analisis K-Means", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Menampilkan data yang diupload
    st.write("Data yang diupload:")
    st.write(df.head())
    
    # Cek missing values
    st.write("Cek missing values:")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    # Tangani missing values (Imputasi dengan mean)
    st.write("Mengisi missing values dengan mean:")
    imputer = SimpleImputer(strategy='mean')  # Bisa juga menggunakan 'median' atau 'most_frequent'
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Cek kembali missing values setelah imputasi
    missing_values_after_imputation = df_imputed.isnull().sum()
    st.write(missing_values_after_imputation)
    
    # Scaling data
    st.write("Melakukan scaling pada data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_imputed)
    
    # Menampilkan grafik Elbow untuk menentukan jumlah klaster
    st.write("Grafik Elbow untuk menentukan jumlah klaster:")
    elbow_method(scaled_data)
    
    # Menjalankan K-Means (misalnya dengan 3 klaster)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(scaled_data)
    
    # Menampilkan hasil klastering
    st.write("Hasil Klastering K-Means:")
    df_imputed['Cluster'] = kmeans.labels_
    st.write(df_imputed.head())

    # Menampilkan grafik hasil klastering
    st.write("Grafik Hasil Klastering:")
    plt.figure(figsize=(8, 6))
    plt.scatter(scaled_data[:, 0], scaled_data[:, 1], c=kmeans.labels_, cmap='rainbow')
    plt.title('Hasil K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot()

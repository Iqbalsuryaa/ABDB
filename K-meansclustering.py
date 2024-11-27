import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Judul aplikasi
st.title("Aplikasi Clustering Curah Hujan 2020-2024")

# Upload file CSV hasil clustering
uploaded_file = st.file_uploader("Upload Hasil Cluster CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca data dari file yang diupload
    df_result = pd.read_csv(uploaded_file)
    
    # Menampilkan data hasil preprocessing
    st.subheader("Data Hasil Preprocessing")
    st.write(df_result.head())
    
    # Pastikan kolom 'cluster' ada di dataframe
    if 'cluster' in df_result.columns:
        st.subheader("Evaluasi K-Means Clustering")
        
        # Menangani nilai NaN dengan mengimputasi menggunakan rata-rata untuk kolom numerik
        numeric_columns = df_result.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove('cluster')  # Pastikan kolom 'cluster' tidak terimputasi
        
        imputer = SimpleImputer(strategy='mean')
        df_result[numeric_columns] = imputer.fit_transform(df_result[numeric_columns])
        
        # Menggunakan StandardScaler untuk menormalkan data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_result[numeric_columns])
        
        # Davies-Bouldin Index dan Silhouette Score
        try:
            kmeans_dbi = davies_bouldin_score(X_scaled, df_result['cluster'])
            kmeans_sil = silhouette_score(X_scaled, df_result['cluster'])
            st.write(f"**Davies-Bouldin Index**: {kmeans_dbi:.5f}")
            st.write(f"**Silhouette Score**: {kmeans_sil:.5f}")
        except Exception as e:
            st.write(f"Terjadi kesalahan dalam perhitungan skor: {str(e)}")
        
        # Descriptive statistics of clusters
        st.subheader("Descriptive Statistics per Cluster")
        for cluster in df_result['cluster'].unique():
            st.write(f"Cluster {cluster}:")
            st.write(df_result[df_result['cluster'] == cluster].describe())
        
        # Plotting Elbow Method
        st.subheader("Metode Elbow K-Means")
        range_n_clusters = list(range(1, 11))
        wcss = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_scaled)  # Gunakan data yang sudah diskalakan
            wcss.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 6))
        plt.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
        plt.title('Metode Elbow K-Means')
        plt.xlabel('Jumlah Cluster')
        plt.ylabel('WCSS')
        st.pyplot()

        # Map: Heatmap Curah Hujan
        st.subheader("Peta Heatmap Curah Hujan")
        if 'Latitude' in df_result.columns and 'Longitude' in df_result.columns:
            m = folium.Map(location=[df_result['Latitude'].mean(), df_result['Longitude'].mean()], zoom_start=6)
            heat_data = [[row['Latitude'], row['Longitude'], row['RR']] for index, row in df_result.iterrows()]
            HeatMap(heat_data).add_to(m)
            st.write(m)
        else:
            st.write("Data Latitude dan Longitude tidak ditemukan.")
    else:
        st.write("File yang diupload tidak mengandung kolom 'cluster'.")

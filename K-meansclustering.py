import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import folium
from folium import plugins
import geopandas as gpd
from branca.element import Template, MacroElement

# Membaca file CSV hasil clustering
@st.cache
def load_data():
    return pd.read_csv('Hasilcluster_result.csv')

df = load_data()

# Menampilkan data preprocessing
st.title("Aplikasi Clustering K-Means Curah Hujan")
st.header("Hasil Preprocessing Data")
st.write(df.head())  # Menampilkan preview data

# Elbow Method untuk Menentukan Jumlah Cluster Optimal
st.header("Metode Elbow K-Means")
range_n_clusters = list(range(1, 11))
wcss = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df.drop(columns=['cluster']))
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots()
ax.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
ax.set_title('Metode Elbow K-Means')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Menampilkan Hasil Clustering K-Means
st.header("Hasil Clustering K-Means")

# Memuat model KMeans yang sudah disimpan
import joblib
kmeans = joblib.load('kmeans_model.pkl')

# Menambahkan label cluster ke dalam data
df['cluster'] = kmeans.labels_

# Evaluasi Clustering
st.subheader("Evaluasi Model KMeans")
kmeans_dbi = davies_bouldin_score(df.drop(columns=['cluster']), df['cluster'])
kmeans_sil = silhouette_score(df.drop(columns=['cluster']), df['cluster'])
st.write(f'Davies-Bouldin Index: {kmeans_dbi:.5f}')
st.write(f'Silhouette Score: {kmeans_sil:.5f}')

# Menampilkan Descriptive Statistics untuk setiap cluster
st.subheader("Statistik Deskriptif Tiap Cluster")

for i in range(3):
    st.write(f"Descriptive statistics of cluster {i}")
    cluster_data = df[df['cluster'] == i]
    st.write(cluster_data.describe())

# Distribusi Cluster per Kabupaten
st.subheader("Distribusi Cluster per Kabupaten")
cluster_distribution = df['cluster'].value_counts()
st.bar_chart(cluster_distribution)

# Membuat Peta Heatmap Curah Hujan
st.header("Peta Heatmap Curah Hujan")

# Membuat peta menggunakan folium
m = folium.Map(location=[-7.250445, 112.768845], zoom_start=6)  # Koordinat Indonesia

# Menambahkan marker untuk setiap kota berdasarkan latituda dan longitudanya
for _, row in df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.6
    ).add_to(m)

# Menampilkan peta
st.write(m)

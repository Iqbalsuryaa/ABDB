import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import folium
from folium.plugins import HeatMap
import joblib

# Load Preprocessed Data
@st.cache
def load_data():
    df = pd.read_csv("Preprocessed_Dataset.csv")  # Pastikan file csv sudah tersedia
    return df

df = load_data()

# Menampilkan Data Preprocessing
st.title('Clustering K-Means Curah Hujan')
st.write("Data hasil preprocessing:")
st.dataframe(df.head())

# Menampilkan Metode Elbow
st.subheader('Metode Elbow K-Means')
fig, ax = plt.subplots(figsize=(8, 6))
range_n_clusters = list(range(1, 11))
wcss = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df.drop(columns=['KOTA', 'Latitude', 'Longitude']))  # Sesuaikan kolom yang relevan
    wcss.append(kmeans.inertia_)

ax.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
ax.set_title('Metode Elbow K-Means')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Menampilkan Hasil K-Means Clustering
st.subheader('Hasil K-Means Clustering')
kmeans = KMeans(n_clusters=3, random_state=42).fit(df.drop(columns=['KOTA', 'Latitude', 'Longitude']))
df['cluster'] = kmeans.labels_

st.write(f'Davies-Bouldin Index: {davies_bouldin_score(df.drop(columns=["KOTA", "Latitude", "Longitude"]), kmeans.labels_):.5f}')
st.write(f'Silhouette Score: {silhouette_score(df.drop(columns=["KOTA", "Latitude", "Longitude"]), kmeans.labels_):.5f}')

# Statistik Deskriptif untuk Setiap Cluster
st.subheader('Descriptive Statistics for Clusters')
for i in range(3):
    st.write(f"Descriptive statistics of cluster {i}")
    st.dataframe(df[df['cluster'] == i].describe())

# Distribusi Cluster per Kabupaten
st.subheader('Distribusi Cluster per Kabupaten')
st.bar_chart(df['cluster'].value_counts())

# Menampilkan Peta dengan Heatmap Curah Hujan
st.subheader('Peta dengan Heatmap Curah Hujan')
m = folium.Map(location=[-7.2504, 112.7688], zoom_start=6)  # Koordinat Indonesia

heat_data = [[row['Latitude'], row['Longitude'], row['RR']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(m)

# Menampilkan peta di Streamlit
folium_static(m)

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
import joblib
from sklearn.impute import SimpleImputer

# Memuat dataset yang sudah dibersihkan dan model KMeans
@st.cache
def load_data():
    df = pd.read_csv('Hasilcluster_result.csv')  # Sesuaikan dengan path dataset yang digunakan
    kmeans = joblib.load('kmeans_model.pkl')  # Sesuaikan dengan path model yang disimpan
    return df, kmeans

df, kmeans = load_data()

# Menghapus baris yang mengandung nilai NaN
df = df.dropna(subset=['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])

# Preprocessing
scaler = StandardScaler()
fitur = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
df_scaled = df.copy()
df_scaled[fitur] = scaler.fit_transform(df_scaled[fitur])

# Menampilkan data preprocessing
st.title('Clustering Curah Hujan')
st.subheader('Data Preprocessing')
st.write(df.head())

# Menampilkan grafik Elbow untuk memilih jumlah cluster
st.subheader('Metode Elbow untuk Menentukan K Optimal')
wcss = []
range_n_clusters = list(range(1, 11))
for n_clusters in range_n_clusters:
    kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_temp.fit(df_scaled[fitur])
    wcss.append(kmeans_temp.inertia_())

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
ax.set_title('Metode Elbow K-Means')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Menampilkan hasil clustering K-Means
kmeans = KMeans(n_clusters=3, random_state=42).fit(df_scaled[fitur])
df['cluster'] = kmeans.labels_

# Evaluasi K-Means
davies_bouldin = davies_bouldin_score(df_scaled[fitur], df['cluster'])
silhouette = silhouette_score(df_scaled[fitur], df['cluster'])

st.subheader('Evaluasi K-Means')
st.write(f'Index Davies-Bouldin: {davies_bouldin:.5f}')
st.write(f'Skor Silhouette: {silhouette:.5f}')

# Statistik deskriptif per cluster
st.subheader('Statistik Deskriptif per Cluster')
for i in range(3):
    st.write(f"Cluster {i}")
    st.write(df[df['cluster'] == i].describe())

# Distribusi Cluster per Kabupaten
st.subheader('Distribusi Cluster per Kabupaten')
cluster_counts = df.groupby(['cluster', 'KOTA']).size().unstack().fillna(0)
st.write(cluster_counts)

# Heatmap Curah Hujan
st.subheader('Heatmap Curah Hujan')
# Membuat peta dasar
m = folium.Map(location=[-7.250445, 112.768845], zoom_start=7)

# Menambahkan heatmap
heat_data = [[row['Latitude'], row['Longitude'], row['RR']] for index, row in df.iterrows()]
HeatMap(heat_data).add_to(m)

# Menampilkan peta di Streamlit
st.write(m)

# Menyimpan hasil clustering
st.subheader('Hasil Clustering')
st.write(df[['KOTA', 'cluster']].head())

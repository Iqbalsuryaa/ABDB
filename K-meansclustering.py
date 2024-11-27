import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Fungsi untuk memuat data
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Fungsi untuk membuat heatmap
def create_heatmap(data, lat_col, lon_col, value_col):
    m = folium.Map(location=[data[lat_col].mean(), data[lon_col].mean()], zoom_start=7)
    heat_data = data[[lat_col, lon_col, value_col]].dropna().values.tolist()
    HeatMap(heat_data).add_to(m)
    return m

# Judul aplikasi
st.title("Aplikasi Clustering K-Means untuk Data Curah Hujan")

# Unggah dataset hasil preprocessing
st.sidebar.title("Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV hasil preprocessing", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Menampilkan data awal
    st.subheader("Data Hasil Preprocessing")
    st.write(df.head())

    # Input untuk parameter clustering
    st.sidebar.subheader("Parameter Clustering")
    num_clusters = st.sidebar.slider("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)
    
    # Menjalankan K-Means
    if st.sidebar.button("Jalankan Clustering"):
        cleaned_data = df.select_dtypes(include=['float64', 'int64'])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(cleaned_data)

        # Menampilkan hasil clustering
        st.subheader("Hasil Clustering")
        st.write(df)

        # Menampilkan evaluasi clustering
        dbi = davies_bouldin_score(cleaned_data, df['cluster'])
        sil = silhouette_score(cleaned_data, df['cluster'])
        st.write(f"Davies-Bouldin Index: {dbi:.5f}")
        st.write(f"Silhouette Score: {sil:.5f}")

        # Menampilkan descriptive statistics
        st.subheader("Descriptive Statistics per Cluster")
        for cluster in sorted(df['cluster'].unique()):
            st.write(f"Descriptive statistics for Cluster {cluster}")
            st.write(df[df['cluster'] == cluster].describe())

        # Distribusi cluster per kabupaten
        if 'KOTA' in df.columns:
            st.subheader("Distribusi Cluster per Kabupaten")
            dist_kabupaten = df.groupby(['KOTA', 'cluster']).size().unstack(fill_value=0)
            st.write(dist_kabupaten)

        # Menampilkan peta heatmap
        if 'Latitude' in df.columns and 'Longitude' in df.columns and 'RR' in df.columns:
            st.subheader("Peta Heatmap Curah Hujan")
            heatmap = create_heatmap(df, lat_col='Latitude', lon_col='Longitude', value_col='RR')
            st.write("Peta Distribusi Curah Hujan:")
            folium_static(heatmap)

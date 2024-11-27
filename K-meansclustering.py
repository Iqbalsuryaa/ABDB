import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from io import BytesIO

# Fungsi untuk preprocessing data
def preprocess_data(df):
    numeric_cols = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    le = LabelEncoder()
    df['KOTA'] = le.fit_transform(df['KOTA'])
    return df

# Fungsi untuk visualisasi elbow method
def elbow_method(data):
    wcss = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', markersize=8, color='b')
    plt.title('Metode Elbow K-Means')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    st.pyplot(plt)

# Fungsi untuk clustering dan visualisasi hasil cluster
def perform_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(data)
    return data

# Fungsi untuk visualisasi distribusi cluster
def plot_cluster_distribution(data):
    cluster_summary = data.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='KOTA', y='Count', hue='cluster', data=cluster_summary, ax=ax, palette='viridis')
    plt.title('Distribusi Cluster per Kota')
    plt.xlabel('Kota')
    plt.ylabel('Jumlah')
    st.pyplot(plt)

# Fungsi untuk heatmap
def plot_heatmap(data):
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    heatmap_map = folium.Map(location=map_center, zoom_start=6)
    features = {
        'Tavg': data[['Latitude', 'Longitude', 'Tavg']],
        'RH_avg': data[['Latitude', 'Longitude', 'RH_avg']],
        'RR': data[['Latitude', 'Longitude', 'RR']],
        'ss': data[['Latitude', 'Longitude', 'ss']],
    }
    for feature_name, feature_data in features.items():
        heatmap_layer = plugins.HeatMap(
            feature_data.dropna().values.tolist(), 
            radius=15, 
            name=feature_name
        )
        heatmap_map.add_child(heatmap_layer)
    heatmap_map.add_child(folium.LayerControl())
    return heatmap_map

# Streamlit app layout
st.title("Aplikasi Clustering K-Means untuk Data Curah Hujan")
st.write("Unggah dataset curah hujan untuk melakukan analisis clustering menggunakan metode K-Means.")

uploaded_file = st.file_uploader("Unggah File CSV Anda", type=["csv"])
if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset berhasil diunggah:")
    st.write(df.head())
    
    # Preprocessing
    st.subheader("Preprocessing Data")
    processed_data = preprocess_data(df.copy())
    st.write("Data setelah preprocessing:")
    st.write(processed_data.head())
    
    # Elbow method
    st.subheader("Metode Elbow untuk Menentukan Jumlah Cluster")
    elbow_method(processed_data.drop(columns=['KOTA', 'Latitude', 'Longitude']))
    
    # Clustering
    st.subheader("Hasil Clustering K-Means")
    n_clusters = st.slider("Pilih jumlah cluster:", 2, 10, 3)
    clustered_data = perform_clustering(processed_data.copy(), n_clusters)
    st.write("Data setelah clustering:")
    st.write(clustered_data.head())
    
    # Visualisasi distribusi cluster
    st.subheader("Distribusi Cluster per Kota")
    plot_cluster_distribution(clustered_data)
    
    # Heatmap
    st.subheader("Visualisasi Heatmap")
    heatmap_map = plot_heatmap(clustered_data)
    st.write("Heatmap Data Curah Hujan")
    st.map(clustered_data[['Latitude', 'Longitude']])

    # Simpan hasil
    st.subheader("Simpan Hasil Clustering")
    buffer = BytesIO()
    clustered_data.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="Download hasil clustering",
        data=buffer,
        file_name="hasil_clustering.csv",
        mime="text/csv"
    )
else:
    st.write("Silakan unggah file dataset Anda untuk memulai analisis.")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import folium
from folium.plugins import HeatMap

# Title for the Streamlit app
st.title("Aplikasi Clustering Curah Hujan 2020 - 2024")

# File uploader to upload dataset
uploaded_file = st.file_uploader("Upload file CSV hasil clustering", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Show basic information about the dataset
    st.write("Data yang diunggah:")
    st.write(df.head())
    
    # Preprocessing steps
    # (Jika perlu, sesuaikan preprocessing dengan kode yang telah Anda buat sebelumnya)
    # Example: Scaling the features
    features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_scaled = df.copy()
    scaler = StandardScaler()
    df_scaled[features] = scaler.fit_transform(df[features])
    
    # Elbow method to determine optimal number of clusters
    st.subheader("Metode Elbow K-Means")
    wcss = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_scaled[features])
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='*', markersize=10, markerfacecolor='red')
    plt.title('Metode Elbow K-Means')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    st.pyplot()

    # Apply KMeans clustering with the selected number of clusters (e.g., 3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled[features])
    
    # Clustering evaluation
    davies_bouldin = davies_bouldin_score(df_scaled[features], df['cluster'])
    silhouette = silhouette_score(df_scaled[features], df['cluster'])
    
    st.subheader("Evaluasi K-Means Clustering")
    st.write(f"**Davies-Bouldin Index**: {davies_bouldin:.5f}")
    st.write(f"**Silhouette Score**: {silhouette:.5f}")
    
    # Descriptive statistics per cluster
    st.subheader("Descriptive Statistics per Cluster")
    for i in range(3):  # Assuming 3 clusters
        st.write(f"**Descriptive statistics of cluster {i}:**")
        st.write(df[df['cluster'] == i].describe())
    
    # Visualize the clusters using a pairplot
    st.subheader("Visualisasi Cluster")
    sns.pairplot(df, hue='cluster', diag_kind='kde', palette='tab10')
    st.pyplot()

    # Visualize distribution of clusters per kabupaten
    st.subheader("Distribusi Cluster per Kabupaten")
    cluster_counts = df['cluster'].value_counts()
    st.write(cluster_counts)

    # Heatmap of rainfall distribution
    st.subheader("Peta Distribusi Curah Hujan")
    map_ = folium.Map(location=[-7.250445, 112.768845], zoom_start=5)  # Example center of Indonesia
    heat_data = [[row['Latitude'], row['Longitude'], row['RR']] for _, row in df.iterrows()]
    HeatMap(heat_data).add_to(map_)
    st.write(map_)

else:
    st.write("Silakan upload file CSV hasil clustering Anda.")

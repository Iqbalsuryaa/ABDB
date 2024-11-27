import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score
import folium
from folium.plugins import HeatMap

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("Hasilcluster_result.csv")  # Pastikan file ini sudah diupload di direktori aplikasi
    return df

# Preprocessing Function
@st.cache
def preprocess_data(df):
    df_clean = df.copy()
    scaler = StandardScaler()
    features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_clean[features] = scaler.fit_transform(df_clean[features])
    return df_clean

# Elbow Method for KMeans
@st.cache
def elbow_method(df):
    range_n_clusters = list(range(1, 11))
    wcss = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
    plt.title('Metode Elbow K-Means')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    st.pyplot()

# KMeans Clustering
@st.cache
def kmeans_clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42).fit(df)
    df['cluster'] = kmeans.labels_
    return df, kmeans

# Descriptive statistics of clusters
def describe_clusters(df):
    cluster_dfs = {}
    for cluster in df['cluster'].unique():
        cluster_dfs[cluster] = df[df['cluster'] == cluster]
    for cluster, cluster_df in cluster_dfs.items():
        st.write(f"Descriptive statistics of cluster {cluster}")
        st.write(cluster_df.describe())

# Davies-Bouldin and Silhouette Scores
def evaluate_clustering(df, kmeans):
    db_index = davies_bouldin_score(df, kmeans.labels_)
    silhouette = silhouette_score(df, kmeans.labels_)
    st.write(f"**Davies-Bouldin Index**: {db_index:.5f}")
    st.write(f"**Silhouette Score**: {silhouette:.5f}")

# Mapping Heatmap
def plot_heatmap(df):
    m = folium.Map(location=[-7.250445, 112.768845], zoom_start=6)
    heat_data = [[row['Latitude'], row['Longitude'], row['RR']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(m)
    st.write("**Peta Distribusi Curah Hujan**")
    st.write(m)

def main():
    st.title("Aplikasi Clustering K-Means untuk Curah Hujan")

    # Load Data
    df = load_data()

    # Show raw data and description
    st.write("**Data Hujan Tahun 2020 - 2024**")
    st.write(df.head())

    # Preprocess Data
    df_clean = preprocess_data(df)
    st.write("**Data setelah Preprocessing**")
    st.write(df_clean.head())

    # Elbow Method
    st.subheader("Metode Elbow K-Means")
    elbow_method(df_clean)

    # Perform KMeans Clustering
    st.subheader("Hasil Clustering K-Means")
    clustered_data, kmeans_model = kmeans_clustering(df_clean)
    st.write(clustered_data.head())

    # Evaluation
    st.subheader("Evaluasi K-Means Clustering")
    evaluate_clustering(df_clean, kmeans_model)

    # Descriptive statistics per cluster
    describe_clusters(clustered_data)

    # Plot Heatmap
    st.subheader("Distribusi Curah Hujan per Kabupaten")
    plot_heatmap(clustered_data)

if __name__ == '__main__':
    main()

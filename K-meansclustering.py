import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import KMeans

# Load files
st.title("K-Means Clustering untuk Curah Hujan")

# Upload files
uploaded_preprocessed = st.file_uploader("Upload Preprocessed Dataset CSV", type="csv")
uploaded_cluster_result = st.file_uploader("Upload Hasil Clustering CSV", type="csv")
uploaded_model = st.file_uploader("Upload K-Means Model (PKL)", type="pkl")

if uploaded_preprocessed and uploaded_cluster_result and uploaded_model:
    # Load data
    preprocessed_data = pd.read_csv(uploaded_preprocessed)
    cluster_result = pd.read_csv(uploaded_cluster_result)
    kmeans_model = pickle.load(uploaded_model)

    # Display preprocessed data
    st.subheader("Preprocessed Dataset")
    st.dataframe(preprocessed_data.head())

    # Elbow Method
    st.subheader("Elbow Method")
    X = preprocessed_data.drop(columns=["NO", "Tanggal", "KOTA", "Latitude", "Longitude"])
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 5))
    plt.plot(K, distortions, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    st.pyplot(plt)

    # Clustering Results
    st.subheader("Clustering Results")
    davies_bouldin = davies_bouldin_score(X, cluster_result["cluster"])
    silhouette = silhouette_score(X, cluster_result["cluster"])
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
    st.write(f"Silhouette Score: {silhouette:.2f}")

    # Descriptive Statistics
    st.subheader("Descriptive Statistics per Cluster")
    clusters = cluster_result["cluster"].unique()
    for cluster in clusters:
        st.write(f"Descriptive statistics of cluster {cluster}")
        st.dataframe(cluster_result[cluster_result["cluster"] == cluster].describe())

    # Distribution per Kabupaten
    st.subheader("Distribusi Cluster per Kabupaten")
    cluster_counts = cluster_result.groupby(["KOTA", "cluster"]).size().unstack(fill_value=0)
    st.bar_chart(cluster_counts)

    # Heatmap Visualization
    st.subheader("Heatmap Curah Hujan")
    import folium
    from streamlit_folium import folium_static

    m = folium.Map(location=[-7.5, 112.5], zoom_start=7)
    for _, row in cluster_result.iterrows():
        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
        ).add_to(m)
    
    folium_static(m)

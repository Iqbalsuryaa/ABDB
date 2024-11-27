import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Fungsi untuk load data
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Fungsi untuk plotting elbow method
def elbow_method(data):
    range_n_clusters = range(1, 11)
    wcss = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# Fungsi untuk membuat heatmap
def plot_heatmap(data):
    m = folium.Map(location=[-7.250445, 112.768845], zoom_start=7)
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in data.iterrows()]
    HeatMap(heat_data).add_to(m)
    return m

# Fungsi untuk evaluasi cluster
def evaluate_clustering(data, labels):
    dbi = davies_bouldin_score(data, labels)
    sil = silhouette_score(data, labels)
    return dbi, sil

# Main Streamlit App
st.title("Aplikasi K-Means Clustering Curah Hujan")
st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader("Upload file hasil clustering (CSV)", type="csv")

if uploaded_file:
    # Load data
    data = load_data(uploaded_file)
    st.write("### Data yang Diupload")
    st.dataframe(data.head())

    # Menampilkan deskriptif statistik
    st.write("### Statistik Deskriptif")
    st.write(data.describe())

    # Preprocessing
    st.write("### Data Preprocessing")
    numeric_columns = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[numeric_columns])
    scaled_df = pd.DataFrame(scaled_data, columns=numeric_columns)

    # Elbow Method
    st.write("### Metode Elbow")
    wcss = elbow_method(scaled_df)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title("Metode Elbow")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    # K-Means Clustering
    st.sidebar.header("Clustering Parameters")
    n_clusters = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster'] = kmeans.fit_predict(scaled_df)

    # Evaluasi Cluster
    dbi, sil = evaluate_clustering(scaled_df, data['cluster'])
    st.write("### Evaluasi Clustering")
    st.write(f"Davies-Bouldin Index: {dbi:.5f}")
    st.write(f"Silhouette Score: {sil:.5f}")

    # Statistik per Cluster
    st.write("### Statistik per Cluster")
    for cluster in range(n_clusters):
        st.write(f"#### Cluster {cluster}")
        st.write(data[data['cluster'] == cluster].describe())

    # Distribusi Cluster
    st.write("### Distribusi Cluster per Kabupaten")
    st.bar_chart(data['cluster'].value_counts())

    # Heatmap
    st.write("### Heatmap Curah Hujan")
    map_data = data[['Latitude', 'Longitude']].dropna()
    folium_map = plot_heatmap(map_data)
    folium_static(folium_map)

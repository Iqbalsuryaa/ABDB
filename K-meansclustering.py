import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import folium
from folium import plugins

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    return pd.read_csv('Hasilcluster_result.csv')

# Fungsi untuk menampilkan metode elbow
def elbow_method(data):
    wcss = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', color='b')
    plt.title('Metode Elbow K-Means')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    st.pyplot(plt)

# Fungsi untuk menampilkan heatmap
def create_heatmap(data):
    map_heatmap = folium.Map(
        location=[data['Latitude'].mean(), data['Longitude'].mean()],
        zoom_start=6
    )
    features = {
        'Tavg': data[['Latitude', 'Longitude', 'Tavg']],
        'RH_avg': data[['Latitude', 'Longitude', 'RH_avg']],
        'RR': data[['Latitude', 'Longitude', 'RR']],
        'ss': data[['Latitude', 'Longitude', 'ss']]
    }
    for feature_name, feature_data in features.items():
        heatmap_data = feature_data.dropna().values.tolist()
        heatmap_layer = plugins.HeatMap(heatmap_data, radius=15)
        folium.FeatureGroup(name=feature_name).add_child(heatmap_layer).add_to(map_heatmap)
    folium.LayerControl().add_to(map_heatmap)
    return map_heatmap

# Streamlit Layout
st.title("Aplikasi Clustering K-Means untuk Curah Hujan")
st.sidebar.header("Pengaturan")
menu = st.sidebar.radio("Pilih Menu", ["Metode Elbow", "Hasil Clustering", "Distribusi Cluster", "Heatmap"])

# Load Data
df = load_data()

# Preprocessing Data
cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
encoder = LabelEncoder()
cleaned_kota['KOTA'] = encoder.fit_transform(df['KOTA'])

# Metode Elbow
if menu == "Metode Elbow":
    st.subheader("Metode Elbow")
    elbow_method(cleaned_kota)

# Hasil Clustering
elif menu == "Hasil Clustering":
    st.subheader("Hasil Clustering K-Means")
    rename = {0: 2, 1: 0, 2: 1}
    df['cluster'] = df['cluster'].replace(rename)
    st.dataframe(df.head())

    st.subheader("Statistik Deskriptif per Cluster")
    col_drop = ['Tanggal', 'ddd_car', 'Latitude', 'Longitude', 'KOTA']
    desc_stats = (
        df.drop(col_drop, axis=1)
        .groupby('cluster')
        .aggregate(['mean', 'std', 'min', 'median', 'max'])
        .transpose()
    )
    st.dataframe(desc_stats)

# Distribusi Cluster per Kabupaten
elif menu == "Distribusi Cluster":
    st.subheader("Distribusi Cluster per Kabupaten")
    kota_cluster = df.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
    sns.barplot(data=kota_cluster, x='KOTA', y='Count', hue='cluster', palette='viridis')
    plt.xticks(rotation=90)
    st.pyplot(plt)

# Heatmap
elif menu == "Heatmap":
    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st.markdown(folium.Figure().add_child(heatmap).render(), unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import folium
from folium import plugins
from streamlit_folium import st_folium

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

    # Warna RGB untuk setiap cluster
    cluster_colors = {
        0: "red",      # Cluster 0: Curah hujan tinggi
        1: "blue",     # Cluster 1: Curah hujan rendah
        2: "green"     # Cluster 2: Curah hujan sedang
    }

    # Menambahkan marker untuk setiap cluster
    for _, row in data.iterrows():
        cluster = row['cluster']
        popup_text = f"""
        <b>Cluster:</b> {cluster}<br>
        <b>KOTA:</b> {row['KOTA']}<br>
        <b>Curah Hujan:</b> {row['RR']} mm<br>
        """
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            color=cluster_colors[cluster],
            fill=True,
            fill_color=cluster_colors[cluster],
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(map_heatmap)

    # Heatmap untuk data curah hujan
    heatmap_data = data[['Latitude', 'Longitude', 'RR']].dropna().values.tolist()
    plugins.HeatMap(heatmap_data, radius=15).add_to(map_heatmap)

    # Menambahkan Layer Control
    folium.LayerControl().add_to(map_heatmap)
    return map_heatmap

# Streamlit Layout
st.title("Aplikasi Clustering K-Means untuk Curah Hujan")

# Tambahkan opsi menu
menu = st.sidebar.selectbox("Pilih Menu", ["Metode Elbow", "Distribusi Cluster", "Heatmap"])

# Load Data
df = load_data()

# Preprocessing Data
cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
encoder = LabelEncoder()
cleaned_kota['KOTA'] = encoder.fit_transform(df['KOTA'])

# Menu Metode Elbow
if menu == "Metode Elbow":
    st.subheader("Metode Elbow")
    elbow_method(cleaned_kota)

# Menu Distribusi Cluster
elif menu == "Distribusi Cluster":
    st.subheader("Distribusi Cluster per Kabupaten")
    
    # Hitung distribusi cluster per kota
    kota_cluster = df.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
    
    # Membuat visualisasi barplot
    plt.figure(figsize=(10, 6))  # Mengatur ukuran figure
    sns.barplot(
        data=kota_cluster, 
        x='KOTA', 
        y='Count', 
        hue='cluster', 
        palette='viridis'
    )
    
    # Rotasi label pada sumbu x agar mudah dibaca
    plt.xticks(rotation=45, ha='right', fontsize=8)
    
    # Menambahkan judul dan label sumbu
    plt.title("Distribusi Cluster per Kabupaten", fontsize=14)
    plt.xlabel("Kabupaten/Kota", fontsize=12)
    plt.ylabel("Jumlah Observasi", fontsize=12)
    plt.legend(title="Cluster", fontsize=10, loc='upper right')  # Letak legenda disesuaikan
    
    # Menampilkan diagram pada Streamlit
    st.pyplot(plt)

# Menu Heatmap
elif menu == "Heatmap":
    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st_folium(heatmap, width=700, height=500)

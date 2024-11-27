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
    cluster_colors = {0: "red", 1: "blue", 2: "green"}  # Warna RGB untuk setiap cluster
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
    plugins.HeatMap(data[['Latitude', 'Longitude', 'RR']].dropna().values.tolist(), radius=15).add_to(map_heatmap)
    folium.LayerControl().add_to(map_heatmap)
    return map_heatmap

# Sidebar Menu
st.sidebar.title("Main Menu")
menu = st.sidebar.radio(
    "Pilih Menu:",
    (
        "Home",
        "Prediksi Dengan Metode ARIMA",
        "Klasifikasi Citra Dengan Metode CNN",
        "Klasifikasi Dengan Decision Trees",
        "Clustering Dengan Metode K-Means",
    )
)

# Menentukan menu yang dipilih
if menu == "Clustering Dengan Metode K-Means":
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Halaman ini akan berisi implementasi clustering data curah hujan dengan K-Means.")

    # Load Data
    df = load_data()

    # Preprocessing Data
    cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
    encoder = LabelEncoder()
    cleaned_kota['KOTA'] = encoder.fit_transform(df['KOTA'])

    # Metode Elbow
    st.subheader("Metode Elbow")
    elbow_method(cleaned_kota)

    # Hasil Clustering
    st.subheader("Hasil Clustering K-Means")
    rename = {0: 2, 1: 0, 2: 1}
    df['cluster'] = df['cluster'].replace(rename)
    st.markdown(""" 
    ### Cluster Berdasarkan Curah Hujan:
    1. **Cluster 0**: Curah hujan tinggi (musim hujan).
    2. **Cluster 2**: Curah hujan sedang (cuaca normal).
    3. **Cluster 1**: Curah hujan rendah (musim kering).
    """)
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
    st.subheader("Distribusi Cluster per Kabupaten")
    kota_cluster = df.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=kota_cluster, x='KOTA', y='Count', hue='cluster', palette='viridis')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.title("Distribusi Cluster per Kabupaten", fontsize=14)
    plt.xlabel("Kabupaten/Kota", fontsize=12)
    plt.ylabel("Jumlah Observasi", fontsize=12)
    plt.legend(title="Cluster", fontsize=10, loc='upper right')
    st.pyplot(plt)

    # Penjelasan Cluster Berdasarkan Curah Hujan
    st.subheader("Penjelasan Cluster Berdasarkan Curah Hujan")
    st.markdown(""" 
    1. **Cluster 0 (Curah Hujan Tinggi - Musim Hujan):**
       - Daerah dengan intensitas curah hujan tinggi.
       - Sering terjadi pada musim hujan dengan curah hujan di atas rata-rata.
    2. **Cluster 2 (Curah Hujan Sedang - Cuaca Normal):**
       - Daerah dengan curah hujan sedang, biasanya mencerminkan cuaca normal atau transisi musim.
    3. **Cluster 1 (Curah Hujan Rendah - Musim Kering):**
       - Daerah dengan intensitas curah hujan rendah.
       - Sering terjadi pada musim kemarau atau di wilayah yang lebih kering.
    """)

    # Heatmap
    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st_folium(heatmap, width=700, height=500)

    # Penjelasan Warna pada Heatmap
    st.markdown(""" 
    ### Penjelasan Warna pada Heatmap:
    - **Merah Tua / Oranye**: Daerah dengan curah hujan tinggi, biasanya terjadi pada musim hujan atau daerah tropis dengan intensitas hujan tinggi.
    - **Kuning / Hijau Muda**: Daerah dengan curah hujan sedang, mencerminkan cuaca normal atau transisi musim.
    - **Biru Tua / Biru Muda**: Daerah dengan curah hujan rendah, sering terjadi pada musim kemarau atau wilayah kering.
    """)

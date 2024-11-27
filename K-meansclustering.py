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
elif menu == "Distribusi Cluster":
    st.subheader("Distribusi Cluster per Kabupaten")
    kota_cluster = df.groupby(['cluster', 'KOTA']).size().reset_index(name='Count')
    sns.barplot(data=kota_cluster, x='KOTA', y='Count', hue='cluster', palette='viridis')
    plt.xticks(rotation=90)
    st.pyplot(plt)
    
    st.markdown("""
    ### Penjelasan Cluster Berdasarkan Curah Hujan:
    
    **Cluster 0 (Curah Hujan Tinggi - Musim Hujan)**:
    - Cluster ini menunjukkan daerah-daerah yang mengalami curah hujan tinggi. Biasanya cluster ini mewakili wilayah yang terletak di musim hujan atau daerah dengan iklim tropis yang sering mengalami hujan deras.
    - **Ciri-ciri**: Area yang termasuk dalam cluster ini akan menunjukkan intensitas curah hujan yang lebih tinggi (lebih dari rata-rata), yang biasanya terkait dengan musim hujan.
    
    **Cluster 2 (Curah Hujan Sedang - Cuaca Normal)**:
    - Cluster ini berisi daerah-daerah dengan curah hujan sedang, yang biasanya terjadi pada cuaca normal atau musim transisi antara musim hujan dan kemarau.
    - **Ciri-ciri**: Wilayah yang termasuk dalam cluster ini memiliki tingkat curah hujan yang cukup stabil, tidak terlalu tinggi dan juga tidak terlalu rendah, mencerminkan cuaca yang tidak ekstrem.
    
    **Cluster 1 (Curah Hujan Rendah - Musim Kering)**:
    - Cluster ini mencakup daerah-daerah yang mengalami curah hujan rendah, yang biasanya terjadi pada musim kemarau atau wilayah yang lebih kering.
    - **Ciri-ciri**: Area yang termasuk dalam cluster ini cenderung mengalami sedikit hujan atau bahkan tidak ada hujan sama sekali dalam periode tertentu, mencerminkan musim kering atau iklim yang lebih kering.
    """)

# Heatmap
elif menu == "Heatmap":
    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st_folium(heatmap, width=700, height=500)

    st.markdown("""
    ### Penjelasan Warna pada Heatmap:
    
    **Warna Merah Tua / Kuning Tua**:
    - Menunjukkan daerah dengan curah hujan yang tinggi. Lokasi-lokasi yang lebih intens curah hujannya akan tampak dengan warna yang lebih gelap, daerah dengan intensitas curah hujan tinggi sering kali berwarna merah tua atau oranye terang, menunjukkan curah hujan yang sangat tinggi.

    **Warna Kuning / Hijau Muda**:
    - Menunjukkan daerah dengan curah hujan sedang, Warna-warna seperti kuning atau hijau muda menandakan intensitas hujan yang lebih rendah dibandingkan dengan daerah merah.

    **Warna Biru Tua / Biru Muda**:
    - Menunjukkan daerah dengan curah hujan rendah. Ini biasanya mewakili lokasi-lokasi yang memiliki sedikit atau bahkan tidak ada hujan (seperti musim kemarau). Warna biru gelap atau biru muda ini menandakan intensitas hujan yang sangat rendah.
    """)

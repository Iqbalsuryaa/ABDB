import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import folium
from folium import plugins
from streamlit_folium import st_folium

# Fungsi untuk halaman Insight
def insight_content():
    st.write("Halaman ini menampilkan insight dari data yang telah dianalisis.")
    st.write("""
        - Analisis Tren: Curah hujan cenderung meningkat di musim penghujan.
        - Pola Cuaca: Suhu dan kelembaban memiliki korelasi signifikan terhadap curah hujan.
        - Rekomendasi Data: Data curah hujan dan cuaca perlu diupdate secara berkala untuk akurasi lebih baik.
    """)

# Fungsi untuk halaman Decision
def decision_content():
    st.write("Halaman ini memberikan keputusan berdasarkan analisis data.")
    st.write("""
        - Keputusan: Berdasarkan prediksi curah hujan, disarankan menanam padi pada bulan Desember.
        - Konteks: Wilayah dengan kelembaban di atas 80% dan curah hujan tinggi cocok untuk pertanian basah.
    """)

# Fungsi untuk halaman Conclusion
def conclusion_content():
    st.write("Halaman ini memberikan kesimpulan dari analisis data.")
    st.write("""
        - Kesimpulan: Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - Tindak Lanjut: Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Fungsi untuk halaman Home
def home():
    # Menampilkan Header/Banner
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Header Banner" width="800">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Judul dan Deskripsi
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")    

    # Menampilkan Abstrak
    st.subheader("Abstrak")
    st.write("""
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
    """)

    # Menampilkan Gambar Arsitektur Sistem
    st.subheader("Arsitektur Sistem")
    st.markdown(
        """
        <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/arsitektur sistem70.drawio.png" alt="Gambar Arsi" width="700">
        """,
        unsafe_allow_html=True,
    )

    # Penjelasan Arsitektur Sistem
    st.write("""
        Arsitektur sistem ini menggambarkan alur kerja aplikasi dari pengambilan data,
        preprocessing, hingga analisis. Data curah hujan diolah menggunakan beberapa
        metode untuk menghasilkan prediksi yang akurat. Komponen utama meliputi:
        - Pengumpulan Data: Data cuaca harian dari BMKG atau citra awan.
        - Preprocessing: Normalisasi data, augmentasi gambar, dan transformasi fitur.
        - Model Analitik: Penggunaan algoritma ARIMA untuk data waktu, CNN untuk klasifikasi gambar,
          dan clustering dengan K-Means untuk pengelompokan data.
        - Output: Prediksi cuaca atau rekomendasi tindakan untuk sektor pertanian.
    """)

    # Menampilkan Insight, Decision, dan Conclusion
    st.subheader("Insight")
    insight_content()

    st.subheader("Decision")
    decision_content()

    st.subheader("Conclusion")
    conclusion_content()

# Fungsi untuk Clustering K-Means
def clustering_kmeans():
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Halaman ini akan berisi implementasi clustering data curah hujan dengan K-Means.")
    
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

    # Fungsi untuk heatmap
    def create_heatmap(data):
        map_heatmap = folium.Map(
            location=[data['Latitude'].mean(), data['Longitude'].mean()],
            zoom_start=6
        )
        cluster_colors = {0: "red", 1: "blue", 2: "green"}
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
        plugins.HeatMap(data[['Latitude', 'Longitude', 'RR']].values.tolist(), radius=15).add_to(map_heatmap)
        folium.LayerControl().add_to(map_heatmap)
        return map_heatmap

    # Load Data
    df = load_data()
    cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
    encoder = LabelEncoder()
    cleaned_kota['KOTA'] = encoder.fit_transform(df['KOTA'])

    # Metode Elbow
    st.subheader("Metode Elbow")
    elbow_method(cleaned_kota)

    # Hasil Clustering
    st.subheader("Hasil Clustering K-Means")
    df['cluster'] = df['cluster'].replace({0: 2, 1: 0, 2: 1})
    st.dataframe(df.head())

    # Statistik Deskriptif per Cluster
    st.subheader("Statistik Deskriptif per Cluster")
    col_drop = ['Tanggal', 'ddd_car', 'Latitude', 'Longitude', 'KOTA']
    desc_stats = (
        df.drop(col_drop, axis=1)
        .groupby('cluster')
        .aggregate(['mean', 'std', 'min', 'median', 'max'])
        .transpose()
    )
    st.dataframe(desc_stats)

    # Heatmap
    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st_folium(heatmap, width=700, height=500)

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
if menu == "Home":
    home()
elif menu == "Prediksi Dengan Metode ARIMA":
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
elif menu == "Klasifikasi Citra Dengan Metode CNN":
    st.title("Klasifikasi Citra Awan Curah Hujan dengan Metode CNN")
elif menu == "Klasifikasi Dengan Decision Trees":
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
elif menu == "Clustering Dengan Metode K-Means":
    clustering_kmeans()

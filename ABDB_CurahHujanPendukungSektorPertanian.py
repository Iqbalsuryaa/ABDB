import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import folium
from folium import plugins
from streamlit_folium import st_folium
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Fungsi untuk halaman Home
def home():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Header Banner" width="800">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")    
    st.subheader("Abstrak")
    st.write("""
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
    """)
    st.subheader("Arsitektur Sistem")
    st.markdown(
        """
        <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/arsitektur sistem70.drawio.png" alt="Gambar Arsi" width="700">
        """,
        unsafe_allow_html=True,
    )
    st.write("""
        Arsitektur sistem ini menggambarkan alur kerja aplikasi dari pengambilan data,
        preprocessing, hingga analisis. Data curah hujan diolah menggunakan beberapa
        metode untuk menghasilkan prediksi yang akurat.
    """)
    st.subheader("Insight")
    st.write("""
        - Analisis Tren: Curah hujan cenderung meningkat di musim penghujan.
        - Pola Cuaca: Suhu dan kelembaban memiliki korelasi signifikan terhadap curah hujan.
        - Rekomendasi Data: Data curah hujan dan cuaca perlu diupdate secara berkala untuk akurasi lebih baik.
    """)
    st.subheader("Decision")
    st.write("""
        - Keputusan: Berdasarkan prediksi curah hujan, disarankan menanam padi pada bulan Desember.
        - Konteks: Wilayah dengan kelembaban di atas 80% dan curah hujan tinggi cocok untuk pertanian basah.
    """)
    st.subheader("Conclusion")
    st.write("""
        - Kesimpulan: Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - Tindak Lanjut: Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Sidebar Menu
st.sidebar.title("Pengaturan")
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
    st.write("Halaman ini akan berisi implementasi prediksi curah hujan dengan ARIMA.")
    
    uploaded_file = st.file_uploader("Upload Dataset (format .xlsx):", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(df.head())

        # EDA dan Pembersihan Data
        st.write("### Pembersihan Data:")
        try:
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(
                df['tahun'].astype(str) + '-' +
                df['bulan'].astype(str) + '-' +
                df['Tanggal'].astype(str)
            )
            df.set_index('Date', inplace=True)
            df = df.asfreq('D')  # Set frekuensi data ke harian

            # Memastikan kolom 'RR' ada
            if 'RR' in df.columns:
                st.write("#### Visualisasi Curah Hujan")
                df['RR'].plot(figsize=(12, 6))
                st.pyplot()
                
                # Implementasi ARIMA
                st.write("### Prediksi Curah Hujan dengan ARIMA")
                model = ARIMA(df['RR'], order=(5, 1, 0))
                model_fit = model.fit()
                st.write(model_fit.summary())

                # Prediksi
                forecast = model_fit.forecast(steps=30)
                st.write("### Prediksi 30 Hari ke Depan:")
                st.write(forecast)

                # Visualisasi Hasil Prediksi
                plt.figure(figsize=(12, 6))
                plt.plot(df.index, df['RR'], label='Data Asli')
                plt.plot(pd.date_range(df.index[-1], periods=31, freq='D')[1:], forecast, label='Prediksi', color='red')
                plt.title('Prediksi Curah Hujan Menggunakan ARIMA')
                plt.legend()
                st.pyplot()

        except Exception as e:
            st.write(f"Error: {e}")

elif menu == "Klasifikasi Citra Dengan Metode CNN":
    st.title("Klasifikasi Citra Awan Curah Hujan dengan Metode CNN")
    st.write("Halaman ini akan berisi implementasi klasifikasi citra awan dengan CNN.")

elif menu == "Klasifikasi Dengan Decision Trees":
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini akan berisi implementasi klasifikasi cuaca dengan Decision Trees.")
    
    uploaded_file = st.file_uploader("Upload Dataset Cuaca (CSV):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(df.head())

        # Preprocessing
        X = df.drop(['RR'], axis=1)  # Fitur
        y = df['RR']  # Target

        # Train Decision Tree
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        st.write("### Hasil Prediksi:")
        predictions = clf.predict(X)
        st.write(predictions)

        st.write("### Classification Report:")
        st.text(classification_report(y, predictions))

elif menu == "Clustering Dengan Metode K-Means":
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Halaman ini akan berisi implementasi clustering data curah hujan dengan K-Means.")

    # Load Data
    df = load_data()
    cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
    encoder = LabelEncoder()
    cleaned_kota['KOTA'] = encoder.fit_transform(df['KOTA'])

    st.subheader("Metode Elbow")
    elbow_method(cleaned_kota)
    # Hasil Clustering
    st.subheader("Hasil Clustering K-Means")
    rename = {0: 2, 1: 0, 2: 1}
    df['cluster'] = df['cluster'].replace(rename)
    st.markdown(""" 
    ### Cluster Berdasarkan Curah Hujan:
    1. *Cluster 0*: Curah hujan tinggi (musim hujan).
    2. *Cluster 2*: Curah hujan sedang (cuaca normal).
    3. *Cluster 1*: Curah hujan rendah (musim kering).
    """)
    st.dataframe(df.head())
    
    df['cluster'] = df['cluster'].replace(rename)
    st.markdown(""" 
    ### Cluster Berdasarkan Curah Hujan:
    1. *Cluster 0*: Curah hujan tinggi (musim hujan).
    2. *Cluster 2*: Curah hujan sedang (cuaca normal).
    3. *Cluster 1*: Curah hujan rendah (musim kering).
    """)

    
    st.subheader("Statistik Deskriptif per Cluster")
    col_drop = ['Tanggal', 'ddd_car', 'Latitude', 'Longitude', 'KOTA']
    desc_stats = (
        df.drop(col_drop, axis=1)
        .groupby('cluster')
        .aggregate(['mean', 'std', 'min', 'median', 'max'])
        .transpose()
    )
    st.dataframe(desc_stats)

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
    1. *Cluster 0 (Curah Hujan Tinggi - Musim Hujan):*
       - Daerah dengan intensitas curah hujan tinggi, cluster ini menunjukkan daerah-daerah yang mengalami curah hujan tinggi. Biasanya cluster ini mewakili wilayah yang terletak di musim hujan atau daerah dengan iklim tropis yang sering mengalami hujan deras.
       - Ciri-ciri: Area yang termasuk dalam cluster ini akan menunjukkan intensitas curah hujan yang lebih tinggi (lebih dari rata-rata), yang biasanya terkait dengan musim hujan, sering terjadi pada musim hujan dengan curah hujan di atas rata-rata.
    2. *Cluster 2 (Curah Hujan Sedang - Cuaca Normal):*
       - Daerah dengan curah hujan sedang, biasanya mencerminkan cuaca normal atau transisi musim, cluster ini berisi daerah-daerah dengan curah hujan sedang, yang biasanya terjadi pada cuaca normal atau musim transisi antara musim hujan dan kemarau.
       - Ciri-ciri: Wilayah yang termasuk dalam cluster ini memiliki tingkat curah hujan yang cukup stabil, tidak terlalu tinggi dan juga tidak terlalu rendah, mencerminkan cuaca yang tidak ekstrem.
    3. *Cluster 1 (Curah Hujan Rendah - Musim Kering):*
       - Daerah dengan intensitas curah hujan rendah, cluster ini mencakup daerah-daerah yang mengalami curah hujan rendah, yang biasanya terjadi pada musim kemarau atau wilayah yang lebih kering.
       - Ciri-ciri: Area yang termasuk dalam cluster ini cenderung mengalami sedikit hujan atau bahkan tidak ada hujan sama sekali dalam periode tertentu, mencerminkan musim kering atau iklim yang lebih kering, sering terjadi pada musim kemarau atau di wilayah yang lebih kering.
    """)

    st.subheader("Heatmap")
    heatmap = create_heatmap(df)
    st_folium(heatmap, width=700, height=500)
    
    # Penjelasan Warna pada Heatmap
    st.markdown(""" 
    ### Penjelasan Warna pada Heatmap:
    - Merah Tua / Oranye : Menunjukkan daerah dengan curah hujan yang tinggi, biasanya terjadi pada musim hujan atau daerah tropis dengan intensitas hujan tinggi, Lokasi-lokasi yang lebih intens curah hujannya akan tampak dengan warna yang lebih gelap, daerah dengan intensitas curah hujan tinggi sering kali berwarna merah tua atau oranye terang, menunjukkan curah hujan yang sangat tinggi.
    - Kuning / Hijau Muda : Daerah dengan curah hujan sedang, mencerminkan cuaca normal atau transisi musim, Warna-warna seperti kuning atau hijau muda menandakan intensitas hujan yang lebih rendah dibandingkan dengan daerah merah.
    - Biru Tua / Biru Muda : Daerah dengan curah hujan rendah, sering terjadi pada musim kemarau atau wilayah kering dan Ini biasanya mewakili lokasi-lokasi yang memiliki sedikit atau bahkan tidak ada hujan (seperti musim kemarau). Warna biru gelap atau biru muda ini menandakan intensitas hujan yang sangat rendah.
    """)

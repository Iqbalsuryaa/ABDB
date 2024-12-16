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
import joblib


# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        return pd.read_csv('Hasilcluster_result.csv')
    except FileNotFoundError:
        st.error("File 'Hasilcluster_result.csv' tidak ditemukan.")
        return None


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
        "Klasifikasi Dengan Navie Bayes",
        "Clustering Dengan Metode K-Means",
    )
)

# Menentukan menu yang dipilih
if menu == "Home":
    home()
elif menu == "Prediksi Dengan Metode ARIMA":
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini akan berisi implementasi prediksi curah hujan dengan ARIMA.")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset (format .xlsx):", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(df.head())

    # EDA dan Pembersihan Data
    st.write("### Pembersihan Data:")
    # Teruskan dengan kode ARIMA yang sudah ada sebelumnya...

elif menu == "Klasifikasi Dengan Navie Bayes":
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Navie Bayes")
    st.write("Halaman ini akan berisi implementasi klasifikasi cuaca dengan  Navie Bayes.")
    # Perbaiki kode model Naive Bayes yang terletak di luar if blok...
# Muat model
MODEL_PATH = "naive_bayes_model.pkl"
model = joblib.load(MODEL_PATH)

# Muat dataset
DATA_PATH = "weather_classification_data.csv"
data = pd.read_csv(DATA_PATH)

# Fungsi preprocessing
def preprocess_input(df, input_data):
    # Bersihkan nama kolom
    df.columns = df.columns.str.strip()

    # Isi nilai yang hilang
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna("unknown")

    # Label encode kolom target untuk konsistensi
    label_encoder = LabelEncoder()
    df['WeatherType'] = label_encoder.fit_transform(df['WeatherType'])

    # One-hot encoding untuk fitur kategorikal selain target
    df_encoded = pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=['object']).columns if col != 'WeatherType'], drop_first=True)

    # Sesuaikan fitur input dengan struktur data pelatihan
    X_encoded = pd.DataFrame([input_data], columns=df_encoded.drop(columns=['WeatherType']).columns).fillna(0)
    return X_encoded

# Antarmuka aplikasi Streamlit
st.title("Aplikasi Klasifikasi Cuaca Mengunakan Metode Navie Bayes")
st.write("Aplikasi ini memprediksi jenis cuaca berdasarkan fitur input.")

# Buat input field untuk pengguna
user_input = {}
for col in data.columns[:-1]:  # Kecualikan kolom target
    if data[col].dtype == 'object':
        user_input[col] = st.text_input(f"{col}", "Masukkan nilai")
    else:
        user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Klasifikasikan Cuaca"):
    try:
        # Preproses input
        processed_input = preprocess_input(data, user_input)

        # Prediksi
        prediction = model.predict(processed_input)

        # Dekode hasil prediksi
        label_encoder = LabelEncoder()
        label_encoder.fit(data['WeatherType'])
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        st.success(f"Jenis Cuaca yang Diprediksi: {predicted_label}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

st.write("\n---\n**Informasi Model**")
st.write(f"Lokasi model: {MODEL_PATH}")
st.write(f"Lokasi data: {DATA_PATH}")

st.write("\n---\nDikembangkan oleh [Nama Anda]")

import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import seaborn as sns
from sklearn.cluster import KMeans
import joblib


# Fungsi untuk memuat data
def load_data():
    # Misalkan data ini diambil dari file CSV yang di-upload
    uploaded_file = st.file_uploader("Upload Dataset (format .csv):", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.error("Silakan unggah file dataset.")
        return None

# Fungsi untuk metode Elbow pada K-Means
def elbow_method(df):
    # Mencoba berbagai nilai K untuk menemukan yang terbaik
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        distortions.append(kmeans.inertia_)
    
    # Plot hasil Elbow
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, distortions, marker='o')
    plt.title('Metode Elbow untuk Menentukan K')
    plt.xlabel('Jumlah Cluster (K)')
    plt.ylabel('Distorsi')
    plt.grid(True)
    st.pyplot(plt)

# Fungsi untuk membuat Heatmap
def create_heatmap(df):
    # Menggunakan folium untuk membuat heatmap (asumsi df memiliki kolom Latitude dan Longitude)
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
    folium.plugins.HeatMap(heat_data).add_to(m)
    return m


# Fungsi utama aplikasi
st.title('Aplikasi Cuaca dan Prediksi')

# Sidebar menu
menu = st.sidebar.selectbox("Pengaturan", ["Home", "Prediksi Dengan Metode ARIMA", "Klasifikasi Citra Dengan Metode CNN", "Klasifikasi Dengan Metode Naive Bayes", "Clustering K-Means"])

if menu == "Home":
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
    st.write(
        """
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
        """
    )
    st.subheader("Arsitektur Sistem")
    st.markdown(
        """
        <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/arsitektur sistem70.drawio.png" alt="Gambar Arsi" width="700">
        """,
        unsafe_allow_html=True,
    )
    st.write(
        """
        Arsitektur sistem ini menggambarkan alur kerja aplikasi dari pengambilan data,
        preprocessing, hingga analisis. Data curah hujan diolah menggunakan beberapa
        metode untuk menghasilkan prediksi yang akurat.
        """
    )

elif menu == "Prediksi Dengan Metode ARIMA":
    st.write("### Prediksi Cuaca Menggunakan Metode ARIMA")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload Dataset (format .xlsx):", type="xlsx")
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.write("### Dataset Preview:")
        st.write(df.head())

        # EDA dan Pembersihan Data
        st.write("### Pembersihan Data:")

        try:
            # Menghapus spasi di nama kolom
            df.columns = df.columns.str.strip()

            # Membuat kolom tanggal
            df['Date'] = pd.to_datetime(
                df['tahun'].astype(str) + '-' +
                df['bulan'].astype(str) + '-' +
                df['Tanggal'].astype(str)
            )
            df.set_index('Date', inplace=True)
            df = df.asfreq('D')  # Set frekuensi data ke harian

            # Memastikan kolom 'RR Tuban' ada
            if 'RR Tuban' not in df.columns:
                st.error("Kolom 'RR Tuban' tidak ditemukan dalam dataset. Pastikan nama kolom sesuai.")
                st.stop()

            # Mengonversi data ke numerik dan mengisi nilai NaN
            df['RR Tuban'] = pd.to_numeric(df['RR Tuban'], errors='coerce')
            df['RR Tuban'] = df['RR Tuban'].ffill().bfill()

        except Exception as e:
            st.error(f"Terjadi kesalahan dalam pembersihan data: {e}")
            st.stop()

        st.write("Data setelah pembersihan:")
        st.write(df.head())

        # Pembagian Data
        train_data = df['RR Tuban'][:-30]
        test_data = df['RR Tuban'][-30:]

        # Parameter ARIMA
        p = st.number_input('Masukkan nilai p (AutoRegressive order):', min_value=0, value=1, step=1)
        d = st.number_input('Masukkan nilai d (Difference order):', min_value=0, value=1, step=1)
        q = st.number_input('Masukkan nilai q (Moving Average order):', min_value=0, value=1, step=1)

        if st.button('Lakukan Forecasting'):
            try:
                # Melatih Model ARIMA
                model = ARIMA(train_data, order=(p, d, q))
                model_fit = model.fit()

                # Peramalan
                forecast = model_fit.forecast(steps=30)

                # Evaluasi Hasil
                mae = mean_absolute_error(test_data, forecast)
                mse = mean_squared_error(test_data, forecast)
                rmse = np.sqrt(mse)

                # Visualisasi
                st.write("### Hasil Peramalan:")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(train_data, label='Data Pelatihan')
                ax.plot(test_data, label='Data Aktual', color='orange')
                ax.plot(test_data.index, forecast, label='Peramalan', color='green')
                ax.set_title('Peramalan Cuaca dengan ARIMA')
                ax.set_xlabel('Tanggal')
                ax.set_ylabel('Curah Hujan (RR Tuban)')
                ax.legend()
                st.pyplot(fig)

                # Menampilkan Evaluasi
                st.write("### Evaluasi Model:")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

            except Exception as e:
                st.error(f"Terjadi kesalahan selama proses peramalan: {e}")

elif menu == "Klasifikasi Dengan Metode Navie Bayes":
    st.write("### Klasifikasi Cuaca Menggunakan Naive Bayes")

    # Muat model Naive Bayes
    MODEL_PATH = "naive_bayes_model.pkl"
    model = joblib.load(MODEL_PATH)

    # Muat dataset
    DATA_PATH = "weather_classification_data.csv"
    data = pd.read_csv(DATA_PATH)

    # Fungsi preprocessing
    def preprocess_input(df, input_data):
        df.columns = df.columns.str.strip()
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna("unknown")
        label_encoder = LabelEncoder()
        df['WeatherType'] = label_encoder.fit_transform(df['WeatherType'])
        df_encoded = pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=['object']).columns if col != 'WeatherType'], drop_first=True)
        X_encoded = pd.DataFrame([input_data], columns=df_encoded.drop(columns=['WeatherType']).columns).fillna(0)
        return X_encoded

    # Antarmuka aplikasi Streamlit
    st.title("Aplikasi Klasifikasi Cuaca Menggunakan Metode Naive Bayes")
    st.write("Aplikasi ini memprediksi jenis cuaca berdasarkan fitur input.")

    # Buat input field untuk pengguna
    user_input = {}
    user_input['Feature1'] = st.number_input("Masukkan nilai Fitur 1:")
    user_input['Feature2'] = st.number_input("Masukkan nilai Fitur 2:")
    user_input['Feature3'] = st.number_input("Masukkan nilai Fitur 3:")
    # (Lanjutkan dengan inputan lainnya sesuai dataset)

    if st.button("Prediksi"):
        # Preprocessing data
        X_encoded = preprocess_input(data, user_input)
        prediction = model.predict(X_encoded)
        st.write(f"Jenis cuaca yang diprediksi: {prediction[0]}")

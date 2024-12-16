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
from sklearn.preprocessing import LabelEncoder
from folium import plugins
import joblib


# Fungsi utama aplikasi
st.title('Aplikasi Cuaca dan Prediksi')

# Sidebar menu
menu = st.sidebar.selectbox("Pengaturan", ["Home", "Prediksi Dengan Metode ARIMA", "Klasifikasi Citra Dengan Metode CNN", "Klasifikasi Dengan Metode Navie Bayes", "Clustering K-Means"])

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
    for col in data.columns[:-1]:
        if data[col].dtype == 'object':
            user_input[col] = st.text_input(f"{col}", "Masukkan nilai")
        else:
            user_input[col] = st.number_input(f"{col}", value=0.0)

    if st.button("Klasifikasikan Cuaca"):
        try:
            processed_input = preprocess_input(data, user_input)
            prediction = model.predict(processed_input)
            label_encoder = LabelEncoder()
            label_encoder.fit(data['WeatherType'])
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            st.success(f"Jenis Cuaca yang Diprediksi: {predicted_label}")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")

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
    st.dataframe(df)

    # Map visualization (Optional)
    st.subheader("Visualisasi Cluster pada Peta")
    create_heatmap(df)


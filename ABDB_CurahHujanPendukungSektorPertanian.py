import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium

# Fungsi utama aplikasi
st.title('Aplikasi Cuaca dan Prediksi')

# Sidebar menu
menu = st.sidebar.selectbox("Pengaturan", ["Home", "Prediksi Dengan Metode ARIMA", "Klasifikasi Citra Dengan Metode CNN", "Klasifikasi Dengan", "Clustering K-Means"])

if menu == "Home":
    st.write("### Selamat datang di aplikasi prediksi dan analisis cuaca!")
    st.write("Gunakan menu di sebelah kiri untuk menjelajahi fitur aplikasi.")

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

elif menu == "Visualisasi Heatmap":
    st.write("### Visualisasi Heatmap")
    st.write("Fitur ini sedang dalam pengembangan.")

elif menu == "Clustering K-Means":
    st.write("### Clustering K-Means")
    st.write("Fitur ini sedang dalam pengembangan.")

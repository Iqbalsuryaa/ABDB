import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

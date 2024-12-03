import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load model and data
MODEL_PATH = "decision_tree_model.pkl"
DATA_PATH = "weather_classification_data.csv"

# Fungsi utama aplikasi
def main():
    st.title("Aplikasi Klasifikasi Cuaca dengan Decision Tree")

    # Load dataset
    st.sidebar.header("Dataset")
    df = pd.read_csv(DATA_PATH)

    # Informasi dataset
    st.header("Informasi Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()
    buffer.close()
    st.text(info_text)

    st.subheader("Tampilan Dataset")
    st.write(df.head())

    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    # Distribusi kelas target
    st.subheader("Distribusi Kelas Target")
    plt.figure(figsize=(6, 4))
    df['WeatherType'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Distribusi Kelas Target')
    plt.xlabel('Kelas')
    plt.ylabel('Frekuensi')
    st.pyplot(plt)

    # Load model Decision Tree
    model = joblib.load(MODEL_PATH)

    # Input fitur untuk prediksi
    st.sidebar.header("Input Fitur")
    features = df.drop(columns=['WeatherType']).columns

    input_data = {}
    for feature in features:
        input_data[feature] = st.sidebar.number_input(
            f"{feature}",
            float(df[feature].min()),
            float(df[feature].max()),
            float(df[feature].mean())
        )

    input_df = pd.DataFrame([input_data])

    st.subheader("Input Data untuk Prediksi")
    st.write(input_df)

    # Prediksi
    if st.button("Prediksi Cuaca"):
        prediction = model.predict(input_df)[0]
        st.subheader("Hasil Prediksi")
        st.write(f"Kelas Cuaca: {prediction}")

    # Visualisasi Decision Tree
    st.subheader("Visualisasi Decision Tree")
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=features, class_names=np.unique(df['WeatherType'].astype(str)), filled=True)
    st.pyplot(plt)

if __name__ == "__main__":
    main()

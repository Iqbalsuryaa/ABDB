import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Fungsi utama
def main():
    st.title("Weather Classification with Decision Tree")
    st.write("Aplikasi ini memprediksi jenis cuaca berdasarkan data cuaca yang diberikan.")

    # Load model
    model_path = "decision_tree_model.pkl"
    model = joblib.load(model_path)

    # Load dataset
    csv_path = "weather_classification_data.csv"
    df = pd.read_csv(csv_path)

    # Validasi dataset
    st.subheader("Dataframe Preview")
    st.dataframe(df.head())

    st.subheader("Kolom dan Informasi Dataset")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    # Preprocessing dataset
    df = df.fillna(df.median(numeric_only=True))  # Isi nilai numerik dengan median
    df = df.fillna("unknown")  # Isi nilai kategori dengan "unknown"

    # Encoding kolom target
    label_encoder = LabelEncoder()
    df["WeatherType"] = label_encoder.fit_transform(df["WeatherType"])

    # One-Hot Encoding untuk fitur kategorikal
    df_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=["object"]).columns, drop_first=True)

    # Memisahkan fitur (X) dan target (y)
    X = df_encoded.drop(columns=["WeatherType"])
    y = df["WeatherType"]

    # Filter fitur dengan slider
    st.sidebar.header("Filter Features")
    filtered_data = X.copy()

    for feature in X.columns:
        if pd.api.types.is_numeric_dtype(df[feature]):
            min_value = float(df[feature].min())
            max_value = float(df[feature].max())
            selected_range = st.sidebar.slider(
                f"Filter {feature}",
                min_value=min_value,
                max_value=max_value,
                value=(min_value, max_value)
            )
            filtered_data = filtered_data[(df[feature] >= selected_range[0]) & (df[feature] <= selected_range[1])]

    st.subheader("Filtered Dataframe Preview")
    st.dataframe(filtered_data)

    # Input untuk prediksi
    st.sidebar.header("Input Data untuk Prediksi")
    input_data = []
    for feature in X.columns:
        if pd.api.types.is_numeric_dtype(df[feature]):
            value = st.sidebar.number_input(f"{feature}", float(df[feature].min()), float(df[feature].max()))
            input_data.append(value)
        else:
            value = st.sidebar.selectbox(f"{feature}", df[feature].unique())
            input_data.append(value)

    # Prediksi
    if st.sidebar.button("Predict"):
        prediction = model.predict([input_data])[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        st.subheader("Prediction Result")
        st.write(f"Predicted Weather Type: **{predicted_label}**")

if __name__ == "__main__":
    main()

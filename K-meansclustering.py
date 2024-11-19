import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Judul Aplikasi
st.title("K-Means Clustering untuk Dataset Curah Hujan")

# Upload Dataset
uploaded_file = st.file_uploader("Upload Dataset (.xlsx)", type="xlsx")
if uploaded_file:
    # Membaca Dataset
    @st.cache_data
    def load_data(file):
        return pd.read_excel(file)

    df = load_data(uploaded_file)
    st.write("Data Awal:")
    st.dataframe(df.head())

    # Analisis Missing Value
    st.subheader("Analisis Missing Value")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    # Plot Missing Values (jika ada)
    if df.isnull().values.any():
        sns.set(rc={"figure.figsize": (8, 4)})
        column_with_nan = df.columns[df.isnull().any()]
        percent_nan = [round(df[col].isnull().sum() * 100 / len(df), 2) for col in column_with_nan]
        tab = pd.DataFrame({"Column": column_with_nan, "Percent_NaN": percent_nan})
        p = sns.barplot(x="Percent_NaN", y="Column", data=tab, edgecolor="black", color="deepskyblue")
        p.set_title("Persentasi Missing Value per Kolom\n", fontsize=15)
        p.set_xlabel("\nPersentase Missing Value")
        plt.tight_layout()
        st.pyplot(plt)
        plt.clf()

    # Data Cleaning
    st.subheader("Data Cleaning")
    df_clean = df.dropna(axis=0).reset_index(drop=True)
    st.write("Data setelah menghapus missing values:")
    st.dataframe(df_clean.head())

    # Scaling Features
    st.subheader("Scaling Data")
    num_features = df_clean.select_dtypes(include=["float64", "int64"]).columns
    scaler = StandardScaler()
    df_scaled = df_clean.copy()
    df_scaled[num_features] = scaler.fit_transform(df_clean[num_features])
    st.write("Data setelah Scaling:")
    st.dataframe(df_scaled.head())

    # K-Means Clustering
    st.subheader("Clustering dengan K-Means")
    n_clusters = st.slider("Pilih Jumlah Cluster", min_value=2, max_value=10, value=3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_scaled['Cluster'] = kmeans.fit_predict(df_scaled[num_features])

    st.write("Hasil Clustering:")
    st.dataframe(df_scaled.head())

    # Plot Pairplot
    st.subheader("Visualisasi Clustering")
    sns.pairplot(df_scaled, hue="Cluster", diag_kind="kde", palette="tab10")
    st.pyplot(plt)
    plt.clf()

    # Save Model
    st.subheader("Simpan Model")
    if st.button("Simpan Model"):
        model_path = "kmeans_model.pkl"
        joblib.dump(kmeans, model_path)
        st.success(f"Model berhasil disimpan sebagai `{model_path}`!")

    # Load Model
    st.subheader("Load Model yang Sudah Ada")
    model_file = st.file_uploader("Upload Model (.pkl)", type="pkl")
    if model_file:
        loaded_model = joblib.load(model_file)
        st.success("Model berhasil dimuat!")

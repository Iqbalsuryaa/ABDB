import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline

# Judul Aplikasi
st.title("Clustering Curah Hujan dengan K-Means")

# Fungsi untuk visualisasi missing value
def visualize_missing_values(df):
    column_with_nan = df.columns[df.isnull().any()]
    percent_nan = [(col, round(df[col].isnull().sum() * 100 / len(df), 2)) for col in column_with_nan]
    tab = pd.DataFrame(percent_nan, columns=["Kolom", "Persentase Missing Value"]).sort_values(by="Persentase Missing Value", ascending=False)
    st.write(tab)
    sns.barplot(x="Persentase Missing Value", y="Kolom", data=tab, edgecolor="black", color="deepskyblue")
    plt.title("Persentase Missing Value per Kolom")
    st.pyplot()

# File uploader
uploaded_file = st.file_uploader("Upload dataset (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Baca dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Dataset Asli")
        st.dataframe(df.head())

        st.write("Dimensi Data:", df.shape)
        st.write("Informasi Data:")
        st.text(df.info())
        st.write("Deskripsi Statistik:")
        st.write(df.describe())

        st.subheader("Analisis Missing Value")
        visualize_missing_values(df)

        # Preprocessing
        st.subheader("Pra-pemrosesan Data")
        imputer = SimpleImputer(strategy="mean")
        df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
        st.write("Data Setelah Mengisi Nilai Hilang:")
        st.dataframe(df_imputed.head())

        # Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_imputed)
        df_scaled = pd.DataFrame(scaled_features, columns=df_imputed.columns)

        # Label Encoding untuk kolom non-numerik
        if 'KOTA' in df.columns:
            encoder = LabelEncoder()
            df['KOTA'] = encoder.fit_transform(df['KOTA'])

        # Clustering
        st.subheader("Clustering")
        features = st.multiselect("Pilih Fitur untuk Clustering:", options=df_scaled.columns)
        if features:
            n_clusters = st.slider("Jumlah Cluster (k):", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(df_scaled[features])

            df['Cluster'] = kmeans.labels_
            st.write("Dataset dengan Cluster:")
            st.dataframe(df)

            # Elbow Method
            wcss = []
            for i in range(1, 11):
                kmeans_elbow = KMeans(n_clusters=i, random_state=42)
                kmeans_elbow.fit(df_scaled[features])
                wcss.append(kmeans_elbow.inertia_)

            plt.figure()
            plt.plot(range(1, 11), wcss, marker='o', color='blue')
            plt.title('Elbow Method')
            plt.xlabel('Number of Clusters')
            plt.ylabel('WCSS')
            st.pyplot()

            # Silhouette Score & Davies-Bouldin Index
            sil_score = silhouette_score(df_scaled[features], kmeans.labels_)
            dbi_score = davies_bouldin_score(df_scaled[features], kmeans.labels_)
            st.write(f"Silhouette Score: {sil_score:.4f}")
            st.write(f"Davies-Bouldin Index: {dbi_score:.4f}")

            # Visualisasi Cluster
            if len(features) == 2:
                # Scatter plot untuk dua fitur
                plt.figure()
                sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df['Cluster'], palette='viridis')
                plt.title("Visualisasi Cluster dengan Dua Fitur")
                st.pyplot()
            elif len(features) == 4:
                # Pair plot untuk empat fitur
                st.write("Visualisasi Clustering dengan Empat Fitur:")
                df_viz = df[features + ['Cluster']]
                pair_plot = sns.pairplot(df_viz, hue='Cluster', palette='viridis', diag_kind='kde', corner=True)
                st.pyplot(pair_plot.fig)

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan unggah file dataset terlebih dahulu.")

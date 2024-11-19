import subprocess
import sys

# Fungsi untuk memastikan pustaka terinstal
def install_and_import(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"Menginstal pustaka {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Pustaka {package_name} berhasil diinstal.")

# Instal pustaka yang diperlukan
required_packages = ["streamlit", "pandas", "seaborn", "matplotlib", "openpyxl", "scikit-learn"]
for package in required_packages:
    install_and_import(package)

# Import pustaka
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Streamlit App
st.title("Analisis Data Cuaca dengan Streamlit")
st.sidebar.header("Pengaturan")

# File Uploader
uploaded_file = st.sidebar.file_uploader("Unggah file Excel Anda", type=["xlsx"])

if uploaded_file:
    # Load Dataset
    df = pd.read_excel(uploaded_file)
    st.write("### Dataframe")
    st.dataframe(df.head())
    
    # Informasi Dataset
    st.write("### Informasi Dataset")
    st.write(f"Dimensi dataset: {df.shape}")
    buffer = []
    df.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    # Analisis Missing Values
    st.write("### Missing Value Analysis")
    missing = df.isnull().sum()
    st.write("Jumlah Missing Value per Kolom:")
    st.write(missing)
    st.write("Persentase Missing Value:")
    st.bar_chart(missing / len(df) * 100)
    
    # Deteksi Outliers
    st.write("### Deteksi Outliers dengan Boxplot")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df[numeric_cols], ax=ax)
    st.pyplot(fig)

    # Matriks Korelasi
    st.write("### Matriks Korelasi")
    df_num = df.select_dtypes(exclude=["object"])
    corr = df_num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # K-Means Clustering
    st.write("### Clustering dengan K-Means")
    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
    
    # Preprocessing Data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num.dropna())
    
    # Metode Elbow
    st.write("#### Metode Elbow")
    wcss = []
    range_clusters = range(1, 11)
    for i in range_clusters:
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range_clusters, wcss, marker='o')
    ax.set_title("Elbow Method")
    ax.set_xlabel("Jumlah Cluster")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)
    
    # K-Means Model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df_scaled)
    df["Cluster"] = kmeans.labels_
    
    # Pairplot Berdasarkan Cluster
    st.write("#### Pairplot Berdasarkan Cluster")
    fig = sns.pairplot(data=df, hue="Cluster", palette="tab10")
    st.pyplot(fig)

    # Heatmap untuk Cluster Result
    st.write("#### Heatmap untuk Cluster Result")
    cluster_means = df.groupby("Cluster").mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cluster_means, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Rata-Rata Fitur dalam Setiap Cluster")
    st.pyplot(fig)

    # Visualisasi Distribusi Cluster
    st.write("#### Distribusi Cluster")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x="Cluster", data=df, ax=ax, palette="tab10")
    ax.set_title("Distribusi Jumlah Data dalam Cluster")
    st.pyplot(fig)

    # Evaluasi Model
    silhouette = silhouette_score(df_scaled, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(df_scaled, kmeans.labels_)
    st.write(f"Silhouette Score: {silhouette:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

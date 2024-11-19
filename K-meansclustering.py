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
    
    # Basic Info
    st.write("### Informasi Dataset")
    st.write(f"Dimensi dataset: {df.shape}")
    st.write(df.info())

    # Missing Values
    st.write("### Missing Value Analysis")
    missing = df.isnull().sum()
    st.write("Jumlah Missing Value per Kolom:")
    st.write(missing)
    st.write("Persentase Missing Value:")
    st.bar_chart(missing / len(df) * 100)
    
    # Outliers
    st.write("### Deteksi Outliers dengan Boxplot")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df[numeric_cols], ax=ax)
    st.pyplot(fig)

    # Correlation Matrix
    st.write("### Matriks Korelasi")
    df_num = df.select_dtypes(exclude=["object"])
    corr = df_num.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # K-Means Clustering
    st.write("### Clustering dengan K-Means")
    n_clusters = st.sidebar.slider("Jumlah Cluster", min_value=2, max_value=10, value=3)
    
    # Preprocessing
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_num.dropna())
    
    # Elbow Method
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
    
    st.write("#### Pairplot Berdasarkan Cluster")
    fig = sns.pairplot(data=df, hue="Cluster", palette="tab10")
    st.pyplot(fig)

    # Evaluasi
    silhouette = silhouette_score(df_scaled, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(df_scaled, kmeans.labels_)
    st.write(f"Silhouette Score: {silhouette:.2f}")
    st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

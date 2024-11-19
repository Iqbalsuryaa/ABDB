import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import numpy as np

# Fungsi untuk memuat dataset
@st.cache
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

# Fungsi Winsorization
def winsorize(df, cols, limits):
    for col in cols:
        q1, q3 = df[col].dropna().quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - limits * iqr
        upper_bound = q3 + limits * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

# Load dataset
st.title("Analisis Data Curah Hujan dengan K-Means Clustering")
uploaded_file = st.file_uploader("Upload file Excel:", type=["xlsx"])

if uploaded_file:
    df = load_data(uploaded_file)

    # Tampilkan data awal
    st.subheader("Data Awal")
    st.dataframe(df.head())

    # Informasi dataset
    st.subheader("Informasi Dataset")
    st.write(df.info())

    # Statistik deskriptif
    st.subheader("Statistik Deskriptif")
    st.write(df.describe())

    # Visualisasi missing value
    st.subheader("Visualisasi Missing Value")
    missing = df.isnull().mean() * 100
    fig, ax = plt.subplots()
    missing.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title("Missing Value (%)")
    st.pyplot(fig)

    # Winsorize data
    st.subheader("Winsorization")
    num_cols = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_winsorized = winsorize(df.copy(), num_cols, 1.5)

    # Visualisasi sebelum dan sesudah Winsorization
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxplot(data=df[num_cols], ax=axs[0])
    axs[0].set_title("Sebelum Winsorization")
    sns.boxplot(data=df_winsorized[num_cols], ax=axs[1])
    axs[1].set_title("Sesudah Winsorization")
    st.pyplot(fig)

    # Clustering dengan K-means
    st.subheader("K-Means Clustering")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_winsorized[num_cols])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)

    # Visualisasi kluster
    st.write("Hasil Clustering")
    fig = sns.pairplot(data=df, vars=num_cols[:3], hue='cluster', palette='tab10')
    st.pyplot(fig)

    # Evaluasi K-means
    dbi = davies_bouldin_score(df_scaled, df['cluster'])
    sil_score = silhouette_score(df_scaled, df['cluster'])
    st.write(f"Davies-Bouldin Index: {dbi:.4f}")
    st.write(f"Silhouette Score: {sil_score:.4f}")

    # Statistik tiap kluster
    st.subheader("Statistik Tiap Kluster")
    st.write(
        df.groupby('cluster')[num_cols]
        .agg(['mean', 'std', 'min', 'median', 'max'])
        .transpose()
    )

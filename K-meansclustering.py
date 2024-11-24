import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Judul Aplikasi
st.title('Aplikasi Visualisasi Clustering K-Means')

# Upload file
uploaded_file = st.file_uploader("Upload file dataset", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Membaca file yang diupload
    df = pd.read_excel(uploaded_file)
    st.write("Dataframe yang diupload:")
    st.write(df.head())
    
    # EDA: Info Dataset
    if st.button('Tampilkan Info Dataset'):
        st.write(df.info())
    
    # Missing values
    if st.button('Tampilkan Missing Values'):
        missing_data = df.isnull().sum()
        st.write(missing_data)

    # Visualisasi Missing Value
    if st.button('Visualisasi Missing Values'):
        column_with_nan = df.columns[df.isnull().any()]
        column_name = []
        percent_nan = []

        for i in column_with_nan:
            column_name.append(i)
            percent_nan.append(round(df[i].isnull().sum() * 100 / len(df), 2))

        tab = pd.DataFrame(column_name, columns=["Column"])
        tab["Percent_NaN"] = percent_nan
        tab.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)

        sns.set(rc={"figure.figsize": (8, 4)})
        sns.set_style("whitegrid")
        p = sns.barplot(x="Percent_NaN", y="Column", data=tab, edgecolor="black", color="deepskyblue")
        p.set_title("Persentase Missing Value per Kolom", fontsize=20)
        p.set_xlabel("Persentase Missing Value", fontsize=15)
        st.pyplot()

    # Preprocessing Data (Scaling)
    if st.button('Tampilkan Preprocessing Data'):
        df_clean = df.copy()

        # Pilih hanya kolom numerik untuk scaling
        numerical_features = df_clean.select_dtypes(include=['float64', 'int64']).columns

        # Lakukan scaling hanya pada kolom numerik
        scaler = StandardScaler()
        df_clean[numerical_features] = scaler.fit_transform(df_clean[numerical_features])

        # Tampilkan data yang sudah di-preprocess
        st.write(df_clean.head())

    # Model K-means Clustering
    if st.button('Tampilkan Hasil Clustering K-Means'):
        df_clean = df.copy()

        # Pilih hanya kolom numerik untuk clustering
        numerical_features = df_clean.select_dtypes(include=['float64', 'int64']).columns

        kmeans = KMeans(n_clusters=3, random_state=42)
        df_clean['cluster'] = kmeans.fit_predict(df_clean[numerical_features])
        st.write(df_clean.head())

        # Evaluasi Model
        db_score = davies_bouldin_score(df_clean[numerical_features], df_clean['cluster'])
        sil_score = silhouette_score(df_clean[numerical_features], df_clean['cluster'])
        st.write(f"Davies-Bouldin Score: {db_score:.5f}")
        st.write(f"Silhouette Score: {sil_score:.5f}")

        # Visualisasi Cluster
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='Tavg', y='RH_avg', hue='cluster', data=df_clean, palette="Set1", ax=ax)
        ax.set_title("Clustering K-Means (Tavg vs RH_avg)")
        st.pyplot(fig)

    # Menampilkan hasil visualisasi lainnya (misalnya Heatmap)
    if st.button('Tampilkan Heatmap Korelasi'):
        df_num = df.select_dtypes(exclude=["object"])
        corr = df_num.corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        st.pyplot()

    # Menampilkan Peta Heatmap untuk Visualisasi Geospasial
    if st.button('Tampilkan Peta Heatmap Geospasial'):
        import geopandas as gpd
        import folium
        from folium.plugins import HeatMap

        # Pastikan dataset memiliki kolom koordinat (latitude, longitude)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # Membuat objek peta menggunakan folium
            m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)

            # Menggunakan HeatMap dari folium untuk visualisasi
            heat_data = [[row['latitude'], row['longitude']] for index, row in df.iterrows()]
            HeatMap(heat_data).add_to(m)

            # Menampilkan peta di Streamlit
            st.write("Peta Heatmap Geospasial:")
            st.write(m._repr_html_(), unsafe_allow_html=True)
        else:
            st.write("Dataset tidak memiliki kolom latitude dan longitude untuk peta heatmap geospasial.")

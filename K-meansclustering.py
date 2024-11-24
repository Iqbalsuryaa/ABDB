import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import streamlit as st
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# Sidebar Input File
st.sidebar.title("Pengaturan Input")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data sesuai dengan jenis file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    # Tampilkan Data Awal
    st.write("### Data Awal")
    st.dataframe(df.head())

    # Deskripsi Fitur
    st.write("### Deskripsi Fitur Dataset")
    st.write("""
    1. Tn: Temperatur Minimum (°C)  
    2. Tx: Temperatur Maksimum (°C)  
    3. Tavg: Temperatur Rata-Rata (°C)  
    4. RH_avg: Kelembaban Rata-Rata (%)  
    5. RR: Curah Hujan (mm)  
    6. ss: Lamanya Penyinaran Matahari (Jam)  
    7. ff_x: Kecepatan Angin Maksimum (m/s)  
    8. ddd_x: Arah Angin Saat Kecepatan Maksimum  
    9. ff_avg: Kecepatan Angin Rata-Rata (m/s)  
    10. ddd_car: Arah Angin Terbanyak
    """)

    # Missing Values
    st.write("### Persentase Missing Value")
    missing_values = df.isnull().mean() * 100
    if not missing_values.empty:
        st.bar_chart(missing_values)

    # Imputasi Missing Values
    st.write("### Penanganan Missing Values")
    fitur = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg']
    imputer = SimpleImputer(strategy='median')
    df[fitur] = imputer.fit_transform(df[fitur])
    st.write("Missing values telah diimputasi menggunakan nilai median.")

    # Correlation Matrix
    st.write("### Correlation Matrix")
    corr_matrix = df[fitur].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

    # Clustering
    st.write("### Klastering KMeans")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[fitur])

    # Jumlah Klaster Optimal
    st.sidebar.write("**Elbow Method**")
    max_clusters = st.sidebar.slider("Jumlah Maksimal Klaster", min_value=1, max_value=10, value=5)
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-', color='blue')
    plt.xlabel("Jumlah Klaster")
    plt.ylabel("WCSS")
    plt.title("Metode Elbow")
    st.pyplot()

    # Input Jumlah Klaster
    optimal_clusters = st.sidebar.number_input("Jumlah Klaster Optimal", min_value=2, max_value=max_clusters, value=3)
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    st.write("Hasil Klastering:")
    st.dataframe(df)

    # Heatmap
    if "Latitude" in df.columns and "Longitude" in df.columns:
        st.write("### Visualisasi Heatmap")
        locations = df[["Latitude", "Longitude"]].dropna().values.tolist()
        map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
        folium_map = folium.Map(location=map_center, zoom_start=10)
        HeatMap(locations).add_to(folium_map)
        st_folium(folium_map, width=700, height=500)
    else:
        st.write("Data tidak memiliki kolom Latitude dan Longitude untuk membuat peta heatmap.")

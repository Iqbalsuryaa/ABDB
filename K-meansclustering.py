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

# Upload file CSV or Excel
uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data sesuai dengan jenis file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    
    st.write("Data awal:")
    st.dataframe(df.head())

    # Feature Description
    st.write("""
    1. Tn: Temperatur Minimum (Derajat Celcius)
    2. Tx: Temperatur Maximum (Derajat Celcius)
    3. Tavg: Temperatur Rata-Rata (Derajat Celcius)
    4. RH_avg: Kelembaban Rata-Rata (%)
    5. RR: Curah Hujan (mm)
    6. ss: Lamanya Penyinaran Matahari (Jam)
    7. ff_x: Kecepatan Angin Maksimum (m/s)
    8. ddd_x: Arah Angin Saat Kecepatan Maksimum
    9. ff_avg: Kecepatan Angin Rata-Rata (m/s)
    10. ddd_car: Arah Angin Terbanyak
    """)

    # Eksplorasi Data Awal
    st.write(f"Dimensi Data: {df.shape}")
    st.write(f"Missing Values:\n{df.isnull().sum()}")

    # Visualisasi Missing Values
    missing_values = df.isnull().mean() * 100
    missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
    if not missing_values.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_values.values, y=missing_values.index, color="dodgerblue")
        plt.xlabel("Persentase Missing Value (%)")
        plt.title("Persentase Missing Value per Kolom")
        st.pyplot()

    # Menangani Missing Values dengan Imputasi Median
    fitur = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg']
    imputer = SimpleImputer(strategy='median')
    df[fitur] = imputer.fit_transform(df[fitur])

    # Outlier Handling dengan Winsorization
    def winsorize(df, columns, limits=1.5):
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_limit = q1 - limits * iqr
            upper_limit = q3 + limits * iqr
            df[col] = np.clip(df[col], lower_limit, upper_limit)
        return df

    numeric_columns = fitur
    df = winsorize(df, numeric_columns)

    # Correlation Matrix
    numeric_data = df[numeric_columns]
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    st.pyplot()

    # Clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)

    # Menentukan jumlah klaster optimal menggunakan Elbow Method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='blue')
    plt.xlabel("Jumlah Klaster")
    plt.ylabel("WCSS")
    plt.title("Metode Elbow")
    st.pyplot()

    # Menggunakan jumlah klaster optimal
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.write("Hasil Klastering:")
    st.dataframe(df)

    # Visualisasi dengan Heatmap
    if "Latitude" in df.columns and "Longitude" in df.columns:
        locations = df[["Latitude", "Longitude"]].dropna().values.tolist()
        map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
        folium_map = folium.Map(location=map_center, zoom_start=10)
        HeatMap(locations).add_to(folium_map)
        map_file = '/tmp/heatmap.html'
        folium_map.save(map_file)
        st.markdown(f'<a href="file://{map_file}" target="_blank">Klik di sini untuk melihat peta</a>', unsafe_allow_html=True)
    else:
        st.write("Data tidak memiliki kolom Latitude dan Longitude untuk membuat peta heatmap.")

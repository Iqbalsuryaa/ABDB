import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

    # Exploratory Data Analysis (EDA)
    df.info()
    st.write(f"Dimensi Data: {df.shape}")
    st.write(f"Missing Values:\n{df.isnull().sum()}")

    # Handle outliers using IQR
    def iqr_outliers(df):
        out = []
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        Lower_tail = q1 - 1.5 * iqr
        Upper_tail = q3 + 1.5 * iqr
        for column in numeric_df.columns:
            for i in numeric_df[column]:
                if i > Upper_tail[column] or i < Lower_tail[column]:
                    out.append(i)
        print("Outliers:", out)

    def Box_plots(df):
        plt.figure(figsize=(8, 4))
        plt.title("Box Plot")
        sns.boxplot(data=df)
        plt.show()

    iqr_outliers(df)
    Box_plots(df)

    # Check for duplicate values
    st.write(f"Jumlah data duplikat: {df.duplicated().sum()}")

    # Descriptive statistics of the dataset
    st.write(f"Deskripsi Data:\n{df.describe()}")

    # Checking unique values
    st.write(f"Unique values in the dataset:\n{df.nunique()}")

    # Filter data for certain regions
    kota_jatim = ["SIDOARJO", "TUBAN", "PASURUAN", "MALANG"]
    df_kota = df[df['KOTA'].isin(kota_jatim)]
    st.write("Data untuk Kota Jatim:")
    st.dataframe(df_kota)

    # Clustering with K-Means
    df_kota_scaled = df_kota.copy()
    scaler = StandardScaler()
    fitur = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_kota_scaled[fitur] = scaler.fit_transform(df_kota_scaled[fitur])

    # Drop unnecessary columns for clustering
    columns_to_drop = ['Tanggal', 'Latitude', 'Longitude', 'KOTA', 'ddd_car']
    available_columns_to_drop = [col for col in columns_to_drop if col in df_kota_scaled.columns]
    cleaned_kota = df_kota_scaled.drop(columns=available_columns_to_drop, errors='ignore')

    # KMeans Elbow Method to determine the optimal number of clusters
    range_n_clusters = list(range(1, 11))
    wcss = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(cleaned_kota)
        wcss.append(kmeans.inertia_)

    # Elbow method plot
    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, wcss, marker='o', linestyle='-', color='b')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Within-cluster Sum of Squares (WCSS)')
    st.pyplot()

    # Train KMeans with the optimal number of clusters
    optimal_clusters = 3  # Based on Elbow Method
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_kota_scaled['Cluster'] = kmeans.fit_predict(cleaned_kota)

    # Display clustered data
    st.write("Data dengan Hasil Klastering:")
    st.dataframe(df_kota_scaled.head())

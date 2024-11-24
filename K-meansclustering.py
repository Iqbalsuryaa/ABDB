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

# Fungsi untuk menangani outlier
def winsorize(df, columns, limits=1.5):
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - limits * iqr
        upper_limit = q3 + limits * iqr
        df[col] = np.clip(df[col], lower_limit, upper_limit)
    return df

# Fungsi untuk membuat heatmap
def create_heatmap(data, features):
    latitude_center = data['Latitude'].mean()
    longitude_center = data['Longitude'].mean()

    map_heatmap = folium.Map(location=[latitude_center, longitude_center], zoom_start=5)

    for feature_name in features:
        if feature_name in data.columns:
            # Hapus baris dengan nilai NaN
            feature_data = data[['Latitude', 'Longitude', feature_name']].dropna()
            heatmap_data = feature_data.values.tolist()
            feature_group = folium.FeatureGroup(name=feature_name, show=(feature_name == features[0]))
            heatmap_layer = HeatMap(heatmap_data, radius=20)
            feature_group.add_child(heatmap_layer)
            map_heatmap.add_child(feature_group)

    folium.LayerControl().add_to(map_heatmap)
    return map_heatmap

# Streamlit interface
st.title("Eksplorasi dan Visualisasi Data Cuaca dengan Heatmap dan Klasterisasi")

# Upload file CSV atau Excel
uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data sesuai dengan jenis file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    st.write("Data awal:")
    st.dataframe(df.head())

    # Deskripsi fitur
    st.write("""
    **Deskripsi Fitur:**
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

    # Visualisasi dengan Heatmap Interaktif
    if "Latitude" in df.columns and "Longitude" in df.columns:
        features = ['Tavg', 'RH_avg', 'RR', 'ss']
        heatmap = create_heatmap(df, features)

        # Simpan peta sebagai file HTML
        map_file = '/tmp/heatmap.html'
        heatmap.save(map_file)

        # Tampilkan peta di Streamlit
        st.markdown(f'<iframe src="file://{map_file}" width="100%" height="500"></iframe>', unsafe_allow_html=True)
    else:
        st.write("Data tidak memiliki kolom Latitude dan Longitude untuk membuat peta heatmap.")

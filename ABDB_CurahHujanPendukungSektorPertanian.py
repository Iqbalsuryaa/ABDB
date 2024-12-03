import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import folium
from folium import plugins
from streamlit_folium import st_folium

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    return pd.read_csv('Hasilcluster_result.csv')

# Fungsi untuk menampilkan metode elbow
def elbow_method(data):
    wcss = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), wcss, marker='o', color='b')
    plt.title('Metode Elbow K-Means')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('WCSS')
    st.pyplot(plt)

# Fungsi untuk menampilkan heatmap
def create_heatmap(data):
    map_heatmap = folium.Map(
        location=[data['Latitude'].mean(), data['Longitude'].mean()],
        zoom_start=6
    )
    cluster_colors = {0: "red", 1: "blue", 2: "green"}  # Warna RGB untuk setiap cluster
    for _, row in data.iterrows():
        cluster = row['cluster']
        popup_text = f"""
        <b>Cluster:</b> {cluster}<br>
        <b>KOTA:</b> {row['KOTA']}<br>
        <b>Curah Hujan:</b> {row['RR']} mm<br>
        """
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=5,
            color=cluster_colors[cluster],
            fill=True,
            fill_color=cluster_colors[cluster],
            fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(map_heatmap)
    plugins.HeatMap(data[['Latitude', 'Longitude', 'RR']].dropna().values.tolist(), radius=15).add_to(map_heatmap)
    folium.LayerControl().add_to(map_heatmap)
    return map_heatmap

# Fungsi untuk klasifikasi dengan Naive Bayes
def naive_bayes_classification(data):
    st.subheader("Klasifikasi dengan Naive Bayes")

    # Pemilihan fitur dan label
    features = data[['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ff_avg']]
    label = data['cluster']

    # Split data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

    # Inisialisasi model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Akurasi Model: {acc * 100:.2f}%")

    st.write("\n**Classification Report:**")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    st.write("\n**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    st.pyplot(plt)

# Sidebar Menu
st.sidebar.title("Pengaturan")
menu = st.sidebar.radio(
    "Pilih Menu:",
    (
        "Home",
        "Prediksi Dengan Metode ARIMA",
        "Klasifikasi Citra Dengan Metode CNN",
        "Klasifikasi Dengan Decision Trees",
        "Klasifikasi Dengan Naive Bayes",
        "Clustering Dengan Metode K-Means",
    )
)

# Menentukan menu yang dipilih
if menu == "Home":
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")
elif menu == "Prediksi Dengan Metode ARIMA":
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini akan berisi implementasi prediksi curah hujan dengan ARIMA.")
elif menu == "Klasifikasi Citra Dengan Metode CNN":
    st.title("Klasifikasi Citra Awan Curah Hujan dengan Metode CNN")
    st.write("Halaman ini akan berisi implementasi klasifikasi citra awan dengan CNN.")
elif menu == "Klasifikasi Dengan Decision Trees":
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini akan berisi implementasi klasifikasi cuaca dengan Decision Trees.")
elif menu == "Klasifikasi Dengan Naive Bayes":
    df = load_data()
    naive_bayes_classification(df)
elif menu == "Clustering Dengan Metode K-Means":
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Halaman ini akan berisi implementasi clustering data curah hujan dengan K-Means.")

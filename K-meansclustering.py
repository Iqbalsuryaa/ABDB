import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import folium
from folium import plugins

# Membaca data
def load_data():
    data = pd.read_csv('Preprocessed_Dataset.csv')  # Ganti dengan lokasi file Anda
    return data

df = load_data()

# Preprocessing: Encoding dan seleksi kolom
def preprocess_data(df):
    cleaned_kota = df.drop(columns=['Tanggal', 'Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car'])
    encoder = LabelEncoder()
    cleaned_kota['KOTA'] = encoder.fit_transform(cleaned_kota['KOTA'])
    return cleaned_kota

cleaned_kota = preprocess_data(df)

# Menentukan jumlah cluster terbaik dengan metode elbow
st.title("Clustering Curah Hujan dengan K-Means")

range_n_clusters = list(range(1, 11))
wcss = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(cleaned_kota)
    wcss.append(kmeans.inertia_)

# Menampilkan grafik Elbow Method
st.subheader("Metode Elbow")
fig, ax = plt.subplots()
ax.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
ax.set_title('Metode Elbow K-Means')
ax.set_xlabel('Jumlah Cluster')
ax.set_ylabel('WCSS')
st.pyplot(fig)

# Melakukan clustering dengan jumlah cluster yang dipilih (misalnya 3)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(cleaned_kota)

# Menampilkan hasil cluster
st.subheader("Hasil K-Means Clustering")
st.write(df.head())

# Menampilkan statistik deskriptif per cluster
def q25(x):
    return x.quantile(0.25)

def q75(x):
    return x.quantile(0.75)

col_drop = ['Tanggal', 'ddd_car', 'Latitude','Longitude', 'KOTA']
cluster_result = df.drop(col_drop, axis=1)

st.subheader("Descriptive Statistics per Cluster")
st.write(
    cluster_result.groupby('cluster')
    .aggregate(['mean', 'std', 'min', q25, 'median', q75, 'max'])
    .transpose()
)

# Menampilkan distribusi cluster per kota
st.subheader("Distribusi Cluster per Kota")
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='KOTA', hue='cluster', data=df, palette='viridis')
plt.title('Distribusi Cluster per Kota')
plt.xticks(rotation=90)
st.pyplot()

# Peta Heatmap
st.subheader("Peta Heatmap")
latitude_center = df['Latitude'].mean()
longitude_center = df['Longitude'].mean()

map_heatmap = folium.Map(location=[latitude_center, longitude_center], zoom_start=5)

features = {
    'Tavg': df[['Latitude', 'Longitude', 'Tavg']],
    'RH_avg': df[['Latitude', 'Longitude', 'RH_avg']],
    'RR': df[['Latitude', 'Longitude', 'RR']],
    'ss': df[['Latitude', 'Longitude', 'ss']]
}

for feature_name, feature_data in features.items():
    feature_data = feature_data.dropna(subset=['Latitude', 'Longitude', feature_name])
    heatmap_data = feature_data.values.tolist()
    feature_group = folium.FeatureGroup(name=feature_name, show=(feature_name == 'Tavg'))
    heatmap_layer = plugins.HeatMap(heatmap_data, radius=20)
    feature_group.add_child(heatmap_layer)
    map_heatmap.add_child(feature_group)

folium.LayerControl().add_to(map_heatmap)
st.write(map_heatmap)


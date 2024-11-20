# Import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
# Ganti 'dataset.csv' dengan path dataset Anda
data = pd.read_csv('dataset.csv')

# Menampilkan deskripsi fitur data
print("Deskripsi Fitur Dataset:")
deskripsi_fitur = {
    "Tn": "Temperatur Minimum (°C)",
    "Tx": "Temperatur Maksimum (°C)",
    "Tavg": "Temperatur Rata-rata (°C)",
    "RH_avg": "Kelembaban Rata-rata (%)",
    "RR": "Curah Hujan (mm)",
    "ss": "Lamanya Penyinaran Matahari (jam)",
    "ff_x": "Kecepatan Angin Maksimum (m/s)",
    "ddd_x": "Arah Angin Saat Kecepatan Maksimum",
    "ff_avg": "Kecepatan Angin Rata-rata (m/s)",
    "ddd_car": "Arah Angin Terbanyak"
}
for fitur, deskripsi in deskripsi_fitur.items():
    print(f"{fitur}: {deskripsi}")

# Menampilkan 5 data pertama
print("\nDataset:")
print(data.head())

# Heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()

# Standarisasi data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

# Menentukan jumlah cluster dengan metode Elbow
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Inertia')
plt.title('Metode Elbow untuk Menentukan Cluster Optimal')
plt.show()

# Clustering menggunakan K-Means (jumlah cluster = 3 sebagai contoh)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Menampilkan hasil cluster
print("\nHasil Cluster:")
print(data.groupby('Cluster').mean())

# Visualisasi cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=data['Tavg'], y=data['RH_avg'], hue=data['Cluster'], palette='viridis'
)
plt.title('Visualisasi Cluster')
plt.xlabel('Temperatur Rata-rata (°C)')
plt.ylabel('Kelembaban Rata-rata (%)')
plt.legend(title='Cluster')
plt.show()

# Menampilkan hasil per cluster
for cluster in sorted(data['Cluster'].unique()):
    print(f"\nCluster {cluster}:")
    print(data[data['Cluster'] == cluster])

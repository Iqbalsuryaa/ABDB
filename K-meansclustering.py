import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Mengimpor dataset
# Gantilah dengan path atau variabel yang sesuai
data = pd.read_csv("path_to_your_dataset.csv")

# Menampilkan informasi tentang dataset dan deskripsi fitur
print("Informasi Data:")
print(data.info())
print("\nDeskripsi Fitur:")
print(data.describe())

# Melakukan pra-pemrosesan (misalnya, menstandarisasi data)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Melakukan K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Evaluasi K-Means: Silhouette Score
sil_score = silhouette_score(data_scaled, data['Cluster'])
print(f"\nSilhouette Score: {sil_score:.4f}")

# Menampilkan Box Plot untuk setiap fitur berdasarkan cluster
plt.figure(figsize=(12, 8))
sns.boxplot(x='Cluster', y='feature_column_name', data=data)  # Gantilah 'feature_column_name' dengan nama kolom yang sesuai
plt.title('Box Plot per Cluster')
plt.show()

# Menampilkan hasil cluster 0, 1, 2 (analisis berdasarkan cluster)
for i in range(3):
    print(f"\nCluster {i}:\n")
    print(data[data['Cluster'] == i].describe())

# Menggunakan PCA untuk visualisasi 2D
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
data['PCA1'] = principal_components[:, 0]
data['PCA2'] = principal_components[:, 1]

# Plot hasil K-Means Clustering pada PCA 2D
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='Set1')
plt.title('Hasil K-Means Clustering (PCA 2D)')
plt.show()

# Membuat Heatmap dari korelasi antar fitur
correlation_matrix = data.drop('Cluster', axis=1).corr()  # Menghapus kolom 'Cluster' sebelum menghitung korelasi
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Korelasi Antar Fitur')
plt.show()


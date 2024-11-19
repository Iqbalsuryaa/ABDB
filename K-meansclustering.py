import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
st.title("Clustering Curah Hujan dengan K-Means")
uploaded_file = st.file_uploader("Upload file dataset (.csv atau .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    st.write("Dataset:")
    st.dataframe(data)

    # Pilih fitur untuk clustering
    features = st.multiselect("Pilih fitur untuk clustering:", options=data.columns)
    
    if len(features) > 0:
        # Scaling data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[features])

        # Tentukan jumlah cluster
        n_clusters = st.slider("Jumlah Cluster (k):", min_value=2, max_value=10, value=3)

        # Model K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)

        # Tambahkan hasil cluster ke dataset
        data["Cluster"] = labels
        st.write("Dataset dengan Cluster:")
        st.dataframe(data)

        # Visualisasi cluster
        if len(features) == 2:  # Jika hanya dua fitur
            fig, ax = plt.subplots()
            scatter = ax.scatter(data[features[0]], data[features[1]], c=labels, cmap='viridis')
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        else:
            st.write("Visualisasi hanya didukung untuk dua fitur.")

        # Menampilkan centroids
        st.write("Centroids:")
        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Judul Aplikasi
st.title("Clustering Curah Hujan dengan Metode K-Means")

# Upload File CSV
uploaded_file = st.file_uploader("Upload file CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    data = pd.read_csv(uploaded_file)

    st.write("Data yang diunggah:")
    st.dataframe(data.head())

    # Pilih kolom untuk clustering
    st.sidebar.header("Pengaturan Clustering")
    selected_columns = st.sidebar.multiselect(
        "Pilih kolom untuk clustering (minimal 2 kolom):",
        data.columns
    )

    if len(selected_columns) >= 2:
        clustering_data = data[selected_columns]

        # Normalisasi data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)

        # Pilih jumlah kluster
        num_clusters = st.sidebar.slider("Pilih jumlah kluster:", 2, 10, 3)

        # K-Means Clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = clusters

        st.write("Hasil Clustering:")
        st.dataframe(data)

        # Visualisasi
        if len(selected_columns) == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(
                clustering_data[selected_columns[0]],
                clustering_data[selected_columns[1]],
                c=clusters,
                cmap='viridis'
            )
            ax.set_xlabel(selected_columns[0])
            ax.set_ylabel(selected_columns[1])
            plt.colorbar(scatter)
            st.pyplot(fig)
        else:
            st.write("Visualisasi hanya tersedia untuk 2 kolom.")

        # Menampilkan centroid
        st.write("Centroid Kluster:")
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns))
    else:
        st.warning("Pilih minimal 2 kolom untuk clustering.")
else:
    st.info("Unggah file CSV untuk memulai.")

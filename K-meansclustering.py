import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import streamlit as st
import numpy as np

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

    # Visualize missing values percentage
    column_with_nan = df.columns[df.isnull().any()]
    column_name = []
    percent_nan = []

    for i in column_with_nan:
        column_name.append(i)
        percent_nan.append(round(df[i].isnull().sum() * 100 / len(df), 2))

    tab = pd.DataFrame(column_name, columns=["Column"])
    tab["Percent_NaN"] = percent_nan
    tab.sort_values(by=["Percent_NaN"], ascending=False, inplace=True)

    sns.set(rc={"figure.figsize": (8, 4)})
    sns.set_style("whitegrid")
    p = sns.barplot(
        x="Percent_NaN",
        y="Column",
        data=tab,
        edgecolor="black",
        color="deepskyblue"
    )
    p.set_title("Persentase Missing Value per Kolom\n", fontsize=20)
    p.set_xlabel("\nPersentase Missing Value", fontsize=20)
    st.pyplot()

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

    # Plot bar chart for data per Kabupaten
    df_kota['KOTA'].value_counts().plot(kind='bar')
    plt.xlabel('Kabupaten')
    plt.ylabel('Jumlah')
    plt.title('Jumlah Data per Kabupaten')
    st.pyplot()

    # Sample one record per city
    def sampel_per_kota(group):
        return group.sample(n=1)

    df_sampled = df_kota.groupby('KOTA').apply(sampel_per_kota).reset_index(drop=True)
    st.write("Sample Data per Kota:")
    st.dataframe(df_sampled)

    # Plot distribution of wind direction (ddd_car)
    df_kota['ddd_car'].value_counts().plot(kind='bar')
    plt.xlabel('Arah Mata Angin')
    plt.ylabel('Jumlah')
    plt.title('Jumlah Data per Arah Mata Angin')
    st.pyplot()

    # Correlation matrix
    df_num = df.select_dtypes(exclude=["object"])
    corr = df_num.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    st.pyplot()

    # Winsorization for handling outliers
    def winsorize(df_clean, cols, limits):
        for col in cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            q1, q3 = df_clean[col].dropna().quantile([0.25, 0.75])
            iqr_val = q3 - q1
            lower_bound = q1 - limits * iqr_val
            upper_bound = q3 + limits * iqr_val
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        return df_clean

    num_col = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_clean2 = winsorize(df_kota, num_col, 1.5)

    # Boxplot before and after Winsorization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df[num_col])
    plt.title('Sebelum Winsorization')
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_clean2[num_col])
    plt.title('Setelah Winsorization')
    plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot()

    # Clustering with K-Means
    df_kota_scaled = df_kota.copy()
    scaler = StandardScaler()
    fitur = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_kota_scaled[fitur] = scaler.fit_transform(df_kota_scaled[fitur])

    # Drop unnecessary columns for clustering
    cleaned_kota = df_kota_scaled.drop(columns=['Tanggal', 'Latitude', 'Longitude', 'KOTA', 'ddd_car'])

    # KMeans Elbow Method to determine the optimal number of clusters
    range_n_clusters = list(range(1, 11))
    wcss = []

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(cleaned_kota)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range_n_clusters, wcss, marker='*', markersize=10, markerfacecolor='red')
    plt.title('Elbow Method of K-Means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    st.pyplot()

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_labels = kmeans.fit_predict(cleaned_kota)
    df_kota_scaled['Cluster'] = kmeans_labels

    # Evaluation
    kmeans_dbi = davies_bouldin_score(cleaned_kota, kmeans_labels)
    kmeans_sil = silhouette_score(cleaned_kota, kmeans_labels)

    st.write(f"K-Means Clustering\nDavies-Bouldin Index: {kmeans_dbi:.5f}")
    st.write(f"Silhouette Score: {kmeans_sil:.5f}")

    # Display cluster data
    df_kmeans_cluster = df_kota_scaled.sort_values(by='Cluster')

    # Show cluster distribution
    cluster_dfs = {}
    for cluster in df_kmeans_cluster['Cluster'].unique():
        cluster_dfs[cluster] = df_kmeans_cluster[df_kmeans_cluster['Cluster'] == cluster]

    # Display cluster summaries
    st.write("Cluster 0:")
    st.dataframe(cluster_dfs[0].describe())

    st.write("Cluster 1:")
    st.dataframe(cluster_dfs[

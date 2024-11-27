import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Fungsi untuk mendeteksi outliers menggunakan metode IQR
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
    return out

# Fungsi untuk Winsorization
def winsorize(df_clean, cols, limits):
    for col in cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        q1, q3 = df_clean[col].dropna().quantile([0.25, 0.75])
        iqr_val = q3 - q1
        lower_bound = q1 - limits * iqr_val
        upper_bound = q3 + limits * iqr_val
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean

# Fungsi geocoding untuk mendapatkan koordinat
def get_coordinates(kab, attempts=5, timeout=10):
    geolocator = Nominatim(user_agent="test_geo")
    for attempt in range(attempts):
        try:
            location = geolocator.geocode(f"{kab}", timeout=timeout)
            if location:
                return location.latitude, location.longitude
            else:
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < attempts - 1:
                time.sleep(1)
                continue
            else:
                return None, None

# Main Streamlit App
st.title("Analisis Curah Hujan dan Data Cuaca")

# Upload file
uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    
    # Tampilkan informasi umum tentang dataset
    st.subheader("Informasi Data")
    st.write(df.info())
    
    # Tampilkan dimensi dataset
    st.write(f"Dimensi Data: {df.shape}")
    
    # Mengecek missing values
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)
    
    # Visualisasi missing values dengan barplot
    st.subheader("Visualisasi Missing Values")
    missing_data = pd.DataFrame({'Column': missing_values.index, 'Percent_NaN': missing_values.values})
    missing_data['Percent_NaN'] = missing_data['Percent_NaN'] * 100 / len(df)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Percent_NaN', y='Column', data=missing_data, edgecolor="black", color="deepskyblue", ax=ax)
    ax.set_title("Persentase Missing Values per Kolom")
    st.pyplot(fig)
    
    # Deteksi outliers dengan IQR
    st.subheader("Deteksi Outliers dengan IQR")
    outliers = iqr_outliers(df)
    st.write(f"Outliers: {outliers}")
    
    # Boxplot sebelum Winsorization
    st.subheader("Boxplot Sebelum Winsorization")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']), ax=ax)
    ax.set_title("Boxplot Sebelum Winsorization")
    st.pyplot(fig)
    
    # Winsorization untuk menangani outliers
    num_col = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss', 'ff_x', 'ddd_x', 'ff_avg', 'ddd_car']
    df_clean2 = winsorize(df.copy(), num_col, 1.5)
    
    # Boxplot setelah Winsorization
    st.subheader("Boxplot Setelah Winsorization")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df_clean2[num_col], ax=ax)
    ax.set_title("Boxplot Setelah Winsorization")
    st.pyplot(fig)
    
    # Geocoding untuk mendapatkan latitude dan longitude
    st.subheader("Geocoding Kota")
    df_clean2['Latitude'] = None
    df_clean2['Longitude'] = None
    
    for index, row in df_clean2.iterrows():
        kab = row['KOTA']
        latitude, longitude = get_coordinates(kab)
        df_clean2.at[index, 'Latitude'] = latitude
        df_clean2.at[index, 'Longitude'] = longitude
    
    # Tampilkan hasil geocoding
    st.write(df_clean2[['KOTA', 'Latitude', 'Longitude']].head())


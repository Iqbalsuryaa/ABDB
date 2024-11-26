import streamlit as st

# Fungsi untuk halaman Insight
def insight():
    st.title("Insight")
    st.write("Halaman ini menampilkan insight dari data yang telah dianalisis.")
    st.write("""
        - **Analisis Tren:** Curah hujan cenderung meningkat di musim penghujan.
        - **Pola Cuaca:** Suhu dan kelembaban memiliki korelasi signifikan terhadap curah hujan.
        - **Rekomendasi Data:** Data curah hujan dan cuaca perlu diupdate secara berkala untuk akurasi lebih baik.
    """)

# Fungsi untuk halaman Decision
def decision():
    st.title("Decision")
    st.write("Halaman ini memberikan keputusan berdasarkan analisis data.")
    st.write("""
        - **Keputusan:** Berdasarkan prediksi curah hujan, disarankan menanam padi pada bulan Desember.
        - **Konteks:** Wilayah dengan kelembaban di atas 80% dan curah hujan tinggi cocok untuk pertanian basah.
    """)

# Fungsi untuk halaman Conclusion
def conclusion():
    st.title("Conclusion")
    st.write("Halaman ini memberikan kesimpulan dari analisis data.")
    st.write("""
        - **Kesimpulan:** Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - **Tindak Lanjut:** Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Fungsi untuk halaman Home
def home():
    # Menampilkan Header/Banner
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Header Banner" width="800">
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Judul dan Deskripsi Home
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")    

    # Menampilkan Abstrak
    st.subheader("Abstrak")
    st.write("""
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
    """)

    # Tombol-Tombol di bagian bawah Home
    st.subheader("Pilih Metode Analisis:")
    choice = st.radio("Pilih metode analisis untuk melanjutkan:",
                      ("Pilih Metode", "Prediksi Curah Hujan dengan ARIMA", 
                       "Klasifikasi Citra Awan Curah Hujan dengan CNN",
                       "Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees", 
                       "Clustering Curah Hujan dengan K-Means"))

    # Menampilkan Gambar Arsitektur Sistem
    st.subheader("Arsitektur Sistem")
    st.markdown(
        """
        <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Gambar Hider" width="700">
        """,
        unsafe_allow_html=True,
    )

    # Penjelasan Arsitektur Sistem
    st.write("""
        Arsitektur sistem ini menggambarkan alur kerja aplikasi dari pengambilan data,
        preprocessing, hingga analisis. Data curah hujan diolah menggunakan beberapa
        metode untuk menghasilkan prediksi yang akurat. Komponen utama meliputi:
        - **Pengumpulan Data:** Data cuaca harian dari BMKG atau citra awan.
        - **Preprocessing:** Normalisasi data, augmentasi gambar, dan transformasi fitur.
        - **Model Analitik:** Penggunaan algoritma ARIMA untuk data waktu, CNN untuk klasifikasi gambar,
          dan clustering dengan K-Means untuk pengelompokan data.
        - **Output:** Prediksi cuaca atau rekomendasi tindakan untuk sektor pertanian.
    """)

    # Konten Berdasarkan Pilihan
    if choice == "Prediksi Curah Hujan dengan ARIMA":
        arima_page()
    elif choice == "Klasifikasi Citra Awan Curah Hujan dengan CNN":
        cnn_page()
    elif choice == "Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees":
        decision_trees_page()
    elif choice == "Clustering Curah Hujan dengan K-Means":
        kmeans_page()

# Halaman ARIMA
def arima_page():
    st.title("Prediksi Curah Hujan dengan ARIMA")
    st.write("Halaman ini berisi implementasi prediksi curah hujan menggunakan model ARIMA.")
    # Contoh peramalan cuaca menggunakan ARIMA
    st.write("""
        Di halaman ini, kita akan menampilkan hasil peramalan curah hujan dengan model ARIMA.
        (Implementasi kode ARIMA dapat ditambahkan di sini)
    """)

# Halaman CNN
def cnn_page():
    st.title("Klasifikasi Citra Awan Curah Hujan dengan CNN")
    st.write("Halaman ini berisi implementasi klasifikasi citra awan dengan CNN.")
    # Implementasi klasifikasi CNN bisa ditambahkan di sini.

# Halaman Decision Trees
def decision_trees_page():
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini berisi implementasi klasifikasi cuaca dengan Decision Trees.")
    # Implementasi Decision Trees bisa ditambahkan di sini.

# Halaman K-Means
def kmeans_page():
    st.title("Clustering Curah Hujan dengan K-Means")
    st.write("Halaman ini berisi implementasi clustering data curah hujan dengan K-Means.")
    # Implementasi K-Means bisa ditambahkan di sini.

# Sidebar Menu
st.sidebar.title("Main Menu")
menu = st.sidebar.radio(
    "Pilih Menu:",
    (
        "Home",
        "Insight",
        "Decision",
        "Conclusion",
    )
)

# Menentukan menu yang dipilih
if menu == "Home":
    home()
elif menu == "Insight":
    insight()
elif menu == "Decision":
    decision()
elif menu == "Conclusion":
    conclusion()

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

# Fungsi untuk halaman Conclusion
def arima_page():
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini menampilkan implementasi dari Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("""
        - **Kesimpulan:** Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - **Tindak Lanjut:** Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Fungsi untuk halaman Conclusion
def cnn_page():
    st.title("Klasifikasi Citra Awan untuk prediksi Curah Hujan dengan CNN")
    st.write("Halaman ini implementasi dari Klasifikasi Citra Awan untuk prediksi Curah Hujan dengan CNN")
    st.write("""
        - **Kesimpulan:** Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - **Tindak Lanjut:** Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Fungsi untuk halaman Conclusion
def decision_trees_page():
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini implementasi dari Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("""
        - **Kesimpulan:** Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - **Tindak Lanjut:** Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

# Fungsi untuk halaman Conclusion
def kmeans_page():
    st.title("Clustering Curah Hujan dengan menggunakan Metode K-Means")
    st.write("Halaman ini implementasi dari Hasil Clustering Curah Hujan dengan K-Means")
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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Prediksi Curah Hujan dengan ARIMA"):
            st.session_state.page = "ARIMA"
    with col2:
        if st.button("Klasifikasi Citra Awan Curah Hujan dengan CNN"):
            st.session_state.page = "CNN"
    with col3:
        if st.button("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees"):
            st.session_state.page = "Decision Trees"
    with col4:
        if st.button("Clustering Curah Hujan dengan K-Means"):
            st.session_state.page = "K-Means"

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

# Menangani halaman berdasarkan tombol yang dipilih di Home
if "page" in st.session_state:
    page = st.session_state.page
    if page == "ARIMA":
        arima_page()
    elif page == "CNN":
        cnn_page()
    elif page == "Decision Trees":
        decision_trees_page()
    elif page == "K-Means":
        kmeans_page()
else:
    st.session_state.page = "Home"

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

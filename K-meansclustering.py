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

# Halaman ARIMA
def arima_page():
    st.title("Prediksi Curah Hujan dengan ARIMA")
    st.write("Halaman ini berisi implementasi prediksi curah hujan menggunakan model ARIMA.")
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

# Menentukan halaman berdasarkan menu atau tombol
if "page" not in st.session_state:
    st.session_state.page = "Home"

if menu == "Home":
    st.session_state.page = "Home"
elif menu == "Insight":
    st.session_state.page = "Insight"
elif menu == "Decision":
    st.session_state.page = "Decision"
elif menu == "Conclusion":
    st.session_state.page = "Conclusion"

# Logika untuk memanggil halaman berdasarkan `st.session_state.page`
if st.session_state.page == "Home":
    home()
elif st.session_state.page == "Insight":
    insight()
elif st.session_state.page == "Decision":
    decision()
elif st.session_state.page == "Conclusion":
    conclusion()
elif st.session_state.page == "ARIMA":
    arima_page()
elif st.session_state.page == "CNN":
    cnn_page()
elif st.session_state.page == "Decision Trees":
    decision_trees_page()
elif st.session_state.page == "K-Means":
    kmeans_page()

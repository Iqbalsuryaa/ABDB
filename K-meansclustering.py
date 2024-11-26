import streamlit as st

# Inisialisasi session_state untuk menyimpan halaman saat ini
if "page" not in st.session_state:
    st.session_state.page = "Home"

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

# Fungsi untuk halaman ARIMA
def arima_page():
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini menampilkan implementasi dari Prediksi Curah Hujan dengan Metode ARIMA.")
    st.write("""
        - Model ARIMA memberikan prediksi berbasis data time series.
        - Membutuhkan parameter seperti differencing order, p, dan q.
    """)

# Fungsi untuk halaman CNN
def cnn_page():
    st.title("Klasifikasi Citra Awan untuk Prediksi Curah Hujan dengan CNN")
    st.write("Halaman ini menampilkan implementasi klasifikasi citra awan dengan CNN.")
    st.write("""
        - Model CNN digunakan untuk analisis berbasis gambar.
        - Melibatkan preprocessing dan augmentasi gambar untuk hasil lebih akurat.
    """)

# Fungsi untuk halaman Decision Trees
def decision_trees_page():
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini menampilkan hasil klasifikasi curah hujan dengan Decision Trees.")
    st.write("""
        - Decision Trees cocok untuk data diskrit.
        - Membantu mengelompokkan kondisi cuaca berdasarkan kriteria tertentu.
    """)

# Fungsi untuk halaman K-Means
def kmeans_page():
    st.title("Clustering Curah Hujan dengan K-Means")
    st.write("Halaman ini menampilkan hasil clustering curah hujan dengan K-Means.")
    st.write("""
        - K-Means digunakan untuk mengelompokkan data curah hujan.
        - Memerlukan inisialisasi jumlah cluster (k).
    """)

# Fungsi untuk halaman Home
def home():
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")    

    st.subheader("Pilih Metode Analisis:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Prediksi Curah Hujan dengan ARIMA"):
            st.session_state.page = "ARIMA"
    with col2:
        if st.button("Klasifikasi Citra Awan dengan CNN"):
            st.session_state.page = "CNN"
    with col3:
        if st.button("Decision Trees"):
            st.session_state.page = "Decision Trees"
    with col4:
        if st.button("K-Means"):
            st.session_state.page = "K-Means"

# Navigasi halaman berdasarkan session_state
def navigate_page():
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "ARIMA":
        arima_page()
    elif st.session_state.page == "CNN":
        cnn_page()
    elif st.session_state.page == "Decision Trees":
        decision_trees_page()
    elif st.session_state.page == "K-Means":
        kmeans_page()
    elif st.session_state.page == "Insight":
        insight()
    elif st.session_state.page == "Decision":
        decision()
    elif st.session_state.page == "Conclusion":
        conclusion()

# Sidebar untuk navigasi manual
st.sidebar.title("Main Menu")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ("Home", "Insight", "Decision", "Conclusion"),
    index=["Home", "Insight", "Decision", "Conclusion"].index(st.session_state.page),
)
if menu != st.session_state.page:
    st.session_state.page = menu

# Panggil fungsi navigasi
navigate_page()

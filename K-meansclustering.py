import streamlit as st

# Inisialisasi nilai awal untuk session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Fungsi untuk setiap halaman
def home():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://raw.githubusercontent.com/Iqbalsuryaa/ABDB/main/hider.png" alt="Header Banner" width="800">
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("Home")
    st.write("Selamat datang di aplikasi Analisis Curah Hujan Menggunakan Pendekatan Big Data untuk Mendukung Pertanian!")
    
    st.subheader("Abstrak")
    st.write("""
        Aplikasi ini dirancang untuk memprediksi curah hujan berdasarkan data cuaca 
        dan analisis citra awan. Berbagai metode seperti ARIMA, CNN, Decision Trees, 
        dan K-Means digunakan untuk mendukung analisis ini. Tujuannya adalah 
        untuk membantu sektor pertanian dan masyarakat dalam memahami pola cuaca 
        yang lebih baik.
    """)

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

def insight():
    st.title("Insight")
    st.write("Halaman ini menampilkan insight dari data yang telah dianalisis.")
    st.write("""
        - **Analisis Tren:** Curah hujan cenderung meningkat di musim penghujan.
        - **Pola Cuaca:** Suhu dan kelembaban memiliki korelasi signifikan terhadap curah hujan.
        - **Rekomendasi Data:** Data curah hujan dan cuaca perlu diupdate secara berkala untuk akurasi lebih baik.
    """)

def decision():
    st.title("Decision")
    st.write("Halaman ini memberikan keputusan berdasarkan analisis data.")
    st.write("""
        - **Keputusan:** Berdasarkan prediksi curah hujan, disarankan menanam padi pada bulan Desember.
        - **Konteks:** Wilayah dengan kelembaban di atas 80% dan curah hujan tinggi cocok untuk pertanian basah.
    """)

def conclusion():
    st.title("Conclusion")
    st.write("Halaman ini memberikan kesimpulan dari analisis data.")
    st.write("""
        - **Kesimpulan:** Model ARIMA dan CNN mampu memberikan prediksi yang cukup akurat untuk mendukung pengambilan keputusan di sektor pertanian.
        - **Tindak Lanjut:** Integrasi lebih lanjut dengan data real-time diperlukan untuk meningkatkan keandalan sistem.
    """)

def arima_page():
    st.title("Prediksi Curah Hujan dengan Metode ARIMA")
    st.write("Halaman ini menampilkan implementasi dari Prediksi Curah Hujan dengan Metode ARIMA.")

def cnn_page():
    st.title("Klasifikasi Citra Awan untuk Prediksi Curah Hujan dengan CNN")
    st.write("Halaman ini menampilkan implementasi klasifikasi citra awan untuk prediksi curah hujan menggunakan CNN.")

def decision_trees_page():
    st.title("Klasifikasi Cuaca Curah Hujan menggunakan Decision Trees")
    st.write("Halaman ini menampilkan implementasi klasifikasi cuaca curah hujan menggunakan Decision Trees.")

def kmeans_page():
    st.title("Clustering Curah Hujan dengan Metode K-Means")
    st.write("Halaman ini menampilkan hasil clustering curah hujan menggunakan metode K-Means.")

# Pemetaan halaman ke fungsi
pages = {
    "Home": home,
    "Insight": insight,
    "Decision": decision,
    "Conclusion": conclusion,
    "ARIMA": arima_page,
    "CNN": cnn_page,
    "Decision Trees": decision_trees_page,
    "K-Means": kmeans_page,
}

# Sidebar navigasi
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Home", "Insight", "Decision", "Conclusion"],
    index=["Home", "Insight", "Decision", "Conclusion"].index(
        st.session_state.page if st.session_state.page in ["Home", "Insight", "Decision", "Conclusion"] else "Home"
    )
)

# Sinkronisasi pilihan sidebar dengan halaman
if menu != st.session_state.page:
    st.session_state.page = menu

# Render halaman berdasarkan pilihan
pages[st.session_state.page]()

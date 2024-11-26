import streamlit as st

# Inisialisasi nilai awal untuk session state
if "page" not in st.session_state:
    st.session_state.page = "Home"
    st.session_state.metode_analisis = "ARIMA"  # Set default method

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
    
    # Menggunakan radio button untuk memilih metode analisis
    analisis_metode = st.radio(
        "Pilih metode analisis yang diinginkan:",
        ("ARIMA", "CNN", "Decision Trees", "K-Means"),
        index=["ARIMA", "CNN", "Decision Trees", "K-Means"].index(st.session_state.metode_analisis)
    )
    
    # Menyimpan pilihan metode analisis ke dalam session_state
    if analisis_metode != st.session_state.metode_analisis:
        st.session_state.metode_analisis = analisis_metode

    st.subheader("Insight, Decision, and Conclusion:")
    # Tampilkan insight, decision dan conclusion berdasarkan metode yang dipilih
    insight_decision_conclusion()


def insight_decision_conclusion():
    # Menampilkan insight, keputusan, dan kesimpulan
    st.write("Halaman ini menampilkan insight, keputusan, dan kesimpulan dari analisis yang dilakukan.")
    
    if st.session_state.metode_analisis == "ARIMA":
        st.write("### Insight: Prediksi Curah Hujan dengan ARIMA")
        st.write("- **Analisis Tren:** Curah hujan cenderung meningkat di musim penghujan.")
        st.write("- **Pola Cuaca:** Suhu dan kelembaban memiliki korelasi signifikan terhadap curah hujan.")
        st.write("- **Rekomendasi:** Data curah hujan perlu diupdate secara berkala.")
        st.write("### Decision:")
        st.write("- **Keputusan:** Menanam padi pada bulan Desember disarankan berdasarkan prediksi ARIMA.")
        st.write("### Conclusion:")
        st.write("- **Kesimpulan:** ARIMA memberikan prediksi yang cukup akurat untuk mendukung pertanian.")
    
    elif st.session_state.metode_analisis == "CNN":
        st.write("### Insight: Klasifikasi Citra Awan dengan CNN")
        st.write("- **Analisis Tren:** Analisis citra awan membantu mengidentifikasi pola curah hujan.")
        st.write("- **Pola Cuaca:** Citra awan memiliki korelasi dengan curah hujan.")
        st.write("- **Rekomendasi:** Citra awan perlu dianalisis dengan lebih detail untuk meningkatkan akurasi.")
        st.write("### Decision:")
        st.write("- **Keputusan:** Berdasarkan citra awan, wilayah tertentu dapat diprioritaskan untuk pertanian.")
        st.write("### Conclusion:")
        st.write("- **Kesimpulan:** CNN dapat digunakan untuk prediksi cuaca yang lebih akurat berdasarkan citra awan.")
    
    elif st.session_state.metode_analisis == "Decision Trees":
        st.write("### Insight: Klasifikasi Cuaca dengan Decision Trees")
        st.write("- **Analisis Tren:** Decision trees memberikan gambaran yang jelas tentang faktor yang mempengaruhi curah hujan.")
        st.write("- **Pola Cuaca:** Analisis kecepatan angin dan kelembaban dapat diprediksi dengan baik.")
        st.write("- **Rekomendasi:** Decision trees efektif untuk klasifikasi cuaca berulang.")
        st.write("### Decision:")
        st.write("- **Keputusan:** Wilayah dengan pola cuaca tertentu disarankan untuk bercocok tanam.")
        st.write("### Conclusion:")
        st.write("- **Kesimpulan:** Decision Trees memberikan hasil yang kuat untuk prediksi pola cuaca.")
    
    elif st.session_state.metode_analisis == "K-Means":
        st.write("### Insight: Clustering Curah Hujan dengan K-Means")
        st.write("- **Analisis Tren:** K-Means digunakan untuk mengelompokkan wilayah berdasarkan curah hujan.")
        st.write("- **Pola Cuaca:** Clustering dapat memberikan wawasan tentang wilayah dengan pola cuaca serupa.")
        st.write("- **Rekomendasi:** Clustering ini membantu dalam merencanakan pertanian berdasarkan pola curah hujan.")
        st.write("### Decision:")
        st.write("- **Keputusan:** Wilayah dengan curah hujan tinggi dapat diprioritaskan untuk pertanian basah.")
        st.write("### Conclusion:")
        st.write("- **Kesimpulan:** K-Means memberikan cluster yang jelas tentang curah hujan dan pola cuaca.")


# Pemetaan halaman ke fungsi
pages = {
    "Home": home,
}

# Sidebar navigasi
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Home"],
    index=["Home"].index(st.session_state.page)
)

# Sinkronisasi pilihan sidebar dengan halaman
if menu != st.session_state.page:
    st.session_state.page = menu

# Render halaman berdasarkan pilihan
pages[st.session_state.page]()

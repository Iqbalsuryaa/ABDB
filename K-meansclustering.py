import streamlit as st

# Fungsi untuk setiap menu
def home():
    st.title("Home")
    st.write("Selamat datang di aplikasi Streamlit!")
    st.write("Aplikasi ini memiliki tiga menu utama: **Home**, **Tasks**, dan **Settings**.")
    st.image(
        "https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png",
        caption="Streamlit Logo",
        use_column_width=True,
    )

def tasks():
    st.title("Tasks")
    st.write("Berikut adalah daftar tugas Anda:")
    
    tasks = ["Tugas 1: Belajar Streamlit", "Tugas 2: Membuat aplikasi", "Tugas 3: Menyelesaikan proyek"]
    for i, task in enumerate(tasks, 1):
        st.checkbox(f"{i}. {task}")
    
    st.text_area("Catatan Tambahan", "Tulis catatan untuk tugas-tugas Anda di sini...")

def settings():
    st.title("Settings")
    st.write("Atur preferensi Anda:")
    
    username = st.text_input("Nama Pengguna", "Guest")
    theme = st.selectbox("Pilih Tema", ["Light", "Dark", "System Default"])
    notifications = st.checkbox("Aktifkan Notifikasi", value=True)
    
    st.write("**Ringkasan Pengaturan:**")
    st.write(f"- Nama Pengguna: {username}")
    st.write(f"- Tema: {theme}")
    st.write(f"- Notifikasi Aktif: {'Ya' if notifications else 'Tidak'}")

# Membuat Sidebar
st.sidebar.title("Main Menu")
menu = st.sidebar.radio("Pilih Menu:", ("Home", "Tasks", "Settings"))

# Logika navigasi menu
if menu == "Home":
    home()
elif menu == "Tasks":
    tasks()
elif menu == "Settings":
    settings()

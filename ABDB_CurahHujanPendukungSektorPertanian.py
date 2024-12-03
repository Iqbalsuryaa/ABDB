elif menu == "Klasifikasi Dengan Naive Bayes":
    st.title("Klasifikasi Cuaca Menggunakan Naive Bayes")
    st.write("Halaman ini akan berisi implementasi klasifikasi cuaca dengan Naive Bayes.")
    
    # Contoh Implementasi Naive Bayes
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Load Data
    df = load_data()

    # Pemrosesan data sederhana untuk Naive Bayes
    st.subheader("Pemrosesan Data")
    selected_columns = ['Tavg', 'RH_avg', 'RR']  # Pilih kolom fitur
    label_column = 'cluster'  # Kolom target
    df = df.dropna(subset=selected_columns + [label_column])  # Hapus baris kosong
    X = df[selected_columns]
    y = df[label_column]

    # Split data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    st.write("Data berhasil dipisah menjadi 70% data latih dan 30% data uji.")

    # Naive Bayes
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi Model
    st.subheader("Hasil Evaluasi")
    st.write("Akurasi Model:", accuracy_score(y_test, y_pred))
    st.write("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred))
    st.write("Matriks Kebingungan:")
    st.write(confusion_matrix(y_test, y_pred))

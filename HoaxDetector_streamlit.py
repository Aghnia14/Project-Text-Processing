import streamlit as st
import joblib

model_path = 'svm_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    st.error(f"Gagal memuat model atau vectorizer: {e}")
    st.stop()

# Streamlit
st.title('Sistem Deteksi Hoaks tentang Pemilu 2024 di Twitter/X')
st.markdown("""
    Sistem ini dirancang untuk mendeteksi apakah sebuah teks tentang Pemilu 2024 di Twitter/X mengandung hoaks. 
    Cukup masukkan teks di bawah, dan sistem akan menentukan apakah teks tersebut merupakan hoaks atau bukan.
""")


user_input = st.text_area('Masukkan teks untuk diperiksa:')

if st.button('Prediksi'):
    if user_input:
        try:
            # Vektorisasi input teks
            user_input_vectorized = vectorizer.transform([user_input])
            
            # Prediksi dengan model
            prediction = model.predict(user_input_vectorized)
            
            # Tampilkan hasil
            result = 'Hoaks' if prediction[0] == 1 else 'NonHoaks'
            st.success(f'Teks ini diklasifikasikan sebagai: {result}')
        except Exception as e:
            st.error(f"Gagal melakukan prediksi: {e}")
    else:
        st.warning('Silakan masukkan teks untuk diklasifikasikan.')

st.markdown("---")
st.markdown("© 2024 - Text Processing | Group 4 Data Science 2023C")
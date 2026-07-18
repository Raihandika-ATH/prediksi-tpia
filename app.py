import streamlit as st
import joblib
import numpy as np

model = joblib.load('/content/drive/MyDrive/Penambangan Data_Raihandika/model_tpia.joblib')

st.set_page_config(page_title="Prediksi Saham TPIA", layout="centered")

st.title("📈 Prediksi Harga Penutupan TPIA")
st.write("Aplikasi untuk memprediksi harga saham PT Chandra Asri Pacific Tbk berdasarkan pergerakan harga hari ini.")

st.sidebar.header("Input Data Harian TPIA")
open_price = st.sidebar.number_input("Harga Pembukaan (Open)", min_value=0.0, value=7500.0, step=50.0)
high_price = st.sidebar.number_input("Harga Tertinggi (High)", min_value=0.0, value=7700.0, step=50.0)
low_price = st.sidebar.number_input("Harga Terendah (Low)", min_value=0.0, value=7400.0, step=50.0)
volume = st.sidebar.number_input("Volume Transaksi", min_value=0, value=15000000)

if st.sidebar.button("Prediksi Harga Close"):
    input_data = np.array([[open_price, high_price, low_price, volume]])
    pred = model.predict(input_data)

    st.success("Prediksi Berhasil Dilakukan!")
    st.metric(label="Estimasi Harga Penutupan (Close) TPIA", value=f"Rp {pred[0]:,.2f}")
    st.info("Catatan: Prediksi ini menggunakan model regresi machine learning berbasis data historis. Bukan murni rekomendasi finansial (NFA).")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Judul
st.title("Prediksi Penyakit Jantung")
st.write("Model klasifikasi menggunakan Support Vector Classifier")

# Input fitur
st.header("Masukkan Data Pasien")

age = st.number_input("Usia", min_value=1, max_value=120, value=40)
sex = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")
cp = st.selectbox("Tipe Nyeri Dada (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Tekanan darah saat istirahat (trestbps)", value=120)
chol = st.number_input("Kolesterol (chol)", value=200)
fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Hasil EKG istirahat (restecg)", [0, 1, 2])
thalach = st.number_input("Detak jantung maksimal (thalach)", value=150)
exang = st.selectbox("Angina karena olahraga (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (depresi ST)", value=1.0, step=0.1)
slope = st.selectbox("Kemiringan ST (slope)", [0, 1, 2])
ca = st.selectbox("Jumlah pembuluh darah utama (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
                            oldpeak, slope, ca, thal]])
    
    # Scaling
    scaler = MinMaxScaler()
    df = pd.read_csv("heart.csv")
    X = df.drop(columns=["target"])
    scaler.fit(X)
    input_scaled = scaler.transform(input_data)

    # Model
    model = SVC(kernel='rbf', C=2)
    Y = df["target"]
    X_scaled = scaler.transform(X)
    model.fit(X_scaled, Y)

    # Prediksi
    hasil = model.predict(input_scaled)
    hasil_text = "Pasien **berisiko** terkena penyakit jantung." if hasil[0] == 1 else "Pasien **tidak berisiko** terkena penyakit jantung."

    st.subheader("Hasil Prediksi:")
    st.success(hasil_text)

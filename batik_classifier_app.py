import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# Load model hanya sekali
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

class_names = [
    "JawaBarat_Megamendung",
    "Kalimantan_CorakInsang",
    "Kalimantan_Dayak",
    "Papua_Cendrawasih",
    "Solo_Parang",
    "Tiongkok_IkatCelup",
    "Yogyakarta_Kawung"
]

# UI Streamlit
st.set_page_config(page_title="Klasifikasi Batik", layout="centered")

st.title("Klasifikasi Batik Nusantara")
st.write("Unggah gambar batik untuk memprediksi jenisnya.")

uploaded_file = st.file_uploader("📤 Unggah Gambar Batik", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diperkecil
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Preview Gambar", width=300)

    # Preprocessing untuk model
    img_array = image.img_to_array(img)
    img_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_expanded)

    # Prediksi
    with st.spinner("Mengklasifikasikan..."):
        predictions = model.predict(img_preprocessed)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = predictions[predicted_idx] * 100

    # Tampilkan hasil
    st.success(f"✅ Hasil: **{predicted_class}** ({confidence:.2f}%)")

    # Tampilkan bar chart dengan label persentase
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(class_names, predictions * 100, color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probabilitas (%)")
    ax.invert_yaxis()

    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                f"{predictions[i]*100:.2f}%", va='center')

    st.pyplot(fig)

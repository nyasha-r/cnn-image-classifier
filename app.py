import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("cnn_model.keras")

class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

# ------------------ HEADER ------------------
st.title("🧠 CIFAR-10 Image Classifier")
st.markdown("Upload an image and the AI model will predict its class.")
st.divider()

# ------------------ FILE UPLOADER ------------------
uploaded_file = st.file_uploader(
    "📂 Upload an image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file)

    # Show original image
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image_resized = image.resize((32, 32))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    with st.spinner("🔍 Analyzing image..."):
        predictions = model.predict(image_array)[0]

    # Get top 3 predictions
    top_indices = predictions.argsort()[-3:][::-1]

    with col2:
        st.subheader("🏆 Top Predictions")
        for i in top_indices:
            st.write(f"**{class_names[i]}** — {predictions[i]*100:.2f}%")

    st.divider()

    # ------------------ PROBABILITY CHART ------------------
    st.subheader("📊 Prediction Probabilities")

    fig, ax = plt.subplots()
    ax.bar(class_names, predictions)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.success("✅ Prediction Complete")

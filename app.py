import os
print(os.getcwd())

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("cnn_model.keras")


class_names = [
    'airplane','automobile','bird','cat','deer',
    'dog','frog','horse','ship','truck'
]

st.title("CIFAR-10 Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Predicted Class: {predicted_class}")

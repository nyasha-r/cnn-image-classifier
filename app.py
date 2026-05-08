import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Vision System",
    layout="wide"
)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>

/* DARK GREY BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #1f2937, #111827);
    color: #e5e7eb;
}

/* GLASS EFFECT */
.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0px 8px 32px rgba(0,0,0,0.6);
}

/* MAIN TITLE */
.main-title {
    text-align: center;
    font-size: 60px;
    font-weight: 700;
    margin-top: 10px;
    background: linear-gradient(90deg,#22c55e,#3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* SUBTITLE */
.sub-title {
    text-align: center;
    color: #9ca3af;
    font-size: 18px;
    margin-bottom: 30px;
}

/* IMAGE */
.img-box img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 240px;
    border-radius: 12px;
}

/* METRIC */
.metric {
    font-size: 30px;
    font-weight: 600;
}

/* PROGRESS BAR */
.bar {
    height: 8px;
    border-radius: 10px;
    background: #374151;
}

.fill {
    height: 8px;
    border-radius: 10px;
    background: linear-gradient(90deg,#22c55e,#3b82f6);
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">AI Vision System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Deep Learning Image Classification with CNN</div>',
    unsafe_allow_html=True
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            "cnn_model.h5",
            compile=False
        )
        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# ---------------- CLASS LABELS ----------------
class_names = [
    'Airplane',
    'Automobile',
    'Bird',
    'Cat',
    'Deer',
    'Dog',
    'Frog',
    'Horse',
    'Ship',
    'Truck'
]

# ---------------- STOP IF MODEL FAILS ----------------
if model is None:
    st.stop()

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.6])

    # ---------------- IMAGE ----------------
    with col1:
        st.markdown(
            '<div class="glass img-box">',
            unsafe_allow_html=True
        )

        st.image(image, width=240)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ---------------- PREDICTION ----------------
    with col2:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        img = image.resize((32, 32))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)

        probs = preds[0]

        pred_class = class_names[np.argmax(probs)]

        confidence = np.max(probs) * 100

        st.markdown(
            f"<div class='metric'>Prediction: {pred_class}</div>",
            unsafe_allow_html=True
        )

        st.write(f"Confidence: {confidence:.2f}%")

        st.markdown("#### Confidence Breakdown")

        for i in np.argsort(probs)[::-1][:5]:

            value = probs[i] * 100

            st.write(class_names[i])

            st.markdown(f"""
            <div class="bar">
                <div class="fill" style="width:{value}%"></div>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"{value:.2f}%")

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ---------------- TABS ----------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Analysis",
        "🔍 Interpretation",
        "ℹ️ Model"
    ])

    # ---------------- ANALYSIS ----------------
    with tab1:

        df = pd.DataFrame({
            "Class": class_names,
            "Confidence": probs
        }).sort_values(
            by="Confidence",
            ascending=False
        )

        st.bar_chart(df.set_index("Class"))

    # ---------------- INTERPRETATION ----------------
    with tab2:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        st.write("""
        The CNN extracts hierarchical spatial features from the image.

        High-confidence predictions indicate strong alignment with learned patterns.

        Future improvement:
        Grad-CAM visualization to highlight important regions.
        """)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ---------------- MODEL INFO ----------------
    with tab3:

        st.markdown(
            '<div class="glass">',
            unsafe_allow_html=True
        )

        st.write("""
        **Architecture:** Convolutional Neural Network

        **Dataset:** CIFAR-10

        **Input Size:** 32x32 RGB

        **Framework:** TensorFlow / Keras
        """)

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "🚀 Advanced AI System | Designed for Production & Research Showcase"
)
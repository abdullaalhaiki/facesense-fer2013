import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="FaceSense",
    page_icon="🎭",
    layout="wide"
)

CLASSES = [
    "angry", "disgust", "fear", "happy",
    "neutral", "sad", "surprise"
]

@st.cache_resource
def load_resources():
    model_path = Path(__file__).resolve().parent / "fer2013_cnn.h5"
    model = tf.keras.models.load_model(model_path, compile=False)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return model, detector

model, detector = load_resources()

st.title("🎭 FaceSense")
st.subheader("Facial Expression Recognition with Deep Learning")
st.write("Upload a face image or capture one using your webcam.")

st.sidebar.header("About FaceSense")
st.sidebar.write("Dataset: FER2013")
st.sidebar.write("Model: Convolutional Neural Network")
st.sidebar.metric("Test Accuracy", "48.61%")
st.sidebar.write("Classes: 7 facial expressions")

mode = st.radio(
    "Select input method",
    ["Upload Image", "Use Webcam"],
    horizontal=True
)

if mode == "Upload Image":
    uploaded = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded = st.camera_input("Capture your face")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    rgb = np.array(image)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(30, 30)
    )

    detected = len(faces) > 0

    if detected:
        x, y, w, h = max(
            faces, key=lambda f: f[2] * f[3]
        )
        pad = int(0.1 * max(w, h))
        face = gray[
            max(0, y-pad):min(gray.shape[0], y+h+pad),
            max(0, x-pad):min(gray.shape[1], x+w+pad)
        ]
    else:
        face = gray

    face = cv2.resize(face, (48, 48))
    input_data = face.astype("float32").reshape(1, 48, 48, 1)

    probabilities = model.predict(input_data, verbose=0)[0]
    best = int(np.argmax(probabilities))

    left, right = st.columns(2)

    with left:
        st.image(image, caption="Original Image", use_container_width=True)
        st.image(face, caption="Processed Face (48×48)", width=150)

    with right:
        st.success(f"Predicted Expression: {CLASSES[best].capitalize()}")
        st.metric("Model Score", f"{probabilities[best]*100:.1f}%")

        results = pd.DataFrame({
            "Expression": [c.capitalize() for c in CLASSES],
            "Probability": probabilities
        }).set_index("Expression")

        st.bar_chart(results)

    if not detected:
        st.warning("No face detected. Prediction used the entire image.")

    st.caption(
        "This model classifies visible facial expressions, "
        "not a person's actual emotional state. "
        "Prediction scores are not guaranteed to be calibrated probabilities."
    )

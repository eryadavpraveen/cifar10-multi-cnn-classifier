import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CIFAR-10 Multi-Model Classifier",
    page_icon="🧠",
    layout="wide"
)

CLASS_NAMES = [
    'Airplane ✈️', 'Automobile 🚗', 'Bird 🐦', 'Cat 🐱', 'Deer 🦌',
    'Dog 🐶', 'Frog 🐸', 'Horse 🐴', 'Ship 🚢', 'Truck 🚚'
]

MODEL_DIR = "save_models"

MODEL_DISPLAY_NAMES = {
    "alex_net.h5": "AlexNet",
    "lenet.h5": "LeNet-5",
    "vgg.h5": "VGG-16",
    "resnet.h5": "ResNet"
}

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==========================================
# 🎨 MAIN APP
# ==========================================

def main():
    st.title("🧠 CIFAR-10 Multi-Model Image Classifier")
    st.markdown("Select a model and upload an image to classify it.")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("⚙️ Model Selection")

    if not os.path.exists(MODEL_DIR):
        st.sidebar.error("❌ No saved_models directory found. Train models first.")
        return

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

    if not model_files:
        st.sidebar.warning("⚠️ No trained models found.")
        return

    selected_model_file = st.sidebar.selectbox(
        "Choose Model",
        model_files,
        format_func=lambda x: MODEL_DISPLAY_NAMES.get(x, x)
    )

    model_path = os.path.join(MODEL_DIR, selected_model_file)
    model = load_model(model_path)

    if model:
        st.sidebar.success(f"✅ Loaded: {MODEL_DISPLAY_NAMES.get(selected_model_file)}")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

            if st.button("🔍 Classify Image", type="primary"):

                if model is None:
                    st.error("Model not loaded properly.")
                else:
                    with st.spinner("Analyzing image..."):

                        processed_img = preprocess_image(image)
                        predictions = model.predict(processed_img)
                        scores = predictions[0]

                        top_index = np.argmax(scores)
                        top_class = CLASS_NAMES[top_index]
                        confidence = scores[top_index] * 100

                        top_3 = np.argsort(scores)[-3:][::-1]

                        with col2:
                            st.subheader("2️⃣ Results")

                            st.success(f"### 🎯 Prediction: {top_class}")
                            st.info(f"Confidence: {confidence:.2f}%")

                            st.markdown("### 🔝 Top 3 Predictions")
                            for i in top_3:
                                st.write(
                                    f"{CLASS_NAMES[i]} → {scores[i]*100:.2f}%"
                                )

                            st.markdown("### 📊 All Class Probabilities")

                            df = pd.DataFrame({
                                "Class": CLASS_NAMES,
                                "Confidence": scores
                            })

                            st.bar_chart(df.set_index("Class"))


if __name__ == "__main__":
    main()
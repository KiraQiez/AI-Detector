# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import PIL.Image as Image

from math_detector import (
    compute_texture_grid,
    create_math_grid_overlay,
    get_math_label,
)

MODEL_PATH = "models/ai_detector_model.h5"
IMG_SIZE = (224, 224)


# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# -----------------------
# Preprocess image
# -----------------------
def preprocess_image(file, target_size=IMG_SIZE):
    img = Image.open(file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0), img  


# -----------------------
# STREAMLIT UI
# -----------------------
st.set_page_config(
    page_title="AI Image Detection (CNN + Math)",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI-Generated Image Detection")
st.caption("Upload once • Get two views: CNN prediction and math-based texture analysis")

left_col, right_col = st.columns([1, 1], gap="large")

# ===== LEFT: upload + image only =====
with left_col:
    st.subheader("Image Upload")

    uploaded_file = st.file_uploader(
        "Upload an image (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
    )

    img_array = None
    img_resized = None

    if uploaded_file is not None:
        img_array, img_resized = preprocess_image(uploaded_file, target_size=IMG_SIZE)
        st.image(
            img_resized,
            caption="Input image (224×224)",
            use_container_width=True,
        )
    else:
        st.info("Upload an image to begin.")


# ===== RIGHT: CNN result + math result + grid =====
with right_col:
    st.subheader("Result")

    if uploaded_file is not None and img_array is not None and img_resized is not None:
        # ---------- 1. CNN MODEL RESULT ----------
        st.markdown("### 🧠 CNN (Model) Prediction")

        preds = model.predict(img_array)
        ai_prob = float(preds[0][0]) 
        real_prob = 1.0 - ai_prob
        cnn_label = "AI-Generated" if ai_prob >= 0.5 else "Real"

        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="Label (CNN)", value=cnn_label)
        with c2:
            st.metric(label="AI Probability (CNN)", value=f"{ai_prob*100:.1f}%")

        st.markdown("**AI Confidence**")
        st.progress(int(ai_prob * 100))
        st.caption(
            f"CNN real probability: {real_prob*100:.1f}% • "
            "Threshold: 0.5 (≥ 50% = AI-generated)."
        )

        st.divider()

        # ---------- 2. MATH-BASED RESULT ----------
        st.markdown("### 📐 Math-Based Texture Analysis (16×16 Grid)")

        # Compute 16×16 texture/edge grid
        texture_grid = compute_texture_grid(img_resized, grid_size=16)
        texture_score = float(texture_grid.mean())  # 0–1
        math_label = get_math_label(texture_score)

        m3, m4 = st.columns(2)
        with m3:
            # Show only the short label (before bracket)
            short_label = math_label.split(" (")[0]
            st.metric(label="Label (Math)", value=short_label)
        with m4:
            st.metric(label="Texture Score", value=f"{texture_score:.2f}")

        st.caption(
            "Texture score is computed from edge/gradient strength in each of the "
            "16×16 blocks. Higher score = more local detail and edges."
        )

        # Visualize math heatmap with 16×16 grid overlay
        overlay_img = create_math_grid_overlay(
            img_resized,
            texture_grid,
            overlay_alpha=0.5,
        )

        st.image(
            overlay_img,
            caption=(
                "Math-based 16×16 grid • "
                "Green = smooth • Yellow = medium detail • Red = highly detailed/edgy"
            ),
            use_container_width=True,
        )

    else:
        st.info("CNN result and math-based grid will appear here after you upload an image.")

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
# CUSTOM CSS (reduce spacing to fit screen)
# -----------------------
st.markdown("""
<style>
    /* Remove top padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    /* Reduce spacing between elements */
    .element-container {
        margin-bottom: 0.4rem;
    }
    /* Reduce header size */
    h1, h2, h3, h4 {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

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
    page_title="AI Detector",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 AI Image Detection")
st.caption("Upload once • View CNN prediction & math heatmap side-by-side")

# Keep same main columns
left_col, right_col = st.columns([0.9, 1.1], gap="small")


# ============================
# LEFT PANEL — UPLOAD
# ============================
with left_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Upload (.jpg, .jpeg, .png)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    img_array = None
    img_resized = None

    if uploaded_file:
        img_array, img_resized = preprocess_image(uploaded_file, target_size=IMG_SIZE)

        # Force image to be small enough to avoid scrolling
        st.image(img_resized, caption="Input (224×224)", width=300)
    else:
        st.info("Upload an image to begin.")


# ============================
# RIGHT PANEL — GRID + RESULTS
# ============================
with right_col:
    st.subheader("Analysis Results")

    if uploaded_file and img_array is not None:

        # ---- Compute everything once ----
        preds = model.predict(img_array)

        # IMPORTANT: assume model output is P(real), not P(ai)
        real_prob = float(preds[0][0])
        ai_prob = 1.0 - real_prob

        cnn_label = "Real" if real_prob >= 0.5 else "AI"

        # Math texture grid (std-based from math_detector.py)
        texture_grid = compute_texture_grid(img_resized, 16)
        texture_score = float(texture_grid.mean())
        math_label = get_math_label(texture_score)

        overlay_img = create_math_grid_overlay(
            img_resized,
            texture_grid,
            overlay_alpha=0.5
        )

        # ---- Split right panel: LEFT = grid, RIGHT = CNN + Math results ----
        grid_col, info_col = st.columns([1, 1], gap="small")

        # LEFT SIDE: heatmap grid
        with grid_col:
            st.markdown("#### 📐 Texture Heatmap (16×16 Grid)")
            st.image(
                overlay_img,
                caption="Green = smooth • Yellow = medium detail • Red = high detail",
                use_container_width=True,
            )

        # RIGHT SIDE: CNN + Math stacked
        with info_col:
            # CNN block
            st.markdown("#### 🧠 CNN Prediction")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Label (CNN)", cnn_label)
            with c2:
                st.metric("AI Probability", f"{ai_prob*100:.1f}%")

            st.progress(int(ai_prob * 100))
            st.caption(
                f"Real probability: {real_prob*100:.1f}% • Threshold: 50% (>= 50% = Real)."
            )

            st.markdown("---")

            # Math block
            st.markdown("#### 📏 Math-Based Texture Result")

            c3, c4 = st.columns(2)
            with c3:
                st.metric("Label (Math)", math_label)
            with c4:
                st.metric("Texture Score", f"{texture_score:.2f}")

            st.caption(
                "Texture score is based on local brightness variation (standard deviation) "
                "over a 16×16 grid. Low = smooth (AI), High = textured (Real)."
            )

    else:
        st.info("Results will appear here after uploading an image.")

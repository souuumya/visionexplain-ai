import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="VisionExplain AI", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.stApp{background:#0c0d16;color:#f6f5fa}
[data-testid="stSidebar"]{background:#171827}
h1,h2,h3{color:#f6f5fa}
div[data-testid="stMetric"]{background:#191a2a;border:1px solid #35334a;padding:14px;border-radius:12px}
div.stButton>button{background:#ad68ff;color:#100b19;border:0;border-radius:10px;font-weight:700}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading pretrained MobileNetV2…")
def load_model():
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer("out_relu").output, model.output])
    return model, grad_model

def prepare(image):
    resized = image.convert("RGB").resize((224,224))
    batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(batch), resized

def gradcam(grad_model, batch, class_idx):
    with tf.GradientTape() as tape:
        conv, pred = grad_model(batch, training=False)
        score = pred[:, class_idx]
    grads = tape.gradient(score, conv)
    weights = tf.reduce_mean(grads, axis=(1,2))
    cam = tf.reduce_sum(conv[0] * weights[0], axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
    return tf.image.resize(cam[...,None], (224,224)).numpy().squeeze()

def overlay(image, heatmap):
    original = np.asarray(image.resize((224,224)), dtype=np.float32) / 255
    colored = plt.get_cmap("jet")(heatmap)[...,:3]
    blended = np.clip(0.58*original + 0.42*colored, 0, 1)
    return Image.fromarray((blended*255).astype(np.uint8))

st.title("🔍 VisionExplain AI")
st.subheader("Explain My Image Classification")
st.write("Upload an image to see a prediction, model score, and a Grad-CAM heatmap highlighting image regions associated with the prediction.")
st.divider()

with st.sidebar:
    st.header("About")
    st.write("**Challenge:** 10 — Explain My Image Classification")
    st.write("**Track:** Explainable AI (XAI)")
    st.write("**Classifier:** MobileNetV2 / ImageNet")
    st.write("**Explanation:** Grad-CAM")
    st.info("Use a clear photo of a common object, animal, vehicle, or food.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded is None:
    a,b,c = st.columns(3)
    a.metric("Step 1", "Upload image")
    b.metric("Step 2", "Predict class")
    c.metric("Step 3", "Explain regions")
    st.caption("The first run downloads pretrained model weights and may take a few minutes.")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        with st.spinner("Classifying image and generating Grad-CAM…"):
            model, grad_model = load_model()
            batch, resized = prepare(image)
            pred = model(batch, training=False).numpy()[0]
            top_idx = np.argsort(pred)[::-1][:3]
            decoded = tf.keras.applications.mobilenet_v2.decode_predictions(pred[None,:], top=3)[0]
            cam = gradcam(grad_model, batch, int(top_idx[0]))
            explanation = overlay(resized, cam)

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Prediction")
            st.metric("Predicted class", decoded[0][1].replace("_"," ").title())
            st.metric("Model score", f"{decoded[0][2]*100:.1f}%")
            st.caption("A model score is not a guarantee of correctness.")
            st.markdown("**Top 3 predictions**")
            for _, label, score in decoded:
                st.write(f"{label.replace('_',' ').title()} — {score*100:.1f}%")
        with right:
            st.subheader("Grad-CAM explanation")
            st.image(explanation, caption="Warmer regions indicate stronger association with the selected class score.", use_container_width=True)

        with st.expander("How to interpret this heatmap"):
            st.write("Grad-CAM uses gradients in a convolutional feature map to highlight regions associated with a class score. It is an approximate explanation of model behaviour, not proof of causal reasoning.")
        with st.expander("Failure cases to investigate"):
            st.markdown("- Unfamiliar images may be misclassified.\n- Background, lighting, cropping, and image quality can affect predictions.\n- Heatmaps can be misleading.\n- Compare the same object across different backgrounds and crops.")
    except Exception as e:
        st.error("Could not process this image. Please try another JPG or PNG.")
        st.caption(f"Technical detail: {type(e).__name__}: {e}")

st.divider()
st.caption("VisionExplain AI • Hackathon Challenge 10 • Research/demo prototype. Not for medical or other high-stakes decisions.")

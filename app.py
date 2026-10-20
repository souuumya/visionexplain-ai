import streamlit as st
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V2_Weights
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
    weights = MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    target_layer = model.features[-1]
    categories = weights.meta["categories"]
    return model, target_layer, categories

def prepare(image):
    resized = image.convert("RGB").resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(resized).unsqueeze(0)
    return tensor, resized

def gradcam(model, target_layer, tensor, class_idx):
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h_fwd = target_layer.register_forward_hook(forward_hook)
    h_bwd = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad()
        output = model(tensor)
        score = output[0, class_idx]
        score.backward()

        weights = torch.mean(gradients[0], dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations[0], dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_max = np.max(cam)
        if cam_max > 1e-8:
            cam = cam / cam_max
        else:
            cam = np.zeros_like(cam)
        return cam
    finally:
        h_fwd.remove()
        h_bwd.remove()

def overlay(image, heatmap):
    original = np.asarray(image.resize((224, 224)), dtype=np.float32) / 255.0
    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    blended = np.clip(0.58 * original + 0.42 * colored, 0, 1)
    return Image.fromarray((blended * 255).astype(np.uint8))

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

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    a, b, c = st.columns(3)
    a.metric("Step 1", "Upload image")
    b.metric("Step 2", "Predict class")
    c.metric("Step 3", "Explain regions")
    st.caption("The first run downloads pretrained model weights and may take a few minutes.")
else:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        with st.spinner("Classifying image and generating Grad-CAM…"):
            model, target_layer, categories = load_model()
            batch, resized = prepare(image)
            with torch.no_grad():
                init_out = model(batch)
                init_probs = torch.softmax(init_out, dim=1)[0]
                top_probs, top_indices = torch.topk(init_probs, 3)

            top_class = int(top_indices[0].item())
            cam = gradcam(model, target_layer, batch, top_class)
            explanation = overlay(resized, cam)

            decoded = [
                (None, categories[idx.item()], float(prob.item()))
                for idx, prob in zip(top_indices, top_probs)
            ]

        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Prediction")
            st.metric("Predicted class", decoded[0][1].replace("_", " ").title())
            st.metric("Model score", f"{decoded[0][2]*100:.1f}%")
            st.caption("A model score is not a guarantee of correctness.")
            st.markdown("**Top 3 predictions**")
            for _, label, score in decoded:
                st.write(f"{label.replace('_', ' ').title()} — {score*100:.1f}%")
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

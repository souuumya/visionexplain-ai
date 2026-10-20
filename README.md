# VisionExplain AI — Challenge 10

Streamlit prototype for Explainable AI image classification using PyTorch and Grad-CAM.

## Features
- Upload JPG/PNG
- MobileNetV2 pretrained on ImageNet (PyTorch)
- Top-3 predictions and model score
- Grad-CAM heatmap overlay
- Failure-case and responsible-AI notes

## Deploy to Streamlit Community Cloud
1. Ensure your GitHub repository has `app.py`, `requirements.txt`, and `README.md` at the repository root.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app** (or select your repository and branch); set main file path to `app.py`.
4. Deploy and wait for dependency installation.
5. Open the live URL and test it with several ordinary images.

The first launch downloads MobileNetV2 ImageNet weights. Supported on modern Python versions (Python 3.10 to 3.14+).

## Scope and limitations
This prototype uses a pretrained ImageNet classifier; it does not train a custom Cats-vs-Dogs model. Grad-CAM is an approximate visualization of model behaviour and does not prove causal reasoning. Do not use for high-stakes decisions.

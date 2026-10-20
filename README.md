# VisionExplain AI — Challenge 10

Streamlit prototype for Explainable AI image classification.

## Features
- Upload JPG/PNG
- MobileNetV2 pretrained on ImageNet
- Top-3 predictions and model score
- Grad-CAM heatmap overlay
- Failure-case and responsible-AI notes

## Deploy to Streamlit Community Cloud
1. Create a GitHub repository named `visionexplain-ai`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Visit https://share.streamlit.io/ and sign in with GitHub.
4. Click Create app; select your repository and branch; main file path: `app.py`.
5. Deploy and wait for dependency installation.
6. Open the live URL and test it with several ordinary images.

The first launch downloads MobileNetV2 ImageNet weights. Python 3.10 or 3.11 is recommended.

## Scope and limitations
This prototype uses a pretrained ImageNet classifier; it does not train a custom Cats-vs-Dogs model. Grad-CAM is an approximate visualization of model behaviour and does not prove causal reasoning. Do not use for high-stakes decisions.

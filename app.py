from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.models import DANN


CLASS_LABELS = [
    "Class 0 (No DR)",
    "Class 1 (Mild/Moderate)",
    "Class 2 (Severe/Proliferative)",
]

CLASS_ADVICE = {
    0: "Patient is healthy. See you in a year for your annual checkup.",
    1: "Early signs of disease detected. Schedule a routine appointment with an eye doctor to monitor.",
    2: "Vision-threatening disease detected. Urgent medical intervention required to prevent blindness.",
}


@st.cache_resource
def load_model():
    model_path = Path(os.getenv("MODEL_PATH", "outputs/dann_ddr_to_aptos_3class/best_dann.pt"))
    backbone = os.getenv("BACKBONE", "efficientnet_b2")
    img_size = int(os.getenv("IMG_SIZE", "260"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Set MODEL_PATH env var or place checkpoint at the default location."
        )

    model = DANN(num_classes=3, backbone=backbone, pretrained=False)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, model_path, backbone, img_size


def predict(image: Image.Image, model: DANN, transform) -> tuple[int, list[float]]:
    x = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model.classify(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    pred_idx = int(torch.argmax(torch.tensor(probs)).item())
    return pred_idx, probs


def main() -> None:
    st.set_page_config(page_title="Diabetic Retinopathy Screening", page_icon="🩺", layout="centered")

    st.title("Diabetic Retinopathy Screening (3-Class)")
    st.caption("For project/demo use only. Not a medical diagnosis tool.")

    try:
        model, transform, model_path, backbone, img_size = load_model()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.expander("Model info"):
        st.write(f"Checkpoint: `{model_path}`")
        st.write(f"Backbone: `{backbone}`")
        st.write(f"Image size: `{img_size}`")

    uploaded = st.file_uploader("Upload a retina fundus image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded is None:
        st.info("Upload an image to get a prediction.")
        return

    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict", type="primary"):
        pred_idx, probs = predict(image, model, transform)

        st.subheader("Prediction")
        st.success(f"{CLASS_LABELS[pred_idx]}")
        st.write(CLASS_ADVICE[pred_idx])

        st.subheader("Class probabilities")
        prob_dict = {CLASS_LABELS[i]: float(probs[i]) for i in range(3)}
        st.bar_chart(prob_dict)
        st.json({k: round(v, 4) for k, v in prob_dict.items()})


if __name__ == "__main__":
    main()

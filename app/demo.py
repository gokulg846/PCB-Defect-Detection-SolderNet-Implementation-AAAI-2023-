from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import CLASS_NAMES, get_eval_transforms
from src.gradcam import GradCAM, overlay_heatmap
from src.model import create_soldernet_model, get_module_by_name, load_checkpoint


CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
METRICS_JSON = ROOT / "results" / "metrics.json"


@st.cache_resource
def load_model(checkpoint_mtime: float | None) -> tuple[torch.nn.Module, str, torch.device, bool]:
    del checkpoint_mtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, target_layer_name = create_soldernet_model(
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        freeze_backbone_stages=False,
    )
    has_checkpoint = CHECKPOINT_PATH.exists()
    if has_checkpoint:
        checkpoint = load_checkpoint(model, CHECKPOINT_PATH, device)
        target_layer_name = checkpoint.get("target_layer_name", target_layer_name)
    model.to(device)
    model.eval()
    return model, target_layer_name, device, has_checkpoint


def checkpoint_mtime() -> float | None:
    return CHECKPOINT_PATH.stat().st_mtime if CHECKPOINT_PATH.exists() else None


def predict_image(
    image: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 224,
) -> tuple[int, float, pd.DataFrame, torch.Tensor]:
    transform = get_eval_transforms(image_size)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    predicted_idx = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_idx].item())
    chart_df = pd.DataFrame(
        {"confidence": probabilities.numpy()},
        index=CLASS_NAMES,
    )
    return predicted_idx, confidence, chart_df, input_tensor


def make_gradcam_overlay(
    image: Image.Image,
    input_tensor: torch.Tensor,
    model: torch.nn.Module,
    target_layer_name: str,
    target_class: int,
) -> Image.Image:
    target_layer = get_module_by_name(model, target_layer_name)
    gradcam = GradCAM(model, target_layer)
    try:
        heatmap = gradcam(input_tensor, target_class=target_class)
    finally:
        gradcam.remove_hooks()
    overlay = overlay_heatmap(image, heatmap)
    return Image.fromarray(overlay)


def load_metrics() -> dict:
    if not METRICS_JSON.exists():
        return {}
    return json.loads(METRICS_JSON.read_text())


def render_sidebar(has_checkpoint: bool) -> None:
    st.sidebar.header("Project")
    st.sidebar.write("**Model:** ConvNeXt-Tiny SolderNet-style classifier")
    st.sidebar.write("**Dataset:** DeepPCB")
    st.sidebar.write("**Paper:** Gunraj et al., AAAI 2023")
    st.sidebar.write("**GitHub:** https://github.com/your-org/pcb-defect-detection")
    if has_checkpoint:
        st.sidebar.success("Loaded checkpoints/best_model.pth")
    else:
        st.sidebar.warning(
            "No checkpoint found. The app is using random weights, so predictions are for UI exploration only."
        )


def upload_mode(model: torch.nn.Module, target_layer_name: str, device: torch.device) -> None:
    st.header("Upload Image")
    uploaded = st.file_uploader("Upload a PCB crop or image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a JPG or PNG image to run inference and Grad-CAM.")
        return

    image = Image.open(uploaded).convert("RGB")
    predicted_idx, confidence, chart_df, input_tensor = predict_image(image, model, device)
    predicted_label = CLASS_NAMES[predicted_idx]

    st.subheader(f"Prediction: {predicted_label}")
    st.write(f"Confidence: **{confidence:.2%}**")

    overlay = make_gradcam_overlay(image, input_tensor, model, target_layer_name, predicted_idx)
    left, right = st.columns(2)
    with left:
        st.image(image, caption="Original", use_column_width=True)
    with right:
        st.image(overlay, caption="Grad-CAM overlay", use_column_width=True)

    st.subheader("Class confidence")
    st.bar_chart(chart_df)


def test_sample_mode(model: torch.nn.Module, device: torch.device) -> None:
    st.header("Run on Test Set Sample")
    if not TEST_CSV.exists():
        st.warning("No test split found. Run `python data/download_data.py` first.")
        return

    samples = pd.read_csv(TEST_CSV)
    if samples.empty:
        st.warning("The test split is empty.")
        return

    sample_df = samples.sample(n=min(9, len(samples)))
    transform = get_eval_transforms()
    images: list[Image.Image] = []
    tensors: list[torch.Tensor] = []
    true_labels: list[str] = []

    for _, row in sample_df.iterrows():
        image_path = ROOT / row["filepath"]
        image = Image.open(image_path).convert("RGB")
        images.append(image)
        tensors.append(transform(image))
        true_labels.append(str(row["label"]))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)
        predictions = probabilities.argmax(dim=1).cpu().tolist()

    st.subheader("Random test samples")
    for row_idx in range(0, len(images), 3):
        columns = st.columns(3)
        for col_idx, image in enumerate(images[row_idx : row_idx + 3]):
            idx = row_idx + col_idx
            pred_label = CLASS_NAMES[predictions[idx]]
            true_label = true_labels[idx]
            correct = pred_label == true_label
            color = "green" if correct else "red"
            with columns[col_idx]:
                st.image(image, use_column_width=True)
                st.markdown(
                    f"True: **{true_label}**<br>"
                    f"Pred: <span style='color:{color}; font-weight:700'>{pred_label}</span>",
                    unsafe_allow_html=True,
                )

    metrics = load_metrics()
    st.subheader("Saved test metrics")
    if metrics:
        st.metric("Overall accuracy", f"{metrics.get('overall_accuracy', 0.0):.2%}")
        st.metric("Macro F1", f"{metrics.get('macro_f1', 0.0):.3f}")
        per_class = metrics.get("per_class", {})
        if per_class:
            st.dataframe(pd.DataFrame(per_class).T)
    else:
        st.info("No results/metrics.json found yet. Run `python src/evaluate.py` after training.")


def main() -> None:
    st.set_page_config(page_title="PCB Defect Detection - SolderNet", layout="wide")
    st.title("PCB Defect Detection - SolderNet")
    model, target_layer_name, device, has_checkpoint = load_model(checkpoint_mtime())
    render_sidebar(has_checkpoint)
    mode = st.sidebar.radio("Mode", ["Upload Image", "Run on Test Set Sample"])

    if mode == "Upload Image":
        upload_mode(model, target_layer_name, device)
    else:
        test_sample_mode(model, device)


if __name__ == "__main__":
    main()

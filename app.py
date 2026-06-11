"""
🍜 Nigerian Food Classifier — Streamlit App
Loads a trained EfficientNetV2 model and classifies Nigerian food images.
"""

import io
import copy
import math
import time
import warnings
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go
import timm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nigerian Food Classifier",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "best_fold0.pth"   
IMG_SIZE        = 224
DEVICE          = torch.device("cpu")            
TOP_K           = 5


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION  (mirrors your training notebook exactly)
# ─────────────────────────────────────────────────────────────
class NigerianFoodClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=False,
            num_classes=0, global_pool="avg",
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(p=dropout / 2),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.BatchNorm1d(feat_dim // 2),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────────────────────
def _tfm(img_size: int, flip: bool = False, brightness: float = 0.0,
         crop_frac: float = 1.0) -> A.Compose:
    steps = []
    if crop_frac < 1.0:
        steps += [
            A.Resize(img_size, img_size),
            A.CenterCrop(int(img_size * crop_frac), int(img_size * crop_frac)),
        ]
    steps.append(A.Resize(img_size, img_size))
    if flip:
        steps.append(A.HorizontalFlip(p=1.0))
    if brightness != 0.0:
        steps.append(A.RandomBrightnessContrast(
            brightness_limit=(brightness, brightness), contrast_limit=0, p=1.0))
    steps += [
        A.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ToTensorV2(),
    ]
    return A.Compose(steps)


TTA_TRANSFORMS = [
    _tfm(IMG_SIZE),                             # original
    _tfm(IMG_SIZE, flip=True),                  # H-flip
    _tfm(IMG_SIZE, brightness=0.1),             # brighter
    _tfm(IMG_SIZE, crop_frac=0.9),              # center crop 90 %
]


# ─────────────────────────────────────────────────────────────
# MODEL LOADING  (cached — loads once for all users)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    model = NigerianFoodClassifier(
        model_name  = ckpt["model_name"],
        num_classes = ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt["class_names"]


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(img_pil: Image.Image, model: nn.Module, use_tta: bool = True):
    img_np = np.array(img_pil.convert("RGB"), dtype=np.uint8)

    transforms = TTA_TRANSFORMS if use_tta else [TTA_TRANSFORMS[0]]
    all_probs  = []

    for tfm in transforms:
        x     = tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
        all_probs.append(probs)

    return np.mean(all_probs, axis=0)   # (num_classes,)


# ─────────────────────────────────────────────────────────────
# GRAD-CAM  (optional — only runs if pytorch-grad-cam installed)
# ─────────────────────────────────────────────────────────────
def gradcam_overlay(img_pil: Image.Image, model: nn.Module, class_idx: int):
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        return None

    # Find last conv layer
    last_conv = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m

    if last_conv is None:
        return None

    img_np  = np.array(img_pil.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.uint8)
    img_f32 = img_np.astype(np.float32) / 255.0

    tfm = TTA_TRANSFORMS[0]
    x   = tfm(image=img_np)["image"].unsqueeze(0).to(DEVICE)

    cam      = GradCAMPlusPlus(model=model, target_layers=[last_conv])
    mask     = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0]
    overlay  = show_cam_on_image(img_f32, mask, use_rgb=True)

    return Image.fromarray(overlay)


# ─────────────────────────────────────────────────────────────
# PLOTLY CONFIDENCE CHART
# ─────────────────────────────────────────────────────────────
def confidence_chart(class_names, probs, top_k=5):
    top_idx   = probs.argsort()[-top_k:][::-1]
    names     = [class_names[i] for i in top_idx][::-1]
    values    = [float(probs[i]) for i in top_idx][::-1]
    colors    = ["#27AE60" if i == len(values) - 1 else "#3498DB"
                 for i in range(len(values))]

    fig = go.Figure(go.Bar(
        x           = values,
        y           = names,
        orientation = "h",
        marker_color= colors,
        text        = [f"{v:.1%}" for v in values],
        textposition= "outside",
        hovertemplate = "%{y}: %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        margin      = dict(l=10, r=60, t=10, b=10),
        xaxis       = dict(range=[0, 1.12], tickformat=".0%", title="Confidence"),
        yaxis       = dict(title=""),
        height      = 260,
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font        = dict(size=13),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def sidebar(class_names):
    with st.sidebar:
        st.title("⚙️ Settings")
        use_tta  = st.toggle("Test-Time Augmentation (TTA)", value=True,
                             help="Averages 4 views — more accurate, slightly slower")
        show_cam = st.toggle("Show Grad-CAM", value=False,
                             help="Highlights which part of the image drove the prediction")
        st.divider()

        st.subheader("📋 Model Info")
        st.caption(f"**Architecture:** tf_efficientnetv2_s")
        st.caption(f"**Classes:** {len(class_names)}")
        st.caption(f"**Input size:** {IMG_SIZE}×{IMG_SIZE}")
        st.caption(f"**Val F1 (fold 0):** 0.8086")
        st.divider()

        st.subheader("🏷️ All Classes")
        for i, name in enumerate(sorted(class_names)):
            st.caption(f"{i+1}. {name}")

    return use_tta, show_cam


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    # ── Load model ───────────────────────────────────────────
    if not Path(CHECKPOINT_PATH).exists():
        st.error(
            f"❌ Checkpoint not found at `{CHECKPOINT_PATH}`.\n\n"
            "**Steps to fix:**\n"
            "1. Download `best_fold0.pth` from your Kaggle notebook output.\n"
            "2. Place it in a `checkpoints/` folder next to `app.py`."
        )
        st.stop()

    model, class_names = load_model(CHECKPOINT_PATH)

    # ── Sidebar settings ─────────────────────────────────────
    use_tta, show_cam = sidebar(class_names)

    # ── Header ───────────────────────────────────────────────
    st.title("🍜 Nigerian Food Classifier")
    st.caption(
        "Upload a photo of a Nigerian dish or snack and the model will "
        "identify it with confidence scores."
    )
    st.divider()

    # ── Upload ───────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        # Show example placeholder
        st.info(
            "👆 Upload any Nigerian food photo above — jollof rice, egusi soup, "
            "suya, puff puff, moi moi, and 16 other classes supported."
        )
        return

    # ── Layout: image | results ───────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    img_pil = Image.open(uploaded).convert("RGB")

    with col_img:
        st.subheader("📸 Uploaded Image")
        st.image(img_pil, use_container_width=True)

    with col_res:
        st.subheader("🔍 Prediction")

        with st.spinner("Classifying…"):
            t0    = time.time()
            probs = predict(img_pil, model, use_tta=use_tta)
            elapsed = time.time() - t0

        top_idx   = int(probs.argmax())
        top_class = class_names[top_idx]
        top_conf  = float(probs[top_idx])

        # Top prediction badge
        colour = "#27AE60" if top_conf >= 0.6 else "#F39C12" if top_conf >= 0.35 else "#E74C3C"
        st.markdown(
            f"""
            <div style="background:{colour}22; border-left: 4px solid {colour};
                        border-radius:8px; padding:14px 18px; margin-bottom:12px">
                <span style="font-size:1.6rem; font-weight:700">{top_class}</span><br>
                <span style="font-size:1.1rem; color:{colour}; font-weight:600">
                    {top_conf:.1%} confidence
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Confidence bar chart
        st.caption(f"Top-{TOP_K} predictions  ·  inference {elapsed*1000:.0f} ms")
        st.plotly_chart(
            confidence_chart(class_names, probs, TOP_K),
            use_container_width=True,
        )

    # ── Grad-CAM section ─────────────────────────────────────
    if show_cam:
        st.divider()
        st.subheader("🔬 Grad-CAM — What the model is looking at")

        with st.spinner("Generating attention map…"):
            overlay = gradcam_overlay(img_pil, model, top_idx)

        if overlay is not None:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.image(img_pil.resize((IMG_SIZE, IMG_SIZE)), caption="Original",
                         use_container_width=True)
            with c2:
                st.image(overlay, caption="Grad-CAM++ overlay",
                         use_container_width=True)
            with c3:
                st.markdown(
                    """
                    **Reading the map:**
                    - 🔴 **Red / warm** — high activation, the model focused here
                    - 🔵 **Blue / cool** — ignored region
                    
                    A good model lights up the **dish itself**, not the plate, 
                    background, or hands.
                    """
                )
        else:
            st.warning(
                "`pytorch-grad-cam` not found. "
                "Add it to `requirements.txt` and redeploy."
            )

    # ── Footer ───────────────────────────────────────────────
    st.divider()
    st.caption(
        "Model: EfficientNetV2-S · 5-fold CV · Macro F1 ≈ 0.805 · "
        "Built with PyTorch + timm + Streamlit"
    )


if __name__ == "__main__":
    main()

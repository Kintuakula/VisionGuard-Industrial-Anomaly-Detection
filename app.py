
import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from main import build_model, get_transform, score_batch, DEVICE

st.set_page_config(
    page_title="VisionGuard AI",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #080d1b 0%, #11182c 55%, #101026 100%);
    color: #edf2ff;
}
[data-testid="stHeader"] {
    background: transparent;
}
[data-testid="stSidebar"] {
    background: #0c1222;
    border-right: 1px solid #27324a;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1450px;
}
h1, h2, h3, p, label {
    color: #edf2ff;
}
.hero {
    padding: 28px 30px;
    border: 1px solid #303b62;
    border-radius: 22px;
    background: linear-gradient(115deg, #172448, #25204a 65%, #172c48);
    margin-bottom: 24px;
}
.hero-tag {
    color: #8fe7ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
}
.hero-title {
    font-size: clamp(27px, 4vw, 42px);
    font-weight: 850;
    margin: 10px 0;
    line-height: 1.2;
}
.hero-sub {
    color: #b7c4df;
    font-size: 16px;
}
.panel {
    background: #131c30;
    border: 1px solid #2b3853;
    border-radius: 18px;
    padding: 20px;
    min-height: 110px;
}
.panel-title {
    font-size: 19px;
    font-weight: 750;
    margin-bottom: 6px;
}
.muted {
    color: #9caecc;
    font-size: 13px;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #182641, #171c34);
    border: 1px solid #35476b;
    padding: 18px;
    border-radius: 16px;
}
div[data-testid="stMetricLabel"] {
    color: #a9bce0;
}
div[data-testid="stMetricValue"] {
    color: #f4f7ff;
}
div.stButton > button {
    background: linear-gradient(90deg, #6366f1, #0ea5e9);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}
div[data-testid="stFileUploader"] {
    background: #111a2d;
    border: 1px dashed #53668e;
    padding: 12px;
    border-radius: 14px;
}
hr {
    border-color: #293651;
}
.stCaption {
    color: #9caecc;
}
</style>
""", unsafe_allow_html=True)

PROJECT_DIR = Path(__file__).resolve().parent
MEMORY_PATH = PROJECT_DIR / "outputs" / "bottle" / "memory_bank.pt"


@st.cache_resource
def load_detector():
    if not MEMORY_PATH.exists():
        raise FileNotFoundError(
            f"Memory bank not found: {MEMORY_PATH}"
        )

    net = build_model()
    saved = torch.load(
        MEMORY_PATH,
        map_location="cpu",
        weights_only=False
    )
    memory = F.normalize(saved["memory"].float(), p=2, dim=1)
    return net, memory


with st.sidebar:
    st.markdown("## 🔬 VisionGuard")
    st.caption("INDUSTRIAL AI PLATFORM")
    st.divider()

    st.markdown("### ⚙️ Inspection Settings")
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Information")
    st.markdown("**Backbone:** ResNet18")
    st.markdown("**Approach:** Patch-based anomaly detection")
    st.markdown("**Category:** Bottle")
    st.markdown(f"**Compute:** {DEVICE.upper()}")

    st.markdown("---")
    st.caption("Prototype · Visual quality inspection")


st.markdown("""
<div class="hero">
    <div class="hero-tag">INTELLIGENT QUALITY CONTROL</div>
    <div class="hero-title">Industrial Vision Inspector</div>
    <div class="hero-sub">
        Detect visual abnormalities in product images using
        self-supervised feature-based anomaly detection.
    </div>
</div>
""", unsafe_allow_html=True)

top1, top2, top3 = st.columns(3)

with top1:
    st.markdown("""
    <div class="panel">
        <div class="muted">SYSTEM STATUS</div>
        <div class="panel-title">🟢 Ready to Inspect</div>
        <div class="muted">Upload a product image to begin</div>
    </div>
    """, unsafe_allow_html=True)

with top2:
    st.markdown("""
    <div class="panel">
        <div class="muted">FEATURE EXTRACTOR</div>
        <div class="panel-title">⚡ ResNet18</div>
        <div class="muted">Pretrained visual features</div>
    </div>
    """, unsafe_allow_html=True)

with top3:
    st.markdown("""
    <div class="panel">
        <div class="muted">INSPECTION TARGET</div>
        <div class="panel-title">🧴 Bottle Surface</div>
        <div class="muted">Normality and defect localization</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
left, right = st.columns([0.95, 1.05], gap="large")

with left:
    st.markdown("### 📤 Product Image")
    st.markdown(
        '<div class="muted">Upload a JPG or PNG image for inspection.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose product image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption=f"Input: {uploaded_file.name}",
            use_container_width=True
        )
    else:
        image = None
        st.markdown("""
        <div class="panel">
            <div class="panel-title">Waiting for image</div>
            <div class="muted">
                Select a bottle image above. The inspection result
                will appear in the next panel.
            </div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown("### 📊 Inspection Analysis")

    if image is not None:
        try:
            with st.spinner("Extracting features and analyzing image..."):
                net, memory = load_detector()
                batch = get_transform()(image).unsqueeze(0).to(DEVICE)

                scores, heatmaps = score_batch(net, memory, batch)

                score = float(scores[0])
                heatmap = heatmaps[0]
                prediction = (
                    "Defective" if score >= threshold else "Normal"
                )

            if prediction == "Defective":
                st.error("⚠️ Possible anomaly detected")
            else:
                st.success("✅ Classified as normal")

            c1, c2 = st.columns(2)
            c1.metric("Anomaly Score", f"{score:.4f}")
            c2.metric("Classification", prediction)

            st.progress(min(max(score / 2.0, 0.0), 1.0))
            st.caption(
                f"Threshold: {threshold:.2f} · "
                "The score is model-specific, not a probability."
            )

            st.markdown("### 🌡️ Defect Localization")

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.imshow(image.resize((256, 256)))
            ax.imshow(heatmap, cmap="jet", alpha=0.48)
            ax.axis("off")
            fig.patch.set_facecolor("#131c30")
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Hotter regions indicate higher anomaly scores. "
                "The heatmap is an estimate, not a guaranteed defect mask."
            )

        except Exception as e:
            st.error(f"Inspection failed: {e}")
    else:
        st.info("Your prediction, anomaly score and heatmap will appear here.")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#8999b7;font-size:12px;">'
    'VISIONGUARD AI · SELF-SUPERVISED INDUSTRIAL ANOMALY DETECTION'
    '</div>',
    unsafe_allow_html=True
)

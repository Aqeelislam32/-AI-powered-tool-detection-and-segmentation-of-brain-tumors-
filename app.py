import streamlit as st
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
import io
import yaml

st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: orange;
    color: #1a1a1a;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: orange; }

section[data-testid="stSidebar"] {
    background: skyblue;
    border-right: 1px solid #2a2a3a;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.2rem;
}

.hero-sub {
    color: #374151;
    font-size: 0.95rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-right: 6px;
}

.badge-yolo  { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb33; }
.badge-sam   { background: #1a3a2a; color: #34d399; border: 1px solid #05966933; }
.badge-tumor { background: #3a1a1a; color: #f87171; border: 1px solid #dc262633; }

.result-card {
    background: #13131a;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1.4rem;
    margin-top: 1rem;
}

.metric-box {
    background: #0d0d0f;
    border: 1px solid #2a2a3a;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    text-align: center;
}

.metric-val { font-family: 'Space Mono', monospace; font-size: 1.5rem; color: #a78bfa; }
.metric-lbl { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.07em; }

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.4rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    cursor: pointer;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

.stFileUploader label { color: #a78bfa !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🧠 Brain Tumor Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">YOLO11 Detection · SAM2 Segmentation · Real-time Inference</p>', unsafe_allow_html=True)
st.markdown(
    '<span class="badge badge-yolo">YOLO11</span>'
    '<span class="badge badge-sam">SAM2</span>'
    '<span class="badge badge-tumor">MRI Analysis</span>',
    unsafe_allow_html=True,
)

st.markdown(
    "**Welcome!** This AI-powered tool is designed to assist in the **automatic detection and segmentation of brain tumors** from MRI scans. "
    "By leveraging **YOLO11** for rapid identification and **SAM2** for precise boundary mapping, this application provides "
    "instant visual insights and diagnostic metrics. Simply upload an image and click 'Run Detection' to begin your analysis."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    model_path = st.text_input(
        "YOLO Model Path",
        value="best.pt",
        help="Path to your trained YOLO weights (.pt)",
    )

    yaml_path = st.text_input(
        "Dataset YAML Path",
        value="data.yaml",
        help="Path to your data.yaml file containing class names.",
    )

    use_sam = st.toggle("Enable SAM2 Segmentation", value=True)

    device = st.selectbox(
        "Inference Device",
        options=["cpu", "cuda"],
        index=0,
        help="Choose 'cuda' if you have an NVIDIA GPU installed."
    )

    sam_model_name = st.selectbox(
        "SAM2 Model",
        ["sam2_b.pt", "sam2_l.pt", "sam2_s.pt"],
        disabled=not use_sam,
    )

    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.95, 0.25, 0.05)

    st.markdown("---")
    st.markdown("### 📖 Class Legend")
    
    # Load class names from YAML
    class_names = []
    try:
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            class_names = data_config.get('names', [])
    except Exception:
        st.caption("⚠️ Could not load data.yaml for legend.")

    # Deep, high-contrast colors for Sky Blue background
    colors = ["#1e293b", "#b91c1c", "#c2410c", "#1e40af", "#6d28d9"]
    if class_names:
        for i, name in enumerate(class_names):
            color = colors[i % len(colors)]
            st.markdown(f'<span style="color:{color}; font-weight:700">● {name}</span>', unsafe_allow_html=True)
    else:
        st.info("Load a valid data.yaml to see classes.")

    st.markdown("---")
    st.caption("Built with Ultralytics · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### 📤 Upload MRI Image")
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) selected**")
        preview_cols = st.columns(min(len(uploaded_files), 3))
        for i, f in enumerate(uploaded_files[:3]):
            with preview_cols[i]:
                st.image(f, caption=f.name[:20], use_column_width=True)

    run_btn = st.button("🔍 Run Detection", disabled=not uploaded_files)

# ── Inference ─────────────────────────────────────────────────────────────────
if run_btn and uploaded_files:
    # Lazy imports so the app still loads even if ultralytics not installed
    try:
        from ultralytics import YOLO, SAM
    except ImportError:
        st.error("❌ `ultralytics` not installed. Run: `pip install ultralytics`")
        st.stop()

    # Load models (cached across reruns)
    @st.cache_resource(show_spinner=False)
    def load_yolo(path):
        return YOLO(path)

    @st.cache_resource(show_spinner=False)
    def load_sam(name):
        return SAM(name)

    with st.spinner("Loading models…"):
        try:
            yolo_model = load_yolo(model_path)
        except Exception as e:
            st.error(f"❌ Could not load YOLO model: {e}")
            st.stop()

        sam_model = None
        if use_sam:
            try:
                sam_model = load_sam(sam_model_name)
            except Exception as e:
                st.warning(f"⚠️ SAM model not loaded ({e}). Running YOLO only.")

    with col_result:
        st.markdown("#### 🎯 Detection Results")

        total_detections = 0

        for uploaded_file in uploaded_files:
            # Save upload to temp file
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Analyzing `{uploaded_file.name}`…"):
                try:
                    # ── YOLO inference ────────────────────────────────────
                    yolo_results = yolo_model(
                        tmp_path, 
                        conf=conf_threshold, 
                        device=device,
                        data=yaml_path
                    )
                    result = yolo_results[0]

                    boxes = result.boxes
                    num_det = len(boxes)
                    total_detections += num_det

                    # ── Show annotated image ──────────────────────────────
                    annotated = result.plot()  # numpy array BGR
                    annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

                    # ── SAM segmentation ──────────────────────────────────
                    if sam_model and num_det > 0:
                        sam_results = sam_model(
                            result.orig_img,
                            bboxes=boxes.xyxy,
                            verbose=False,
                            device=device
                        )
                        if sam_results and sam_results[0].masks is not None:
                            # Overlay masks on image
                            masks = sam_results[0].masks.data.cpu().numpy()
                            overlay = annotated_rgb.copy().astype(np.float32)
                            colors = [
                                (167, 139, 250),  # purple
                                (96, 165, 250),   # blue
                                (52, 211, 153),   # green
                                (248, 113, 113),  # red
                            ]
                            for idx, mask in enumerate(masks):
                                color = colors[idx % len(colors)]
                                for c in range(3):
                                    overlay[:, :, c] = np.where(
                                        mask > 0.5,
                                        overlay[:, :, c] * 0.55 + color[c] * 0.45,
                                        overlay[:, :, c],
                                    )
                            annotated_rgb = overlay.astype(np.uint8)

                    st.image(annotated_rgb, caption=uploaded_file.name, use_column_width=True)

                    # ── Download Button ───────────────────────────────────
                    buf = io.BytesIO()
                    Image.fromarray(annotated_rgb).save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Result Image",
                        data=buf.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/png",
                        key=f"dl_{uploaded_file.name}"
                    )

                    # ── Per-image metrics ─────────────────────────────────
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-val">{num_det}</div>'
                            f'<div class="metric-lbl">Detections</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        avg_conf = (
                            f"{boxes.conf.mean().item()*100:.1f}%"
                            if num_det > 0 else "—"
                        )
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-val">{avg_conf}</div>'
                            f'<div class="metric-lbl">Avg Conf</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        sam_status = "✅" if (sam_model and num_det > 0) else "—"
                        st.markdown(
                            f'<div class="metric-box"><div class="metric-val">{sam_status}</div>'
                            f'<div class="metric-lbl">SAM Mask</div></div>',
                            unsafe_allow_html=True,
                        )

                    # ── Box details ───────────────────────────────────────
                    if num_det > 0 and result.names:
                        with st.expander("📋 Box Details"):
                            for i, box in enumerate(boxes):
                                cls_id = int(box.cls.item())
                                cls_name = result.names.get(cls_id, str(cls_id))
                                conf = box.conf.item()
                                xyxy = box.xyxy[0].tolist()
                                st.markdown(
                                    f"**#{i+1}** `{cls_name}` — "
                                    f"conf: `{conf:.3f}` — "
                                    f"bbox: `[{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]`"
                                )
                    elif num_det == 0:
                        st.info("✅ No tumor detected in this image.")

                except Exception as e:
                    st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
                finally:
                    os.unlink(tmp_path)

        # ── Summary ───────────────────────────────────────────────────────
        if len(uploaded_files) > 1:
            st.markdown("---")
            st.markdown(
                f"**Batch summary:** {len(uploaded_files)} images · "
                f"**{total_detections}** total detections"
            )

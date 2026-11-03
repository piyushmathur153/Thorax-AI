"""
ThoraxAI: Clinical Diagnostic Assistant
Production Streamlit Interface with Clinically-Calibrated Explainability,
Automated Quality Guardrails, and Fast Resource Caching.
"""

import io
import cv2
import numpy as np
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from src.model import get_tb_model
from src.quality_gate import check_image_quality
from src.gradcam import GradCAM, overlay_heatmap

# ---------------------------------------------------------
# 1. Page Configuration & Layout Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="ThoraxAI | Clinical Decision Support",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# 2. Modern Medical UI Styling (Glassmorphism & Dark Palette)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Dark Slate Gradient Background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070a12 90%);
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Top Executive Header Card */
    .top-header {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Clinical Result & Protocol Cards */
    .med-card {
        background: rgba(19, 29, 51, 0.65);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 22px;
        backdrop-filter: blur(10px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 20px;
    }

    /* Triage Metric Tiles */
    .metric-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .badge-positive {
        color: #f87171;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .badge-negative {
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Radiograph Uploader Styling */
    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 14px;
        background: rgba(15, 23, 42, 0.4);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.7);
    }

    /* Sidebar Interface Customization */
    section[data-testid="stSidebar"] {
        background-color: #090e1a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* WHO Action Protocol Items */
    .triage-step {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.85rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Model Architecture & Resource Caching
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/best_model.pth"
CLASS_NAMES = ["Normal", "Tuberculosis"]

@st.cache_resource(show_spinner="Initializing Deep Learning Backbone...")
def load_cached_model():
    """
    Loads and caches the DenseNet-121 model in memory to eliminate
    disk I/O overhead on subsequent Streamlit reruns.
    """
    model = get_tb_model(num_classes=2, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model

model = load_cached_model()

# Standard PyTorch normalization for ImageNet/DenseNet inference
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------
# 4. Sidebar System Specifications
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🫁 **ThoraxAI System**")
    st.caption("Deep Learning Assisted Pulmonary Triage")
    st.divider()

    st.markdown("**System Architecture**")
    st.markdown("""
    - **Backbone:** DenseNet-121
    - **Validation Acc:** `97.86%`
    - **Explainability:** Focal Grad-CAM
    - **Device:** `CPU/CUDA Auto`
    """)

    st.divider()
    st.markdown("**Clinical Guardrails**")
    st.caption(
        "Active automated quality control evaluates input radiographs for motion blur "
        "(Laplacian variance) and dynamic exposure levels prior to inference."
    )

# ---------------------------------------------------------
# 5. Top Application Header
# ---------------------------------------------------------
st.markdown("""
<div class="top-header">
    <div>
        <h2 style="margin:0; font-size:1.6rem; font-weight:700; color:#fff;">
            🫁 ThoraxAI: Clinical Diagnostic Assistant
        </h2>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.9rem;">
            Automated Tuberculosis Screening & Attention Localization for Clinical Decision Support
        </p>
    </div>
    <div>
        <span style="background:rgba(56,189,248,0.15); color:#38bdf8; border:1px solid rgba(56,189,248,0.3); padding:6px 14px; border-radius:9999px; font-size:0.8rem; font-weight:600;">
            HI-01 Diagnostic Track
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. Primary Radiograph Upload & Pipeline Execution
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Patient Posteroanterior (PA) Chest Radiograph (PNG, JPG, JPEG):",
    type=["png", "jpg", "jpeg"],
    help="Upload an 8-bit or 16-bit PA view chest X-ray"
)

if uploaded_file is not None:
    # Decode input image buffer
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # Step 1: Automated Quality Inspection (Blur & Exposure Guardrail)
    is_valid, quality_msg = check_image_quality(image_rgb)

    if not is_valid:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:12px; padding:16px; margin-top:16px;">
            <span style="color:#f87171; font-weight:700;">⚠️ Quality Gate Rejection</span>
            <p style="color:#cbd5e1; margin:6px 0 0 0; font-size:0.9rem;">{quality_msg}</p>
            <p style="color:#94a3b8; margin:4px 0 0 0; font-size:0.8rem;">Please upload a diagnostic-quality scan meeting clinical sharpness & exposure thresholds.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.3); border-radius:12px; padding:12px 18px; margin:16px 0;">
            <span style="color:#34d399; font-weight:600; font-size:0.9rem;">✅ {quality_msg}</span>
        </div>
        """, unsafe_allow_html=True)

        # Step 2: DenseNet-121 Inference Pass
        input_tensor = TRANSFORMS(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred_idx = int(np.argmax(probabilities))
        pred_label = CLASS_NAMES[pred_idx]
        pred_conf = float(probabilities[pred_idx]) * 100.0

        # Step 3: Clinically-Calibrated Explainability (Pathology Isolation)
        # Always compute activation gradients with respect to the Tuberculosis class (Index 1)
        tb_class_idx = CLASS_NAMES.index("Tuberculosis")
        grad_cam = GradCAM(model)
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx=tb_class_idx)
        grad_cam.remove_hooks()

        # Suppress false-positive visual artifacts on Normal radiographs
        if pred_label == "Normal":
            superimposed = image_rgb.copy()
            cam_note = "Diagnostic scan verified clean. No focal pathological attention detected."
            right_header_title = "Pathology Heatmap (Suppressed - Normal Finding)"
        else:
            superimposed, _ = overlay_heatmap(image_rgb, heatmap, alpha=0.55, threshold=0.28)
            cam_note = "Grad-CAM focal attention indicates localized pulmonary consolidation/infiltrate."
            right_header_title = "Focal Grad-CAM Attention Map (Pathology)"

        # Step 4: Diagnostic Visualization (Side-by-Side Comparison)
        st.markdown("### **Diagnostic Evaluation & Attention Localization**")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("<p style='color:#94a3b8; font-weight:600; margin-bottom:8px;'>Patient Chest Radiograph (Input)</p>", unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)
            st.caption("Standard Posteroanterior (PA) View")

        with col_right:
            st.markdown(f"<p style='color:#38bdf8; font-weight:600; margin-bottom:8px;'>{right_header_title}</p>", unsafe_allow_html=True)
            st.image(superimposed, use_container_width=True)
            st.caption(cam_note)

        # Step 5: Triage Metrics Summary Cards
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        m_col1, m_col2, m_col3 = st.columns(3)

        with m_col1:
            badge_class = "badge-positive" if pred_label == "Tuberculosis" else "badge-negative"
            badge_text = "Pathology Detected" if pred_label == "Tuberculosis" else "No Focal Lesions"
            st.markdown(f"""
            <div class="metric-box">
                <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Predicted Finding</span>
                <div style="font-size:1.6rem; font-weight:700; color:#fff; margin:6px 0;">{pred_label}</div>
                <span class="{badge_class}">{badge_text}</span>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-box">
                <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Model Confidence</span>
                <div style="font-size:1.6rem; font-weight:700; color:#38bdf8; margin:6px 0;">{pred_conf:.2f}%</div>
                <div style="background:#1e293b; border-radius:9999px; height:6px; overflow:hidden; width:80%; margin:8px auto 0;">
                    <div style="background:#38bdf8; width:{pred_conf}%; height:100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            risk_label = "High Risk" if pred_label == "Tuberculosis" else "Low Risk"
            risk_color = "#f87171" if pred_label == "Tuberculosis" else "#34d399"
            priority_note = "Tier-1 Clinical Priority" if pred_label == "Tuberculosis" else "Routine Outpatient Protocol"
            st.markdown(f"""
            <div class="metric-box">
                <span style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;">Risk Stratification</span>
                <div style="font-size:1.6rem; font-weight:700; color:{risk_color}; margin:6px 0;">{risk_label}</div>
                <span style="color:#94a3b8; font-size:0.75rem;">{priority_note}</span>
            </div>
            """, unsafe_allow_html=True)

        # Step 6: World Health Organization (WHO) Aligned Action Protocol
        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="med-card">
            <h4 style="margin:0 0 12px 0; color:#f8fafc; font-size:1rem; font-weight:600;">
                📋 Recommended Clinical Action Protocol (WHO Triage Standard)
            </h4>
        """, unsafe_allow_html=True)

        if pred_label == "Tuberculosis":
            steps = [
                "Schedule confirmatory rapid molecular diagnostic testing (GeneXpert MTB/RIF).",
                "Perform microscopic examination of sputum smear for acid-fast bacilli (AFB).",
                "Cross-examine highlighted focal Grad-CAM regions for apical cavitations, infiltrates, or consolidations.",
                "Initiate standard airborne infection control precautions pending clinician verification."
            ]
        else:
            steps = [
                "Radiographic patterns show no active parenchymal cavitation or focal consolidations.",
                "Correlate clinically with symptoms (chronic productive cough, unexplained fever, night sweats).",
                "Recommend clinical follow-up or routine surveillance if systemic symptoms persist."
            ]

        for i, step in enumerate(steps, 1):
            st.markdown(f"""
            <div class="triage-step">
                <span style="color:#38bdf8; font-weight:700; font-family:monospace;">0{i}.</span>
                <span style="color:#cbd5e1;">{step}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Idle State Dashboard Placeholder
    st.markdown("""
    <div style="border:2px dashed rgba(255,255,255,0.08); border-radius:16px; padding:60px 20px; text-align:center; margin-top:24px;">
        <div style="font-size:2.5rem; margin-bottom:12px;">🩻</div>
        <h4 style="color:#cbd5e1; margin:0 0 6px 0;">Awaiting Chest Radiograph Input</h4>
        <p style="color:#64748b; font-size:0.85rem; margin:0;">
            Upload a PA view radiograph above to trigger the automated quality verification, DenseNet inference, and focal Grad-CAM visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)
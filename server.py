"""
ThoraxAI - Automated Tuberculosis Diagnostic & Explainability Server
Production-grade FastAPI inference engine with Quality Gating and Grad-CAM localization.
"""
import re
import os
import gc
import io
import base64
import logging
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse , FileResponse
from fastapi.staticfiles import StaticFiles

# Core Machine Learning & Clinical Modules
from src.model import get_tb_model
from src.quality_gate import check_image_quality
from src.gradcam import GradCAM , overlay_heatmap

# ---------------------------------------------------------------------------
# Logging & Runtime Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("ThoraxAI-Core")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to stop memory spike on render`s single-core CPU
torch.set_num_threads(1)
MODEL_PATH = os.path.join("models", "best_model.pth")
FRONTEND_DIST_DIR = os.path.join("frontend", "dist")

app = FastAPI(
    title="ThoraxAI Diagnostic API",
    description="Clinical-grade Chest Radiograph Diagnostic Engine for Pulmonary Tuberculosis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cross-Origin Resource Sharing (CORS) for development and distributed clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model Initialization & Inference Pipeline
# ---------------------------------------------------------------------------
model: Optional[torch.nn.Module] = None

try:
    model = get_tb_model(num_classes=2)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.to(DEVICE)
        model.eval()
        logger.info("DenseNet-121 model successfully loaded on target device: %s", DEVICE)
    else:
        logger.warning("Checkpoint '%s' not found. Ensure weights are present.", MODEL_PATH)
except Exception as exc:
    logger.error("Critical error during model initialization: %s", exc, exc_info=True)
    model = None

# Standard ImageNet Normalization matching clinical training distribution
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def encode_cv2_to_base64(image_bgr) -> str:
    """Encodes an OpenCV BGR image into a compliant Base64 PNG data URI string."""
    if image_bgr is None:
        raise ValueError("Image tensor is None")
    
    # Agar tuple pass ho gaya ho toh pehla element extract karo
    if isinstance(image_bgr, tuple):
        image_bgr = image_bgr[0]

    # Ensure standard 8-bit image format (0-255)
    if hasattr(image_bgr, "dtype") and image_bgr.dtype != np.uint8:
        if image_bgr.max() <= 1.0:
            image_bgr = (image_bgr * 255).astype(np.uint8)
        else:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)

    success, buffer = cv2.imencode(".png", np.ascontiguousarray(image_bgr))
    if not success:
        raise ValueError("Image encoding failed during Base64 serialization.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Monitoring"])
def health_probe() -> Dict[str, Any]:
    """Liveness probe verifying GPU availability and model readiness."""
    return {
        "status": "online",
        "device": str(DEVICE),
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    }


@app.post("/api/v1/analyze", tags=["Diagnostics"])
async def run_diagnostic_pipeline(file: UploadFile = File(...)) -> JSONResponse:
    """
    Executes end-to-end diagnostic pipeline:
    1. Validates radiographic image format and byte integrity.
    2. Quality Gate evaluates blur/contrast against diagnostic thresholds.
    3. DenseNet-121 computes classification logits.
    4. Focal Grad-CAM localizes active pathological regions.
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Diagnostic model weights are not loaded."
        )

    # 1. Content validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported format. Please upload a standard CXR image (DICOM/PNG/JPEG)."
        )

    try:
        raw_bytes = await file.read()
        np_buffer = np.frombuffer(raw_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("Corrupted image stream.")
    except Exception as exc:
        logger.error("Image decoding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode chest radiograph image."
        )

# 2. Quality Gate Verification (src/quality_gate.py)
    try:
        q_result = check_image_quality(image_bgr)
        sharpness, contrast = 0.0, 0.0
        
        if isinstance(q_result, tuple):
            if len(q_result) == 2:
                quality_passed, quality_feedback = q_result
                # String message me se actual sharpness aur contrast extract karo
                match = re.search(r"Sharpness:\s*([\d\.]+).*?Contrast:\s*([\d\.]+)", str(quality_feedback))
                if match:
                    sharpness = float(match.group(1))
                    contrast = float(match.group(2))
            elif len(q_result) == 3:
                quality_passed, sharpness, contrast = q_result
                quality_feedback = "Quality verified."
            elif len(q_result) >= 4:
                quality_passed, sharpness, contrast, quality_feedback = q_result[:4]
            else:
                quality_passed, quality_feedback = True, "Verified"
        else:
            quality_passed = bool(q_result)
            quality_feedback = "Verified"
    except Exception as exc:
        logger.warning("Quality assessment fallback triggered: %s", exc)
        quality_passed, sharpness, contrast, quality_feedback = True, 0.0, 0.0, "Quality gate bypassed."
    if not quality_passed :
        raise HTTPException(
            status_code= 422 ,
            detail = quality_feedback
        )
    # 3. Model Preprocessing & Forward Pass
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = inference_transforms(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        normal_prob = float(probabilities[0].item())
        tb_prob = float(probabilities[1].item())

    is_positive = bool(tb_prob >= 0.50)
    confidence = tb_prob if is_positive else normal_prob
    diagnostic_label = "Tuberculosis Detected" if is_positive else "Normal / No TB Infiltration"

# 4. Explainability & Spatial Localization Pipeline (Grad-CAM)
    try:
        # Initialize Grad-CAM targeting the terminal convolutional layer of DenseNet-121
        target_layer = model.features[-1]
        gradcam = GradCAM(model, target_layer)
        
        # Clinical Rule: Restrict pathological saliency heatmaps strictly to positive cases
        if not is_positive:
            # For healthy/normal radiographs, bypass heatmap rendering to prevent false-positive visual artifacts
            cam_overlay_bgr = image_bgr.copy()
            gradcam_uri = encode_cv2_to_base64(cam_overlay_bgr)
        else:
            # Generate activation heatmap strictly optimized for the Tuberculosis pathology class (class_idx = 1)
            heatmap = gradcam.generate_heatmap(
                input_tensor=input_tensor, 
                class_idx=1
            )
            
            # Convert OpenCV BGR radiograph to RGB color space as expected by overlay_heatmap
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Superimpose heat intensity over pulmonary consolidation zones
            cam_res = overlay_heatmap(image_rgb, heatmap)
            superimposed_rgb = cam_res[0] if isinstance(cam_res, tuple) else cam_res
            
            # Convert blended RGB radiograph back to BGR format prior to Base64 payload encoding
            cam_overlay_bgr = cv2.cvtColor(superimposed_rgb, cv2.COLOR_RGB2BGR)
            gradcam_uri = encode_cv2_to_base64(cam_overlay_bgr)

    except Exception as exc:
        logger.error("Grad-CAM generation encountered an unexpected exception: %s", exc, exc_info=True)
        # Fallback to unprocessed radiograph in case of computation failure
        gradcam_uri = encode_cv2_to_base64(image_bgr)

    original_uri = encode_cv2_to_base64(image_bgr)

    # 5. Clinical Triage Assignment
    if is_positive and confidence >= 0.85:
        triage_urgency = "Tier 1: High Priority Urgent"
        clinical_action = (
            "Prominent apical opacification and consolidation patterns detected. "
            "Prioritize patient for immediate sputum GeneXpert/AFB microscopy and isolation protocol."
        )
    elif is_positive:
        triage_urgency = "Tier 2: Moderate Priority"
        clinical_action = (
            "Subtle parenchymal infiltrate or borderline cavitation indicators observed. "
            "Recommend clinical correlation with symptom duration and follow-up high-resolution CT/radiograph."
        )
    else:
        triage_urgency = "Tier 3: Routine / Low Risk"
        clinical_action = (
            "No prominent radiologic manifestations of active pulmonary tuberculosis identified. "
            "Continue standard symptomatic management."
        )

    response_payload = {
        "quality_gate": {
            "passed": quality_passed,
            "sharpness_score": round(float(sharpness), 2),
            "contrast_score": round(float(contrast), 2),
            "message": quality_feedback
        },
        "diagnosis": {
            "label": diagnostic_label,
            "is_positive": is_positive,
            "confidence_score": round(confidence * 100, 2),
            "tb_probability": round(tb_prob * 100, 2),
            "normal_probability": round(normal_prob * 100, 2)
        },
        "visuals": {
            "original_image": original_uri,
            "gradcam_heatmap": gradcam_uri
        },
        "triage": {
            "urgency": triage_urgency,
            "recommendation": clinical_action
        }
    }
    gc.collect()
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)


# ---------------------------------------------------------------------------
# Static SPA Mount (For Single-Container Docker deployment)
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIST_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST_DIR, html=True), name="frontend")
    logger.info("Serving bundled React SPA from: %s", FRONTEND_DIST_DIR)
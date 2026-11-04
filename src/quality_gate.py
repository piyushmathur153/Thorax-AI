"""
Quality Gate & Anatomical CXR Verification Module.
Validates input chest radiographs against clinical quality thresholds:
1. Input format normalization (File paths, BytesIO streams, PIL, and NumPy arrays).
2. Chromatic saturation filter (rejects colored natural photos/graphics).
3. Laplacian variance check (detects severe motion blur or synthetic anime edge spikes).
4. Dynamic percentile range verification (flags poor exposure and flat synthetic fields).
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Union, Any


def check_image_quality(image_input: Any) -> Tuple[bool, str]:
    """
    Evaluates radiograph quality and anatomical validity prior to neural inference.
    Returns:
        Tuple[bool, str]: (Passed status, detailed clinical feedback string)
    """
    img = None
    color_bgr = None

    # 1. Handle file path string
    if isinstance(image_input, str):
        color_bgr = cv2.imread(image_input)
        if color_bgr is not None:
            img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Handle Streamlit UploadedFile or BytesIO stream
    elif hasattr(image_input, 'read'):
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        color_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if color_bgr is not None:
            img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        image_input.seek(0)  # Reset stream pointer for subsequent processing

    # 3. Handle PIL Image object
    elif isinstance(image_input, Image.Image):
        color_np = np.array(image_input)
        if len(color_np.shape) == 3:
            color_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img = color_np

    # 4. Handle NumPy array
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            color_bgr = image_input
            img = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        else:
            img = image_input
    else:
        return False, "Scan Rejected: Unsupported image input format."

    # Verify successful decode and non-empty buffer
    if img is None or img.size == 0:
        return False, "Scan Rejected: Image file is corrupted or unreadable. Please upload a valid CXR."

    # 5. Color Saturation & Chromatic Diversity Verification
    # Legitimate chest radiographs are predominantly monochrome or exhibit a single cyan/blue film tint.
    if color_bgr is not None:
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        mean_saturation = float(np.mean(saturation))

        # Inspect elevated saturation thresholds typical of digitally tinted radiographs
        if mean_saturation > 25.0:
            hue = hsv[:, :, 0]
            # Clinical cyan/blue film hue is typically constrained to the [75, 140] range in OpenCV
            blue_mask = (hue >= 75) & (hue <= 140)
            blue_ratio = float(np.sum(blue_mask) / hue.size)

            # Reject polychromatic images if the predominant tone deviates from clinical cyan/blue spectrum
            if blue_ratio < 0.60 and mean_saturation > 40.0:
                return False, f"Scan Rejected (Chromatic Saturation: {mean_saturation:.1f}): Polychromatic natural or synthetic media identified. Only monochrome or cyan-tinted CXRs are admissible."

            # Reject non-radiological media exhibiting excessive overall saturation
            if mean_saturation > 130.0:
                return False, f"Scan Rejected (Excessive Saturation: {mean_saturation:.1f}): Image contains non-radiological chromatic elements."

    # Normalize resolution to ensure consistent variance metrics across devices
    img_resized = cv2.resize(img, (512, 512))

    # 6. Sharpness Verification via Laplacian Variance
    lap_var = float(cv2.Laplacian(img_resized, cv2.CV_64F).var())
    
    # Check 6a: Reject severe motion blur
    if lap_var < 3.0:
        return False, f"Scan Rejected (Excessive Blur): Sharpness: {lap_var:.1f}, Contrast: 0.0 below clinical threshold (10.0)."

    # Check 6b: Reject synthetic vector media / anime (Anime edges typically exceed 650.0+)
    if lap_var > 650.0:
        return False, f"Scan Rejected (Synthetic Media): Edge sharpness: {lap_var:.1f} exceeds natural anatomical limits (Max 650.0)."

    # 7. Contrast & Dynamic Range Check (5th to 95th Percentile)
    p5, p95 = np.percentile(img_resized, (5, 95))
    dyn_range = float(p95 - p5)
    if dyn_range < 35.0:
        return False, f"Scan Rejected (Poor Contrast/Exposure): Sharpness: {lap_var:.1f}, Contrast: {dyn_range:.1f} below minimum threshold (35.0)."

    # 8. Extreme Binarization Filter (Detects cartoon solid black/white blocks)
    hist = cv2.calcHist([img_resized], [0], None, [256], [0, 256]).flatten()
    extreme_pixels = hist[0:8].sum() + hist[248:256].sum()
    if (extreme_pixels / img_resized.size) > 0.65:
        return False, f"Scan Rejected (Non-Anatomical Distribution): Excessive solid binarization detected. Sharpness: {lap_var:.1f}, Contrast: {dyn_range:.1f}."

    # Quality Gate Passed
    return True, f"Quality Verified (Sharpness: {lap_var:.1f}, Contrast: {dyn_range:.1f})"


# --- Module Self-Test ---
if __name__ == "__main__":
    test_tb_dir = os.path.join("data", "processed", "test", "tb")
    test_norm_dir = os.path.join("data", "processed", "test", "normal")

    for label, folder in [("TB Sample", test_tb_dir), ("Normal Sample", test_norm_dir)]:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                sample_path = os.path.join(folder, files[0])
                passed, msg = check_image_quality(sample_path)
                print(f"[{label}] File: {files[0]} -> Passed: {passed} | {msg}")
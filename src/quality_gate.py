import cv2
import numpy as np
from PIL import Image

def check_image_quality(image_input):
    """
    Evaluates chest radiograph quality prior to model inference:
    1. Sharpness Check: Laplacian variance >= 15.0 (calibrated for soft-tissue chest radiographs)
    2. Dynamic Range Check: 95th - 5th percentile >= 40.0 (flags underexposed, overexposed, or blank scans)
    """
    img = None

    # 1. Handle file path string
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    
    # 2. Handle Streamlit UploadedFile or BytesIO stream
    elif hasattr(image_input, 'read'):
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image_input.seek(0)  # Reset stream pointer for subsequent PIL processing
        
    # 3. Handle PIL Image object
    elif isinstance(image_input, Image.Image):
        img = np.array(image_input.convert("L"))
        
    # 4. Handle NumPy array
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            img = cv2.cvtColor(image_input, cv2.COLOR_RGB2GRAY)
        else:
            img = image_input
    else:
        return False, "Unsupported image input format."

    if img is None or img.size == 0:
        return False, "Image file is corrupted or unreadable. Please upload a valid radiograph."

    # Normalize resolution to ensure consistent variance metrics across devices
    img_resized = cv2.resize(img, (512, 512))

    # Check 1: Blur / Sharpness verification
    lap_var = float(cv2.Laplacian(img_resized, cv2.CV_64F).var())
    if lap_var < 6.0:
        return False, f"Scan Rejected (Excessive Blur): Sharpness score ({lap_var:.1f}) is below the minimum clinical threshold (15.0)."

    # Check 2: Contrast / Dynamic range verification
    p5, p95 = np.percentile(img_resized, (5, 95))
    dyn_range = float(p95 - p5)
    if dyn_range < 40.0:
        return False, f"Scan Rejected (Poor Contrast/Exposure): Dynamic range ({dyn_range:.1f}) is below the minimum threshold (40.0)."

    return True, f"Quality Verified (Sharpness: {lap_var:.1f}, Contrast: {dyn_range:.1f})"


# --- Module Self-Test ---
if __name__ == "__main__":
    import os
    test_tb_dir = os.path.join("data", "processed", "test", "tb")
    test_norm_dir = os.path.join("data", "processed", "test", "normal")
    
    for label, folder in [("TB Sample", test_tb_dir), ("Normal Sample", test_norm_dir)]:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                sample_path = os.path.join(folder, files[0])
                passed, msg = check_image_quality(sample_path)
                print(f"[{label}] File: {files[0]} -> Passed: {passed} | {msg}")
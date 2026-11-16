# ThoraxAI — Explainable Pulmonary TB Screening & Triage System

> An end-to-end, low-footprint deep learning web application for rapid Tuberculosis screening from chest radiographs using DenseNet-121 and Grad-CAM explainability.

🌐 **Live Web App:** [thorax-ai.onrender.com](https://thorax-ai.onrender.com)  
🛠️ **Tech Stack:** Python 3.11 • PyTorch (CPU) • FastAPI • OpenCV • React 18 • Tailwind CSS • Docker

---

## 1. Overview

In rural primary healthcare centers (PHCs) and district hospitals, trained radiologists are scarce, and generic "black-box" AI models output raw percentages that doctors cannot clinically trust. 

**ThoraxAI** bridges this gap by combining computer vision with visual explainability and edge-level resource efficiency:
- **Pre-Inference Quality Gate:** Uses OpenCV (Laplacian blur variance & contrast checks) to reject blurry, inverted, or non-medical uploads before running inference.
- **Zero Black-Box AI (Grad-CAM):** Generates pixel-aligned anatomical heatmaps highlighting exact lung regions (apical consolidations, opacities) that triggered the prediction.
- **Structured Clinical Triage:** Converts model probabilities into actionable medical protocols (`CRITICAL`, `URGENT_REFERRAL`, `ROUTINE_MONITOR`) along with a print-ready referral slip.
- **<512 MB Cloud Footprint:** Engineered to run smoothly on single-threaded free-tier cloud CPU instances without GPU dependencies or Out-Of-Memory (OOM) crashes.

> **Deployment Note:** Hosted on Render with continuous 5-minute automated health pings (UptimeRobot) to keep the container warm and ensure immediate responsiveness.

---

## 2. Scan Inference Turnaround Time (5–7 Seconds)

A full diagnostic screening run takes approximately **5 to 7 seconds** per Chest X-ray on a single-threaded cloud CPU without requiring dedicated GPUs:
- Full-resolution OpenCV Laplacian focus and contrast validation.
- DenseNet-121 feedforward probability distribution pass.
- Multi-layer gradient backpropagation (`denseblock4`) to generate Grad-CAM heatmaps.
- High-fidelity Jet colormap overlay rendering and explicit garbage collection (`gc.collect()`).

---

## 3. Clinical Need & Triage Workflow

Rural primary healthcare centers often lack trained radiologists, making early Tuberculosis detection difficult. **ThoraxAI** assists frontline doctors by filtering out blurry/invalid scans first, highlighting suspicious lung regions visually via Grad-CAM, and mapping risk scores into clear clinical actions (`CRITICAL`, `URGENT_REFERRAL`, `ROUTINE_MONITOR`).



## 4. Key Innovations & Engineering Highlights

What separates **ThoraxAI** from standard academic prototypes:

1. **Pre-Inference Automated Gatekeeper (Quality Check):**
   - *The Problem:* Most medical ML web tools crash or generate hallucinated false-positive predictions when presented with non-medical photos, inverted scans, or severe motion blur.
   - *The Solution:* ThoraxAI runs an automated OpenCV-based diagnostic gate (Laplacian blur variance and dynamic contrast validation) before tensors reach the deep neural network.

2. **Zero "Black-Box" AI (Physician-First Trust):**
   - *The Problem:* Doctors in primary clinics cannot ethically act on raw percentages like `"TB: 88%"`.
   - *The Solution:* Integrated Grad-CAM generates anatomical heatmaps highlighting exact pulmonary regions (apical consolidations, hilar opacities, pleural thickening) responsible for the classification, keeping clinicians firmly in control.

3. **Radical Hardware Efficiency (<512MB Cloud Budget):**
   - *The Problem:* Advanced vision models usually demand high-cost GPU servers, making them impractical for under-funded rural clinics.
   - *The Solution:* Engineered specifically for zero-cost, resource-constrained runtimes via CPU-only PyTorch wheels, single-core thread throttling (`torch.set_num_threads(1)`), and active garbage-collection sweeps.

4. **Actionable Triage Protocol Over Raw Classification:**
   - *The Problem:* Basic classifiers stop at simple label prediction.
   - *The Solution:* Directly converts risk probabilities into tiered, standard-of-care clinical actions (`CRITICAL`, `URGENT_REFERRAL`, `ROUTINE_MONITOR`) with follow-up protocols (e.g., CBNAAT / GeneXpert confirmation).

5. **Self-Contained Full-Stack Deployment:**
   - *The Problem:* Multi-server setups create CORS conflicts, higher latency, and increased hosting costs.
   - *The Solution:* A unified multi-stage Docker build compiles the React 18 production bundle directly into FastAPI, delivering a single-container service.

6. **Zero-Data-Retention & Instant Clinical Referral:**
   - *The Problem:* Storing sensitive patient medical scans on cloud servers introduces privacy risks, while clinics often lack instant physical documentation for specialist handoff.
   - *The Solution:* ThoraxAI generates immediate print-ready referral receipts and caches diagnostic history strictly within client-side browser storage—guaranteeing zero server-side patient data retention.
   - 

## 5. Dataset & Citation Credit

ThoraxAI was trained and evaluated using the official benchmark database:

* **Dataset Name:** Tuberculosis (TB) Chest X-ray Database
* **Source Platform:** Kaggle
* **Direct Dataset URL:** [https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
* **Creators & Research Collaborators:** Created by a research team from Qatar University (Doha, Qatar), the University of Dhaka (Bangladesh), and international collaborators from Malaysia, in direct collaboration with medical doctors from Hamad Medical Corporation and Bangladesh.
* **Composition:**
  * 3,500 Normal chest radiograph scans
  * Active Tuberculosis (TB) positive chest radiograph scans
* **Scholarly Citation & Credit:**
  > Tawsifur Rahman, Amith Khandakar, Muhammad A. Kadir, Khandaker R. Islam, Khandaker F. Islam, Zaid B. Mahbub, Mohamed Arselene Ayari, Muhammad E. H. Chowdhury. (2020). *"Reliable Tuberculosis Detection using Chest X-ray with Deep Learning, Segmentation and Visualization."* **IEEE Access**.

---

## 6. Model Optimization & Resource Efficiency (512MB RAM Compliance)

Deploying a deep convolutional model (DenseNet-121) alongside real-time Grad-CAM synthesis within Render's free tier (strictly capped at 512MB RAM) requires rigorous systems engineering. Standard PyTorch deployments easily crash with Out-Of-Memory (OOM) errors. We implemented five decisive optimizations:

1. **Custom CPU-Only PyTorch Distribution:**
   * Replaced bloated CUDA/cuDNN GPU distributions with official CPU wheels (`--index-url https://download.pytorch.org/whl/cpu`).
   * Reduced baseline library disk and memory footprint from 850MB+ to ~160MB.
2. **Deterministic Single-Thread Throttling:**
   * Enforced `torch.set_num_threads(1)` on the server runtime.
   * Restricts CPU core contention and suppresses random memory allocation spikes during tensor forward passes.
3. **Active Post-Inference Garbage Collection Hook:**
   * Placed explicit `gc.collect()` triggers immediately following diagnostic payload serialization and Grad-CAM buffer compilation.
   * Clears unreferenced intermediate tensors, gradient graphs, and OpenCV image buffers, ensuring memory returns to baseline immediately after every single scan.
4. **Detached Graph & Inference Optimization:**
   * Model inference executes under strict `torch.no_grad()` blocks with the neural architecture locked in `model.eval()` mode.
   * Targeted autograd hooks are enabled selectively on the final dense block layer (`features.denseblock4`) only during the Grad-CAM backprop pass and are instantly discarded.
5. **Single-Container Multi-Stage Bundle:**
   * Avoids running separate Node.js and Python servers.
   * Compiles React to static assets in Stage 1, then mounts them directly into FastAPI inside a lightweight Python 3.11-slim container.

---

## 7. End-to-End Pipeline & Architecture

```text
[ User Upload (DICOM / JPEG / PNG) ]
                |
                v
+-------------------------------------------------------------+
| STEP 1: Pre-Inference Quality Assurance Gate (OpenCV)       |
| • Laplacian Variance Blur Check (variance >= 100.0)         |
| • Contrast Distribution & Mean Pixel Intensity Validation   |
| • Instant rejection of invalid, non-medical, or blurry scans|
+-------------------------------------------------------------+
                                |
                           (Valid Scan)
                                |
                                v
+-------------------------------------------------------------+
| STEP 2: Deep Learning Inference (DenseNet-121)              |
| • Dense inter-layer feature concatenation                   |
| • Pre-trained weights fine-tuned on pulmonary opacities     |
| • Softmax-calibrated probability generation                 |
+-------------------------------------------------------------+
                |                               |
                v                               v
+-----------------------------+ +-----------------------------+
| STEP 3: Grad-CAM Engine     | | STEP 4: Clinical Triage     |
| • Target: denseblock4       | | • Urgency Level Assignment  |
| • Jet Colormap Overlay      | | • Actionable Follow-up Path |
| • Anatomical Heatmap URI    | | • Confirmatory Test Advice  |
+-----------------------------+ +-----------------------------+
                \                               /
                 \-----------------------------/
                                |
                                v
+-------------------------------------------------------------+
| STEP 5: Delivery & State Management                         |
| • FastAPI Async JSON Response Payload                       |
| • Explicit gc.collect() memory clearing                     |
| • High-contrast React 18 / Tailwind Dashboard View          |
+-------------------------------------------------------------+
                                |
                                v
+-------------------------------------------------------------+
| STEP 6: Clinical Handoff & Privacy-First Audit Trail        |
| • Printable Doctor Referral Slip (Heatmap & Probabilities)  |
| • Zero Cloud Storage: Session History Cached in LocalStorage|
+-------------------------------------------------------------+


```

---

## 8. Key Features & Clinical Capabilities

* **Automated Quality Gatekeeper:** Detects motion artifacts, severe underexposure, and non-CXR imagery, preventing erroneous diagnostic predictions.
* **Calibrated Dual Probability Readout:** Provides explicit percentage scores for both Tuberculosis probability and Normal pulmonary probability.
* **Dynamic Visual Explainability (Grad-CAM):** Pinpoints apical parenchymal consolidations, nodular opacities, and pleural thickening directly on the patient's radiograph.
* **Tiered Actionable Clinical Protocols:**
  * **CRITICAL:** Immediate patient isolation, rapid sputum induction, urgent pulmonology specialist consult.
  * **URGENT_REFERRAL:** Priority referral for CBNAAT (GeneXpert) / Sputum Smear Microscopy confirmation within 24–48 hours.
  * **ROUTINE_MONITOR / NORMAL:** No radiographic evidence of active TB; periodic routine surveillance or differential follow-up if symptoms persist.
* **High-Accessibility Clinical UI:** Designed with medical-grade dark slate aesthetic, clear triage status badges, and interactive original vs. heatmap side-by-side comparison.
* One-Click Clinical Referral Receipt: Instantly generates a print-ready referral slip containing patient metadata, AI diagnostic findings, triage urgency level, and the Grad-CAM heatmap for fast specialist handoff.
* Local Diagnostic History (Privacy-First): Caches previous scans and evaluation results directly within browser local storage, enabling healthcare staff to review audit trails without sending patient data to external servers.
*
---

## 9. Model Performance & Benchmark Metrics

Evaluated on validation splits from the Tuberculosis (TB) Chest X-ray Database:

| Metric | Score | Clinical Interpretation |
| :--- | :--- | :--- |
| **Sensitivity / Recall (TB Positive)** | **94.8%** | Minimizes false negatives; critical for triage screening |
| **Specificity (Normal Radiographs)** | **92.3%** | Minimizes unnecessary referrals from primary clinics |
| **Overall Classification Accuracy** | **93.6%** | High concordance with verified clinical reads |
| **Area Under ROC Curve (AUC-ROC)** | **0.967** | Strong discrimination across varied decision thresholds |
| **Cloud Inference Latency (CPU)** | **~5 to 7s** | Complete evaluation including full Grad-CAM generation |

---

## 10. How to Run ThoraxAI (User & Clinician Workflow)

1. **Step 1:** Open the live web app in your browser: [https://thorax-ai.onrender.com](https://thorax-ai.onrender.com)
2. **Step 2:** On the diagnostic dashboard, click the upload region or drag and drop a standard frontal chest radiograph (JPEG or PNG format).
3. **Step 3:** Click **"Run AI Screening & Triage Analysis"**.
4. **Step 4:** The system executes image quality checks and evaluates neural features.
5. **Step 5:** View the results:
   * Triage Urgency badge (`CRITICAL` / `URGENT_REFERRAL` / `NORMAL`).
   * Probability breakdown and diagnostic confidence score.
   * Grad-CAM heatmap visualization highlighting suspicious lung zones.
   * Next-step clinical protocol instructions.

---

## 11. Screenshots & Interface Preview

*(The live interactive deployment is actively accessible at [https://thorax-ai.onrender.com](https://thorax-ai.onrender.com))*

| Diagnostic Clinical Dashboard | Grad-CAM Explainability Heatmap |
| :---: | :---: |
| *Frontal chest radiograph upload, triage badge readout, and probability distribution* | *High-gradient pulmonary activation overlay showing apical opacities* |

| Pre-Inference Quality Filter Gate | Structured Clinical Triage Protocol |
| :---: | :---: |
| *Automated rejection of blurred images and contrast-deficient scans* | *Tiered referral pathway and WHO-compliant testing guidelines* |

---

## 12. Local Installation & Environment Setup Guide

### Prerequisites
* Python 3.10 or 3.11
* Git
* *(Optional)* Node.js 18+ and npm (only required if modifying the React UI)

### Step-by-Step Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/piyushmathur153/Thorax-AI.git
   cd Thorax-AI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   * On Windows:
     ```bash
     venv\Scripts\activate
     ```
   * On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

4. **Install the lightweight CPU PyTorch wheel:**
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

5. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

6. **Start the FastAPI application:**
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 7860 --reload
   ```

7. Open your web browser at: `http://localhost:7860`

---

## 13. Production Docker Container Deployment

To run the exact single-container multi-stage build locally with a 512MB RAM cap:

1. **Build the Docker image:**
   ```bash
   docker build -t thoraxai .
   ```

2. **Run the container with explicit memory limitation:**
   ```bash
   docker run -p 7860:7860 --memory="512m" thoraxai
   ```

Access the running container at `http://localhost:7860`.

---

## 14. Clinical & Medical Disclaimer

**ThoraxAI is an experimental research and educational decision-support prototype.**

- **Not a Certified Medical Device:** This software does not replace or override the professional medical judgment, diagnosis, or care of a licensed radiologist or physician.
- **Confirmatory Testing Required:** Probability metrics and Grad-CAM heatmaps are meant strictly for preliminary screening assistance. All findings must be verified through standard microbiological diagnostics (such as Sputum Smear Microscopy and CBNAAT/GeneXpert) and clinical evaluation.

---

## 15. Acknowledgments & Credits

- **Dataset:** Trained and validated on the [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) curated by Tawsifur Rahman et al. (Qatar University & University of Dhaka).
- 

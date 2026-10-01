========================================================================
BRAIN TUMOR DETECTION USING ML AND MEDICAL IMAGING (NEUROSCAN AI v2.0)
R. C. Patel Institute of Technology, Shirpur (SES)
Department of Computer Science & Engineering (Data Science)
========================================================================

Recovered and enhanced project files with state-of-the-art Deep Residual ConvNet
(ResNet-MedNet), Explainable AI (Grad-CAM heatmaps), and Modern Web UI.

------------------------------------------------------------------------
HOW TO RUN:
------------------------------------------------------------------------
1. Open this folder in VS Code or Terminal.

2. Activate Python virtual environment:
   .venv\Scripts\activate

3. Install dependencies (if needed):
   pip install -r requirements.txt

4. OPTION A - Start Modern High-End Radiology Workstation Web UI (Recommended):
   python server.py
   Then open your browser at: http://127.0.0.1:8000
   Features:
   - High-End Cyber-Medical Dark Glassmorphic Workstation
   - Dual-View MRI Canvas with Real-Time Grad-CAM Opacity Blend Slider
   - Side-by-Side Comparison (Original MRI vs Grad-CAM Localization Map)
   - Interactive 1-Click Clinical Samples (Glioma, Meningioma, Pituitary, Normal)
   - Printable Hospital Clinical Diagnostic Report (with PDF Export)
   - Model Benchmark & Viva Defense Hub (Page 31 Report Comparison & Confusion Matrix)

5. OPTION B - Start Streamlit Application:
   streamlit run app.py
   Then open your browser at: http://localhost:8501

------------------------------------------------------------------------
PROJECT STRUCTURE:
------------------------------------------------------------------------
- train_improved_model.py : Trains the ResNet-MedNet model with data augmentation,
                            residual skip connections, Batch Normalization, and GAP.
- gradcam.py              : Computes Grad-CAM heatmaps and tumor lesion localization.
- model_engine.py         : Unified inference engine, clinical profiles, and samples.
- server.py               : FastAPI backend server hosting API and static UI.
- static/                 : Modern UI frontend (index.html, styles.css, app.js).
- app.py                  : Upgraded Streamlit application with Grad-CAM and custom CSS.
- dataset/                : Training (5,600 images) and Testing (1,600 images) MRI sets.
- brain_tumor_mednet.keras: Trained high-accuracy ResNet-MedNet model weights.
- model_metrics.json      : Multi-metric evaluation and viva benchmark results.
========================================================================

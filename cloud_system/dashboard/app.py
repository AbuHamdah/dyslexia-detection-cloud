"""
Streamlit Clinical Dashboard — Dyslexia Detection Cloud System.

Pages:
  🏠 Home              — system overview
  🧠 MRI Prediction    — upload structural MRI
  📊 fMRI Prediction   — upload functional MRI
  🔬 Multimodal Fusion — upload both for combined prediction
  ⚙️ System Status     — health & model status
"""

import os
import streamlit as st
import requests
import json
import time

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Dyslexia Detection – Cloud Dashboard",
    page_icon="🧠",
    layout="wide",
)


# ── Sidebar navigation ──

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🧠 MRI Prediction", "📊 fMRI Prediction",
     "🔬 Multimodal Fusion", "⚙️ System Status"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dyslexia Detection Cloud System**  \n"
    "Agentic AI models for structural & functional MRI analysis."
)


# ── Helper ──

def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def show_result(result: dict, title: str = "Prediction Result"):
    """Display a prediction result card."""
    label = result.get("label", result.get("fusion_label", "unknown"))
    conf = result.get("confidence", result.get("fusion_confidence", 0))
    color = "🔴" if label == "dyslexic" else "🟢"

    st.markdown(f"### {title}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Classification", f"{color} {label.upper()}")
    col2.metric("Confidence", f"{conf*100:.1f}%")
    col3.metric("Processing Time", f"{result.get('processing_time_ms', 0):.0f} ms")

    with st.expander("Raw JSON"):
        st.json(result)


# ═══════════════════════════════════════════════
# PAGE: Home
# ═══════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🧠 Dyslexia Detection Cloud System")
    st.markdown(
        """
        Welcome to the **Agentic AI Dyslexia Detection** clinical dashboard.

        This system uses deep learning models trained with an **agentic LLM-guided
        optimization** loop to classify dyslexia from neuroimaging data.

        ### Models
        | Model | Architecture | Input |
        |-------|-------------|-------|
        | **3D-CNN** | Conv3D (16→32→64) → GAP → Dense | Structural MRI |
        | **CNN-LSTM** | TD-Conv2D + BiLSTM | Functional MRI |
        | **HM Fusion** | α·MRI + β·fMRI weighted voting | Both modalities |
        | **Agentic Fusion** | Feature concat → trainable head | Both modalities |

        ### Architecture
        - **Backend**: FastAPI REST API with TensorFlow Serving
        - **Frontend**: This Streamlit dashboard
        - **Deployment**: Docker containers with NGINX reverse proxy
        - **Cloud-ready**: Runs on any cloud (AWS, GCP, Azure) or localhost
        """
    )

    health = api_health()
    if health:
        st.success(f"✅ API is online — v{health.get('version', '?')}, "
                   f"uptime {health.get('uptime_seconds', 0):.0f}s")
    else:
        st.warning("⚠️ API is offline. Start the backend first: "
                   "`uvicorn cloud_system.api.main:app --reload`")


# ═══════════════════════════════════════════════
# PAGE: MRI Prediction
# ═══════════════════════════════════════════════
elif page == "🧠 MRI Prediction":
    st.title("🧠 Structural MRI Prediction")
    st.markdown("Upload a **NIfTI (.nii / .nii.gz)** structural MRI scan.")

    uploaded = st.file_uploader("Choose MRI file", type=["nii", "gz", "nii.gz"])
    use_trained = st.checkbox("Use trained optimal threshold", value=True,
                              help="Uses the AUC-optimised threshold from training")
    if not use_trained:
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
    else:
        threshold = None

    if uploaded and st.button("🔍 Run Prediction", type="primary"):
        with st.spinner("Processing MRI scan..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            data = {"threshold": str(threshold)} if threshold is not None else {}
            try:
                r = requests.post(f"{API_BASE}/api/v1/predict/mri",
                                  files=files, data=data, timeout=120)
                if r.ok:
                    show_result(r.json(), "MRI – 3D-CNN Result")
                else:
                    st.error(f"API error: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════
# PAGE: fMRI Prediction
# ═══════════════════════════════════════════════
elif page == "📊 fMRI Prediction":
    st.title("📊 Functional MRI Prediction")
    st.markdown("Upload a **NIfTI (.nii / .nii.gz)** functional MRI scan.")

    uploaded = st.file_uploader("Choose fMRI file", type=["nii", "gz", "nii.gz"])
    use_trained = st.checkbox("Use trained optimal threshold", value=True,
                              help="Uses the AUC-optimised threshold from training")
    if not use_trained:
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
    else:
        threshold = None

    if uploaded and st.button("🔍 Run Prediction", type="primary"):
        with st.spinner("Processing fMRI scan..."):
            files = {"file": (uploaded.name, uploaded.getvalue())}
            data = {"threshold": str(threshold)} if threshold is not None else {}
            try:
                r = requests.post(f"{API_BASE}/api/v1/predict/fmri",
                                  files=files, data=data, timeout=120)
                if r.ok:
                    show_result(r.json(), "fMRI – CNN-LSTM Result")
                else:
                    st.error(f"API error: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════
# PAGE: Multimodal Fusion
# ═══════════════════════════════════════════════
elif page == "🔬 Multimodal Fusion":
    st.title("🔬 Multimodal Fusion Prediction")
    st.markdown("Upload **both** MRI and fMRI scans for combined diagnosis.")

    col1, col2 = st.columns(2)
    with col1:
        mri_file = st.file_uploader("Structural MRI (.nii/.nii.gz)",
                                     type=["nii", "gz"], key="mri")
    with col2:
        fmri_file = st.file_uploader("Functional MRI (.nii/.nii.gz)",
                                      type=["nii", "gz"], key="fmri")

    fusion_type = st.selectbox("Fusion method",
                                ["hm_fusion", "agentic_fusion"])
    use_trained = st.checkbox("Use trained optimal thresholds", value=True,
                              help="Uses per-model AUC-optimised thresholds")
    if not use_trained:
        threshold = st.slider("Classification threshold", 0.0, 1.0, 0.5, 0.05)
    else:
        threshold = None

    if mri_file and fmri_file and st.button("🔬 Run Fusion", type="primary"):
        with st.spinner("Running multimodal fusion..."):
            files = {
                "mri_file": (mri_file.name, mri_file.getvalue()),
                "fmri_file": (fmri_file.name, fmri_file.getvalue()),
            }
            data = {"model_type": fusion_type}
            if threshold is not None:
                data["threshold"] = str(threshold)
            try:
                r = requests.post(f"{API_BASE}/api/v1/predict/multimodal",
                                  files=files, data=data, timeout=180)
                if r.ok:
                    result = r.json()
                    show_result(result, "Fusion Result")

                    st.markdown("---")
                    st.markdown("### Individual Model Results")
                    c1, c2 = st.columns(2)
                    with c1:
                        show_result(result["mri_result"], "MRI Component")
                    with c2:
                        show_result(result["fmri_result"], "fMRI Component")

                    st.markdown("### Fusion Weights")
                    weights = result.get("fusion_weights", {})
                    st.bar_chart(weights)
                else:
                    st.error(f"API error: {r.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")


# ═══════════════════════════════════════════════
# PAGE: System Status
# ═══════════════════════════════════════════════
elif page == "⚙️ System Status":
    st.title("⚙️ System Status")

    health = api_health()
    if health:
        st.success("API is online")

        col1, col2, col3 = st.columns(3)
        col1.metric("Version", health.get("version", "?"))
        col2.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
        col3.metric("GPU", "Yes ✅" if health.get("gpu_available") else "No (CPU)")

        st.markdown("### Loaded Models")
        models = health.get("models_loaded", {})
        for name, loaded in models.items():
            icon = "✅" if loaded else "❌"
            st.markdown(f"- {icon} **{name}**")
    else:
        st.error("❌ Cannot reach the API backend.")
        st.markdown(
            """
            **Start the backend:**
            ```bash
            cd cloud_system
            uvicorn cloud_system.api.main:app --host 0.0.0.0 --port 8000 --reload
            ```
            """
        )

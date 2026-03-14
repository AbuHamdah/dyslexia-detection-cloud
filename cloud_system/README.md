# Dyslexia Detection Cloud System

A cloud-deployable system for **agentic AI-based dyslexia detection** from structural and functional MRI neuroimaging data.

## Architecture

```
┌─────────────┐      ┌──────────────────┐      ┌──────────────┐
│   Browser    │◄────►│  NGINX (port 80) │◄────►│  Streamlit   │
│   (Client)   │      │  reverse proxy   │      │  Dashboard   │
└─────────────┘      └───────┬──────────┘      │  (port 8501) │
                             │                  └──────────────┘
                             ▼
                     ┌──────────────────┐
                     │   FastAPI REST   │
                     │   (port 8000)    │
                     │  ┌────────────┐  │
                     │  │ 3D-CNN     │  │  ← Structural MRI
                     │  │ CNN-LSTM   │  │  ← Functional MRI
                     │  │ HM Fusion  │  │  ← Weighted Voting
                     │  │ Agentic    │  │  ← Trainable Head
                     │  └────────────┘  │
                     └──────────────────┘
```

## Models

| Model | Architecture | Input | Task |
|-------|-------------|-------|------|
| **3D-CNN** | Conv3D (16→32→64) → GAP → Dense(32) | Structural MRI `(10,128,128,1)` | Binary classification |
| **CNN-LSTM** | TD-Conv2D (32→64→128) + BiLSTM (64→32) | Functional MRI `(64,64,3,30)` | Binary classification |
| **HM Fusion** | α·MRI + β·fMRI weighted voting | Both modalities | Ensemble |
| **Agentic Fusion** | Feature concat → Dense (128→64→32) | Both features | Trainable head |

All models are trained with an **agentic LLM-guided optimization** loop that uses an AI agent to decide training actions (augment, reduce LR, adjust dropout, etc.) based on validation metrics.

## Quick Start (Localhost)

### 1. Install dependencies
```bash
cd cloud_system
pip install -r requirements.txt
```

### 2. Generate demo models
```bash
python -m cloud_system.training.save_demo_models
```

### 3. Start the API server
```bash
uvicorn cloud_system.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Start the dashboard (new terminal)
```bash
streamlit run cloud_system/dashboard/app.py --server.port 8501
```

### 5. Open in browser
- **Dashboard**: http://localhost:8501
- **API docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

## Docker Deployment

```bash
cd cloud_system/docker
docker-compose up --build
```

Then visit http://localhost (NGINX routes to dashboard + API).

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check + model status |
| `POST` | `/api/v1/predict/mri` | Predict from structural MRI |
| `POST` | `/api/v1/predict/fmri` | Predict from functional MRI |
| `POST` | `/api/v1/predict/multimodal` | Fusion prediction (both) |

## Training (with real data)

```bash
# Train 3D-CNN on structural MRI
python -m cloud_system.training.train_mri --data_dir /path/to/mri_data

# Train CNN-LSTM on functional MRI
python -m cloud_system.training.train_fmri --data_dir /path/to/fmri_data

# Train fusion model (requires pretrained base models)
python -m cloud_system.training.train_fusion --mri_data /path/to/mri --fmri_data /path/to/fmri
```

## Project Structure

```
cloud_system/
├── api/                    # FastAPI REST backend
│   ├── main.py             # App entry point + health endpoint
│   ├── routes.py           # Prediction endpoints
│   ├── inference.py        # Model loading + prediction service
│   └── schemas.py          # Pydantic request/response models
├── config/
│   └── settings.py         # Environment-based configuration
├── dashboard/
│   └── app.py              # Streamlit clinical dashboard
├── docker/
│   ├── Dockerfile          # API container
│   ├── Dockerfile.dashboard# Dashboard container
│   ├── docker-compose.yml  # Multi-service orchestration
│   └── nginx.conf          # Reverse proxy config
├── models/
│   ├── cnn3d.py            # 3D-CNN architecture
│   ├── cnn_lstm.py         # CNN-LSTM architecture
│   ├── fusion.py           # HM + Agentic fusion
│   └── agentic_optimizer.py# Agent memory + LLM agent
├── preprocessing/
│   ├── mri_pipeline.py     # Structural MRI preprocessing
│   └── fmri_pipeline.py    # Functional MRI preprocessing
├── training/
│   ├── train_mri.py        # MRI training with agentic loop
│   ├── train_fmri.py       # fMRI training with agentic loop
│   ├── train_fusion.py     # Fusion training
│   └── save_demo_models.py # Generate placeholder models
├── tests/
│   └── test_smoke.py       # Smoke tests
├── saved_models/           # Trained model files (.keras)
├── logs/                   # Agent logs
├── requirements.txt
├── .env.example
└── README.md
```

## Cloud Contribution

This system demonstrates a **Service-Oriented Architecture (SOA)** deployed via Docker containers:
- **Containerised microservices** (API + Dashboard + Proxy)
- **REST API** with standardised JSON responses
- **Horizontal scalability** via Docker Compose replicas
- **Environment-based configuration** (no hardcoded paths)
- **Cloud-agnostic** — runs on AWS, GCP, Azure, or localhost

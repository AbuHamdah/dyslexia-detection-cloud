"""
Configuration settings for the Dyslexia Detection Cloud System.
All values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── App ──
    APP_NAME: str = "Dyslexia Detection System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    # ── Paths ──
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "saved_models"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    LOG_DIR: Path = BASE_DIR / "logs"

    # ── Saved model file names ──
    MRI_MODEL_FILE: str = "3dcnn_agentic.keras"
    FMRI_MODEL_FILE: str = "cnn_lstm.keras"
    FUSION_HM_MODEL_FILE: str = "hm_3dcnn_lstm.keras"
    FUSION_AGENTIC_MODEL_FILE: str = "agentic_3dcnn_lstm.keras"

    # ── MRI preprocessing (from codes/mri.py Config) ──
    MRI_SHAPE: tuple = (10, 128, 128, 1)    # 10 axial slices, 128x128, 1 channel
    MRI_SLICE_COUNT: int = 10
    MRI_SLICE_SIZE: int = 128

    # ── fMRI preprocessing (from codes/fmri.py Config) ──
    FMRI_SPATIAL_SHAPE: tuple = (64, 64, 3)  # H, W, 3 orthogonal slices
    FMRI_TIME_STEPS: int = 30                 # temporal frames

    # ── Fusion weights (from paper: reliability-weighted soft voting) ──
    FUSION_ALPHA: float = 0.489   # MRI weight
    FUSION_BETA: float = 0.511    # fMRI weight
    FUSION_THRESHOLD: float = 0.5

    # ── Training defaults ──
    BATCH_SIZE: int = 8
    EPOCHS: int = 50
    LEARNING_RATE: float = 5e-4
    MAX_ITERATIONS: int = 10

    # ── MRI training (from codes/mri.py) ──
    MRI_BATCH_SIZE: int = 8
    MRI_EPOCHS: int = 50
    MRI_LR: float = 5e-4

    # ── fMRI training (from codes/fmri.py) ──
    FMRI_BATCH_SIZE: int = 4
    FMRI_EPOCHS: int = 100
    FMRI_LR: float = 5e-4

    # ── API ──
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["*"]

    # ── OpenAI (optional — for agentic training) ──
    OPENAI_API_KEY: str = ""

    # ── GPU ──
    TF_FORCE_GPU_ALLOW_GROWTH: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

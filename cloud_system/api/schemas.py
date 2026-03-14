"""
Pydantic schemas for API request / response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class PredictionResult(BaseModel):
    """Single-model prediction response."""
    label: str = Field(..., description="'dyslexic' or 'control'")
    confidence: float = Field(..., ge=0, le=1)
    probability: float = Field(..., ge=0, le=1)
    threshold: float = Field(default=0.5)
    model_name: str = ""
    processing_time_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class MultimodalResult(BaseModel):
    """Fusion prediction response."""
    fusion_label: str
    fusion_confidence: float
    fusion_weights: Dict[str, float]
    mri_result: PredictionResult
    fmri_result: PredictionResult
    model_type: str = "hm_fusion"
    processing_time_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "1.0.0"
    gpu_available: bool = False
    uptime_seconds: float = 0.0
    models_loaded: Dict[str, bool] = {}

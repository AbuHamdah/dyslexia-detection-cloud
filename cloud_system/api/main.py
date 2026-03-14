"""
FastAPI Application — Dyslexia Detection Cloud System.

Starts the API server, loads models on startup, serves health check.
"""

import time
import tensorflow as tf
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cloud_system.config.settings import settings
from cloud_system.api.routes import router
from cloud_system.api.inference import inference_service

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models when the server starts."""
    print(f"[API] Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    inference_service.load_models()
    loaded = inference_service.get_loaded_status()
    for name, ok in loaded.items():
        icon = "[OK]" if ok else "[!!]"
        print(f"  {icon} {name}: {'loaded' if ok else 'not found'}")
    yield
    print("[API] Shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS — allow dashboard to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount prediction routes
app.include_router(router)


@app.get("/health")
def health():
    """Health check endpoint."""
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "gpu_available": gpu_available,
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "models_loaded": inference_service.get_loaded_status(),
    }

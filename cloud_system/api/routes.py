"""
FastAPI REST routes for prediction endpoints.

POST /api/v1/predict/mri        — single MRI prediction
POST /api/v1/predict/fmri       — single fMRI prediction
POST /api/v1/predict/multimodal — multimodal fusion prediction
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from cloud_system.api.inference import inference_service

router = APIRouter(prefix="/api/v1")


@router.post("/predict/mri")
async def predict_mri(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
):
    """Run MRI prediction (3D-CNN)."""
    try:
        contents = await file.read()
        result = inference_service.predict_mri(contents, file.filename, threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/fmri")
async def predict_fmri(
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
):
    """Run fMRI prediction (CNN-LSTM)."""
    try:
        contents = await file.read()
        result = inference_service.predict_fmri(contents, file.filename, threshold)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/multimodal")
async def predict_multimodal(
    mri_file: UploadFile = File(...),
    fmri_file: UploadFile = File(...),
    model_type: str = Form("hm_fusion"),
    threshold: Optional[float] = Form(None),
):
    """Run multimodal fusion prediction."""
    try:
        mri_bytes = await mri_file.read()
        fmri_bytes = await fmri_file.read()
        result = inference_service.predict_multimodal(
            mri_bytes, mri_file.filename,
            fmri_bytes, fmri_file.filename,
            model_type, threshold,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Inference Service — loads models and runs predictions.
Connects preprocessing pipelines → model → result.
Loads optimal thresholds from training if available.
"""

import json
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Dict

from cloud_system.config.settings import settings
from cloud_system.preprocessing.mri_pipeline import preprocess_mri_bytes
from cloud_system.preprocessing.fmri_pipeline import preprocess_fmri_bytes
from cloud_system.models.fusion import HMFusion


class InferenceService:
    """Singleton service that loads models and runs predictions."""

    def __init__(self):
        self.models: Dict[str, tf.keras.Model] = {}
        self.thresholds: Dict[str, float] = {}  # per-model optimal thresholds
        self.calibration: Dict[str, Dict] = {}  # Platt scaling params
        self.hm_fusion = HMFusion(
            alpha=settings.FUSION_ALPHA,
            beta=settings.FUSION_BETA,
            threshold=settings.FUSION_THRESHOLD,
        )
        self._loaded = False

    # ── model loading ──

    def _load_threshold(self, name: str, threshold_file: str):
        """Load optimal threshold and Platt calibration from a JSON file."""
        path = settings.MODEL_DIR / threshold_file
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.thresholds[name] = float(data.get("threshold", 0.5))
                # Load Platt scaling parameters if present
                if "platt_a" in data and "platt_b" in data:
                    self.calibration[name] = {
                        "a": float(data["platt_a"]),
                        "b": float(data["platt_b"]),
                    }
                    print(f"[Inference] Loaded threshold + Platt calibration for "
                          f"{name}: thresh={self.thresholds[name]:.4f}, "
                          f"a={self.calibration[name]['a']:.4f}, "
                          f"b={self.calibration[name]['b']:.4f}")
                else:
                    print(f"[Inference] Loaded threshold for {name}: "
                          f"{self.thresholds[name]:.4f}")
            except Exception as e:
                print(f"[Inference] Could not load threshold for {name}: {e}")
                self.thresholds[name] = 0.5
        else:
            self.thresholds[name] = 0.5

    def load_models(self):
        """Load all saved models from disk."""
        model_dir = settings.MODEL_DIR

        model_map = {
            "3dcnn_agentic": model_dir / settings.MRI_MODEL_FILE,
            "cnn_lstm": model_dir / settings.FMRI_MODEL_FILE,
            "hm_3dcnn_lstm": model_dir / settings.FUSION_HM_MODEL_FILE,
            "agentic_3dcnn_lstm": model_dir / settings.FUSION_AGENTIC_MODEL_FILE,
        }

        for name, path in model_map.items():
            if path.exists():
                try:
                    self.models[name] = tf.keras.models.load_model(
                        str(path), compile=False
                    )
                    print(f"[Inference] Loaded {name} from {path}")
                except Exception as e:
                    print(f"[Inference] Failed to load {name}: {e}")
            else:
                print(f"[Inference] Model file not found: {path}")

        # Load per-model thresholds
        self._load_threshold("3dcnn_agentic", "mri_threshold.json")
        self._load_threshold("cnn_lstm", "fmri_threshold.json")
        self._load_threshold("hm_3dcnn_lstm", "fusion_hm_threshold.json")
        self._load_threshold("agentic_3dcnn_lstm", "fusion_agentic_threshold.json")

        self._loaded = True

    def get_loaded_status(self) -> Dict[str, bool]:
        return {
            "3dcnn_agentic": "3dcnn_agentic" in self.models,
            "cnn_lstm": "cnn_lstm" in self.models,
            "hm_3dcnn_lstm": "hm_3dcnn_lstm" in self.models,
            "agentic_3dcnn_lstm": "agentic_3dcnn_lstm" in self.models,
        }

    # ── single prediction helpers ──

    def _predict_single(self, model_name: str, tensor: np.ndarray) -> Dict:
        """Run a single model and return probability."""
        model = self.models.get(model_name)
        if model is None:
            raise RuntimeError(f"Model '{model_name}' not loaded")

        x = np.expand_dims(tensor, axis=0)  # add batch dim
        raw = model.predict(x, verbose=0)

        # Handle both sigmoid (shape [1,1]) and softmax (shape [1,2])
        if raw.shape[-1] == 1:
            prob = float(raw.flatten()[0])
        else:
            prob = float(raw[0, 1])  # class-1 probability

        return {"probability": prob}

    def _calibrate(self, model_name: str, prob: float) -> float:
        """Apply Platt scaling calibration if available."""
        cal = self.calibration.get(model_name)
        if cal is not None:
            # Platt scaling: sigmoid(a * x + b)
            logit = cal["a"] * prob + cal["b"]
            return float(1 / (1 + np.exp(-logit)))
        return prob

    def _predict_mri_tta(self, tensor: np.ndarray) -> float:
        """
        MRI prediction with test-time augmentation (TTA).
        Averages predictions over geometric variants for higher confidence.
        """
        model = self.models.get("3dcnn_agentic")
        if model is None:
            raise RuntimeError("Model '3dcnn_agentic' not loaded")

        x = np.expand_dims(tensor, axis=0)
        preds = []

        # Original
        raw = model.predict(x, verbose=0)
        p = float(raw.flatten()[0]) if raw.shape[-1] == 1 else float(raw[0, 1])
        preds.append(p)

        # Horizontal flip
        x_hflip = x[:, :, :, ::-1, :]
        raw = model.predict(x_hflip, verbose=0)
        p = float(raw.flatten()[0]) if raw.shape[-1] == 1 else float(raw[0, 1])
        preds.append(p)

        # Vertical flip
        x_vflip = x[:, :, ::-1, :, :]
        raw = model.predict(x_vflip, verbose=0)
        p = float(raw.flatten()[0]) if raw.shape[-1] == 1 else float(raw[0, 1])
        preds.append(p)

        raw_prob = float(np.mean(preds))
        return raw_prob

    # ── public API ──

    def predict_mri(self, file_bytes: bytes, filename: str,
                    threshold: float = None) -> Dict:
        """Full MRI prediction pipeline with TTA. Uses trained threshold."""
        t0 = time.time()

        # Use model-specific threshold if none provided
        if threshold is None:
            threshold = self.thresholds.get("3dcnn_agentic", 0.5)

        tensor = preprocess_mri_bytes(file_bytes, filename)
        if tensor is None:
            raise ValueError("Failed to preprocess MRI file")

        # Use TTA for MRI predictions (3 augmentations averaged)
        prob = self._predict_mri_tta(tensor)
        label = "dyslexic" if prob >= threshold else "control"
        confidence = prob if label == "dyslexic" else 1 - prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "threshold": threshold,
            "model_name": "3dcnn_agentic",
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def predict_fmri(self, file_bytes: bytes, filename: str,
                     threshold: float = None) -> Dict:
        """Full fMRI prediction pipeline. Uses trained threshold if available."""
        t0 = time.time()

        # Use model-specific threshold if none provided
        if threshold is None:
            threshold = self.thresholds.get("cnn_lstm", 0.5)

        tensor = preprocess_fmri_bytes(file_bytes, filename)
        if tensor is None:
            raise ValueError("Failed to preprocess fMRI file")

        result = self._predict_single("cnn_lstm", tensor)
        prob = result["probability"]
        label = "dyslexic" if prob >= threshold else "control"
        confidence = prob if label == "dyslexic" else 1 - prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "probability": round(prob, 4),
            "threshold": threshold,
            "model_name": "cnn_lstm",
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }

    def predict_multimodal(self, mri_bytes: bytes, mri_name: str,
                           fmri_bytes: bytes, fmri_name: str,
                           model_type: str = "hm_fusion",
                           threshold: float = None) -> Dict:
        """Full multimodal fusion prediction pipeline."""
        t0 = time.time()

        mri_result = self.predict_mri(mri_bytes, mri_name, None)
        fmri_result = self.predict_fmri(fmri_bytes, fmri_name, None)

        mri_prob = mri_result["probability"]
        fmri_prob = fmri_result["probability"]

        # Use HM weighted voting (works for both model_type options)
        fusion = self.hm_fusion.predict(mri_prob, fmri_prob)
        fusion_prob = fusion["fusion_probability"]

        # Use fusion-specific trained threshold if none provided
        if threshold is None:
            threshold = self.thresholds.get("hm_3dcnn_lstm", 0.5)

        fusion_label = "dyslexic" if fusion_prob >= threshold else "control"
        fusion_conf = fusion_prob if fusion_label == "dyslexic" else 1 - fusion_prob

        return {
            "fusion_label": fusion_label,
            "fusion_confidence": round(fusion_conf, 4),
            "fusion_weights": {
                "alpha_mri": self.hm_fusion.alpha,
                "beta_fmri": self.hm_fusion.beta,
            },
            "mri_result": mri_result,
            "fmri_result": fmri_result,
            "model_type": model_type,
            "processing_time_ms": round((time.time() - t0) * 1000, 1),
        }


# Global singleton
inference_service = InferenceService()

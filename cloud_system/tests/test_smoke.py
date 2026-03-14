"""
Smoke tests — validate that modules import and models build correctly.

Usage:
    python -m pytest cloud_system/tests/test_smoke.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
import numpy as np


class TestModelBuilds:
    """Test that each model can be constructed."""

    def test_build_3dcnn(self):
        from cloud_system.models.cnn3d import build_3dcnn
        model = build_3dcnn((10, 128, 128, 1))
        assert model.output_shape == (None, 1)
        # forward pass
        dummy = np.random.rand(1, 10, 128, 128, 1).astype("float32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (1, 1)
        assert 0 <= out[0, 0] <= 1

    def test_build_cnn_lstm(self):
        from cloud_system.models.cnn_lstm import build_cnn_lstm
        model = build_cnn_lstm((64, 64, 3), 30)
        assert model.output_shape == (None, 1)
        dummy = np.random.rand(1, 64, 64, 3, 30).astype("float32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (1, 1)

    def test_build_agentic_fusion(self):
        from cloud_system.models.fusion import build_agentic_fusion
        model = build_agentic_fusion(32, 32)
        assert model.output_shape == (None, 1)
        dummy = np.random.rand(1, 64).astype("float32")
        out = model.predict(dummy, verbose=0)
        assert out.shape == (1, 1)


class TestHMFusion:
    """Test HM weighted voting."""

    def test_default_weights(self):
        from cloud_system.models.fusion import HMFusion
        hm = HMFusion(alpha=0.489, beta=0.511, threshold=0.5)
        result = hm.predict(0.8, 0.9)
        expected = 0.489 * 0.8 + 0.511 * 0.9
        assert abs(result["fusion_probability"] - expected) < 1e-4
        assert result["fusion_label"] == "dyslexic"

    def test_control_prediction(self):
        from cloud_system.models.fusion import HMFusion
        hm = HMFusion(alpha=0.489, beta=0.511, threshold=0.5)
        result = hm.predict(0.1, 0.2)
        assert result["fusion_label"] == "control"


class TestConfig:
    """Test config loads."""

    def test_settings_load(self):
        from cloud_system.config.settings import settings
        assert settings.APP_NAME is not None
        assert settings.MRI_SHAPE == (10, 128, 128, 1)
        assert settings.FMRI_TIME_STEPS == 30


class TestFastAPI:
    """Test API app can be created."""

    def test_app_creation(self):
        from cloud_system.api.main import app
        assert app.title is not None

    def test_health_route_exists(self):
        from cloud_system.api.main import app
        routes = [r.path for r in app.routes]
        assert "/health" in routes

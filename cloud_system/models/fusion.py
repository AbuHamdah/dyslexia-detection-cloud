"""
Multimodal Fusion Models for Dyslexia Detection.
Architectures extracted from codes/fusion.py.

Two strategies:
1. HM Fusion  — Reliability-weighted soft voting (α·MRI + β·fMRI)
2. Agentic Fusion — Feature-level concatenation with trainable fusion head
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional


# ════════════════════════════════════════════════════════════════════
# 1) HM 3D-CNN-LSTM  — Weighted Soft Voting (no trainable fusion)
# ════════════════════════════════════════════════════════════════════

class HMFusion:
    """
    Reliability-weighted soft voting from the paper.
    α = 0.489 (MRI weight), β = 0.511 (fMRI weight).
    No trainable parameters — just weighted combination of probabilities.
    """

    def __init__(self, alpha: float = 0.489, beta: float = 0.511,
                 threshold: float = 0.5):
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def predict(self, mri_prob: float, fmri_prob: float) -> Dict:
        """
        Combine MRI and fMRI probabilities.
        Returns fusion probability and label.
        """
        fusion_prob = self.alpha * mri_prob + self.beta * fmri_prob
        label = "dyslexic" if fusion_prob >= self.threshold else "control"
        return {
            "fusion_probability": float(fusion_prob),
            "label": label,
            "mri_prob": float(mri_prob),
            "fmri_prob": float(fmri_prob),
            "alpha": self.alpha,
            "beta": self.beta,
        }


# ════════════════════════════════════════════════════════════════════
# 2) Agentic Fusion — Feature-level concat with trainable fusion head
# ════════════════════════════════════════════════════════════════════

def create_feature_extractor(model: keras.Model,
                             name: str = "features") -> keras.Model:
    """
    Create feature extractor from a trained model.
    Removes final Dense(1, sigmoid) layer to expose the feature vector.
    From codes/fusion.py create_feature_extractor().
    """
    # Find the last Dense layer before the output sigmoid
    feature_layer = None
    for layer in model.layers[:-1]:
        if isinstance(layer, layers.Dense):
            feature_layer = layer

    if feature_layer is None:
        feature_layer = model.layers[-2]

    extractor = keras.Model(
        inputs=model.input,
        outputs=feature_layer.output,
        name=name,
    )

    # Freeze all layers
    for layer in extractor.layers:
        layer.trainable = False

    return extractor


def build_agentic_fusion(mri_extractor: keras.Model,
                         fmri_extractor: keras.Model,
                         mri_shape: tuple = (10, 128, 128, 1),
                         fmri_shape: tuple = (64, 64, 3, 30),
                         dropout: float = 0.5,
                         lr: float = 1e-3) -> keras.Model:
    """
    Build Agentic Fusion model from codes/fusion.py build_fusion_model().

    Architecture:
        Frozen MRI features ──┐
                               ├── Concatenate
        Frozen fMRI features ─┘
             ↓
        Dense(128) → Dropout → BN
        Dense(64)  → Dropout
        Dense(32)  → Dropout
        Dense(1, sigmoid)
    """
    mri_input = keras.Input(shape=mri_shape, name="mri_input")
    fmri_input = keras.Input(shape=fmri_shape, name="fmri_input")

    mri_features = mri_extractor(mri_input)
    fmri_features = fmri_extractor(fmri_input)

    fused = layers.Concatenate(name="fusion")([mri_features, fmri_features])

    x = layers.Dense(128, activation="relu", name="fusion_dense1",
                     kernel_regularizer=keras.regularizers.l2(0.01))(fused)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu", name="fusion_dense2",
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout * 0.6)(x)

    x = layers.Dense(32, activation="relu", name="fusion_dense3",
                     kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout * 0.4)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(
        inputs=[mri_input, fmri_input],
        outputs=outputs,
        name="agentic_fusion",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model

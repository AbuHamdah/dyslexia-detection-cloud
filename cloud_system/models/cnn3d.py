"""
3D-CNN Model for Structural MRI Dyslexia Detection.
Architecture extracted from codes/mri.py build_model().

Input:  (10, 128, 128, 1)  -- 10 axial slices, 128x128, grayscale
Output: (1,)               -- sigmoid probability of dyslexia
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional


def build_3dcnn(input_shape: tuple = (10, 128, 128, 1),
                params: Optional[Dict] = None) -> keras.Model:
    """
    Build the Agentic 3D-CNN (binary, sigmoid output).

    Architecture:
        Conv3D(16) -> BN -> MaxPool3D -> SpatialDropout3D
        Conv3D(32) -> BN -> MaxPool3D -> SpatialDropout3D
        Conv3D(64) -> BN -> GlobalAveragePooling3D
        Dense(64) -> Dropout -> Dense(32) -> Dropout -> Dense(1, sigmoid)
    """
    params = params or {}
    lr = params.get("learning_rate", 5e-4)
    dropout = params.get("dropout", 0.5)
    l2_reg = params.get("l2_reg", 0.01)

    inputs = keras.Input(shape=input_shape, name="mri_input")

    # Conv block 1
    x = layers.Conv3D(16, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(0.2)(x)

    # Conv block 2
    x = layers.Conv3D(32, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(0.25)(x)

    # Conv block 3
    x = layers.Conv3D(64, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    # Dense head
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="agentic_3dcnn")
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def build_3dcnn_softmax(input_shape: tuple = (10, 128, 128, 1),
                        params: Optional[Dict] = None) -> keras.Model:
    """
    Baseline 3D-CNN with 2-class softmax output (for HM fusion voting).
    Same conv backbone, different head.
    """
    params = params or {}
    lr = params.get("learning_rate", 5e-4)
    dropout = params.get("dropout", 0.5)
    l2_reg = params.get("l2_reg", 0.01)

    inputs = keras.Input(shape=input_shape, name="mri_input")

    x = layers.Conv3D(16, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(0.2)(x)

    x = layers.Conv3D(32, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(0.25)(x)

    x = layers.Conv3D(64, 3, activation="relu", padding="same",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(2, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="baseline_3dcnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

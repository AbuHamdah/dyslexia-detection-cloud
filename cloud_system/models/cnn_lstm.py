"""
CNN-LSTM Model for Functional MRI Dyslexia Detection.
Architecture extracted from codes/fmri.py build_model().

Input:  (64, 64, 3, 30)  — 64×64 pixels, 3 orthogonal slices, 30 timeframes
Output: (1,)             — sigmoid probability of dyslexia
"""

from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Optional


def build_cnn_lstm(input_shape: tuple = (64, 64, 3, 30),
                   params: Optional[Dict] = None) -> keras.Model:
    """
    Build the Agentic CNN-LSTM (binary, sigmoid output).

    Architecture from codes/fmri.py:
        Permute → (time_steps, H, W, channels)
        TD-Conv2D(32) → BN → MaxPool → Dropout
        TD-Conv2D(64) → BN → MaxPool → Dropout
        TD-Conv2D(128) → BN → GAP
        BiLSTM(64) → BN → Dropout
        BiLSTM(32) → BN
        Dense(64) → Dropout → Dense(32) → Dropout → Dense(1, sigmoid)
    """
    params = params or {}
    lr = params.get("learning_rate", 5e-4)
    dropout = params.get("dropout", 0.5)
    l2_reg = params.get("l2_reg", 0.01)

    inputs = keras.Input(shape=input_shape, name="fmri_input")

    # Permute to (time_steps, H, W, channels)
    x = layers.Permute((4, 1, 2, 3))(inputs)

    # TD-CNN Block 1
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.3))(x)

    # TD-CNN Block 2
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.4))(x)

    # TD-CNN Block 3
    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # Bidirectional LSTM
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    # Classification head
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs, outputs, name="agentic_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def build_cnn_lstm_softmax(input_shape: tuple = (64, 64, 3, 30),
                           params: Optional[Dict] = None) -> keras.Model:
    """Baseline CNN-LSTM with 2-class softmax (for HM fusion voting)."""
    params = params or {}
    lr = params.get("learning_rate", 5e-4)
    dropout = params.get("dropout", 0.5)
    l2_reg = params.get("l2_reg", 0.01)

    inputs = keras.Input(shape=input_shape, name="fmri_input")
    x = layers.Permute((4, 1, 2, 3))(inputs)

    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.3))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.4))(x)

    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), padding="same", activation="relu",
                      kernel_regularizer=keras.regularizers.l2(l2_reg)))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout * 0.5)(x)
    outputs = layers.Dense(2, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="baseline_cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

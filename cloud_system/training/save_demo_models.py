"""
Save demo models trained on brain-realistic synthetic data.

Both classes have brain-like structure (dark background + bright oval brain).
The difference is INSIDE the brain region:
  - class 0 (control):   symmetric brain, smooth texture
  - class 1 (dyslexic):  asymmetric brain, left-right intensity difference

This ensures real MRI scans (which also have dark background + bright brain)
fall near the decision boundary and produce varied, realistic predictions.

Usage:
    python -m cloud_system.training.save_demo_models
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import tensorflow as tf
from cloud_system.config.settings import settings
from cloud_system.models.cnn3d import build_3dcnn
from cloud_system.models.cnn_lstm import build_cnn_lstm
from cloud_system.models.fusion import create_feature_extractor, build_agentic_fusion

N_SAMPLES = 60          # 30 per class
TRAIN_EPOCHS = 20
BATCH_SIZE = 8


def _brain_mask_2d(h, w):
    """Create an elliptical brain-like mask (True inside the brain)."""
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    ry, rx = h * 0.38, w * 0.35
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    return mask


def make_synthetic_3d(shape, n=N_SAMPLES):
    """
    Create brain-realistic synthetic MRI volumes.
    Both classes have: dark background (~0) + bright brain region (~0.4-0.8).
    Difference:
      - Control (0):  symmetric, uniform brain intensity
      - Dyslexic (1): left-right asymmetry + higher variance in brain
    """
    slices, h, w, c = shape  # (10, 128, 128, 1)
    mask = _brain_mask_2d(h, w)  # (128, 128) boolean
    half = n // 2
    X, y = [], []

    # ── Controls: symmetric, smooth brain ──
    for _ in range(half):
        vol = np.zeros(shape, dtype="float32")
        base_intensity = np.random.uniform(0.4, 0.7)
        for s in range(slices):
            # Slice-dependent intensity (dimmer at edges, brighter in middle)
            depth_factor = 1.0 - 0.3 * abs(s - slices / 2) / (slices / 2)
            brain = np.random.normal(base_intensity * depth_factor, 0.05, (h, w))
            brain = np.clip(brain, 0, 1)
            vol[s, :, :, 0] = brain * mask  # background stays 0
        # Add small global noise
        vol += np.random.normal(0, 0.02, shape).astype("float32")
        vol = np.clip(vol, 0, 1)
        X.append(vol)
        y.append(0)

    # ── Dyslexic: asymmetric brain, left-right difference ──
    for _ in range(half):
        vol = np.zeros(shape, dtype="float32")
        base_intensity = np.random.uniform(0.4, 0.7)
        asym_strength = np.random.uniform(0.12, 0.25)  # left-right difference
        for s in range(slices):
            depth_factor = 1.0 - 0.3 * abs(s - slices / 2) / (slices / 2)
            brain = np.random.normal(base_intensity * depth_factor, 0.08, (h, w))
            # KEY DIFFERENCE: left hemisphere brighter than right
            brain[:, :w // 2] += asym_strength
            brain[:, w // 2:] -= asym_strength * 0.5
            # Add small irregular "hotspots" in left temporal region
            region_y = slice(h // 3, 2 * h // 3)
            region_x = slice(w // 6, w // 3)
            brain[region_y, region_x] += np.random.uniform(0.05, 0.15)
            brain = np.clip(brain, 0, 1)
            vol[s, :, :, 0] = brain * mask
        vol += np.random.normal(0, 0.02, shape).astype("float32")
        vol = np.clip(vol, 0, 1)
        X.append(vol)
        y.append(1)

    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(n)
    return X[idx], y[idx]


def make_synthetic_fmri(shape, n=N_SAMPLES):
    """
    Create brain-realistic synthetic fMRI volumes.
    Both classes have brain structure. Difference is in temporal dynamics:
      - Control (0):  stable temporal signal in brain
      - Dyslexic (1): fluctuating temporal signal + activation bursts
    """
    h, w, ch, t = shape  # (64, 64, 3, 30)
    mask = _brain_mask_2d(h, w)
    half = n // 2
    X, y = [], []

    # ── Controls: stable temporal brain signal ──
    for _ in range(half):
        vol = np.zeros(shape, dtype="float32")
        base = np.random.uniform(0.35, 0.6)
        for ti in range(t):
            for ci in range(ch):
                frame = np.random.normal(base, 0.04, (h, w))
                frame = np.clip(frame, 0, 1)
                vol[:, :, ci, ti] = frame * mask
        vol += np.random.normal(0, 0.015, shape).astype("float32")
        vol = np.clip(vol, 0, 1)
        X.append(vol)
        y.append(0)

    # ── Dyslexic: fluctuating signal with activation bursts ──
    for _ in range(half):
        vol = np.zeros(shape, dtype="float32")
        base = np.random.uniform(0.35, 0.6)
        # Create temporal oscillation pattern
        temporal_signal = np.sin(np.linspace(0, 4 * np.pi, t)) * 0.15
        for ti in range(t):
            for ci in range(ch):
                frame = np.random.normal(base + temporal_signal[ti], 0.07, (h, w))
                # Activation burst in left region during specific time windows
                if 8 <= ti <= 15 or 22 <= ti <= 27:
                    frame[:h // 2, :w // 3] += np.random.uniform(0.1, 0.2)
                frame = np.clip(frame, 0, 1)
                vol[:, :, ci, ti] = frame * mask
        vol += np.random.normal(0, 0.015, shape).astype("float32")
        vol = np.clip(vol, 0, 1)
        X.append(vol)
        y.append(1)

    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(n)
    return X[idx], y[idx]


def main():
    out = settings.MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)

    # ── 1) 3D-CNN (MRI) ──
    print("Training 3D-CNN on synthetic MRI data...")
    mri_shape = settings.MRI_SHAPE                          # (10,128,128,1)
    X_mri, y_mri = make_synthetic_3d(mri_shape)
    m1 = build_3dcnn(mri_shape)
    m1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m1.fit(X_mri, y_mri, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    p1 = out / settings.MRI_MODEL_FILE
    m1.save(str(p1))
    preds1 = m1.predict(X_mri[:4], verbose=0).flatten()
    print(f"✅ Saved {p1}  (sample preds: {np.round(preds1, 3)})\n")

    # ── 2) CNN-LSTM (fMRI) ──
    print("Training CNN-LSTM on synthetic fMRI data...")
    fmri_shape = settings.FMRI_SPATIAL_SHAPE + (settings.FMRI_TIME_STEPS,)  # (64,64,3,30)
    X_fmri, y_fmri = make_synthetic_fmri(fmri_shape)
    m2 = build_cnn_lstm(input_shape=fmri_shape)
    m2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    m2.fit(X_fmri, y_fmri, epochs=TRAIN_EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    p2 = out / settings.FMRI_MODEL_FILE
    m2.save(str(p2))
    preds2 = m2.predict(X_fmri[:4], verbose=0).flatten()
    print(f"✅ Saved {p2}  (sample preds: {np.round(preds2, 3)})\n")

    # ── 3) Agentic Fusion ──
    print("Training Agentic Fusion on synthetic features...")
    mri_feat = create_feature_extractor(m1, name="mri_features")
    fmri_feat = create_feature_extractor(m2, name="fmri_features")
    m3 = build_agentic_fusion(mri_feat, fmri_feat,
                               mri_shape=mri_shape,
                               fmri_shape=fmri_shape)
    # Train fusion end-to-end on the same synthetic data
    m3.fit([X_mri, X_fmri], y_mri, epochs=TRAIN_EPOCHS,
           batch_size=BATCH_SIZE, verbose=1)
    p3 = out / settings.FUSION_AGENTIC_MODEL_FILE
    m3.save(str(p3))
    preds3 = m3.predict([X_mri[:4], X_fmri[:4]], verbose=0).flatten()
    print(f"✅ Saved {p3}  (sample preds: {np.round(preds3, 3)})\n")

    # ── 4) HM Fusion (same architecture, separate weights) ──
    print("Training HM Fusion on synthetic features...")
    m4 = build_agentic_fusion(mri_feat, fmri_feat,
                               mri_shape=mri_shape,
                               fmri_shape=fmri_shape)
    m4.fit([X_mri, X_fmri], y_mri, epochs=TRAIN_EPOCHS,
           batch_size=BATCH_SIZE, verbose=1)
    p4 = out / settings.FUSION_HM_MODEL_FILE
    m4.save(str(p4))
    preds4 = m4.predict([X_mri[:4], X_fmri[:4]], verbose=0).flatten()
    print(f"✅ Saved {p4}  (sample preds: {np.round(preds4, 3)})\n")

    print(f"All demo models saved to {out}")
    print("Models are now trained and will produce varied predictions ≠ 50%.")


if __name__ == "__main__":
    main()

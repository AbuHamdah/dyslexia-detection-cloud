"""
Train pretrained fusion model (HM + Agentic) combining MRI & fMRI models.
Refactored from codes/fusion.py to use cloud_system modules.

Data loading: BIDS format — ds003126_raw has subjects with both anat and func.
Uses class_weight + optimal threshold saving.

Usage:
    python -m cloud_system.training.train_fusion --data_dir "E:/KNOWLEDGE_BASED_SYSTEMS (4)"
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from cloud_system.config.settings import settings
from cloud_system.models.fusion import (
    HMFusion, create_feature_extractor, build_agentic_fusion,
)
from cloud_system.preprocessing.mri_pipeline import preprocess_mri
from cloud_system.preprocessing.fmri_pipeline import preprocess_fmri
from cloud_system.models.agentic_optimizer import AgentMemory, LLMAgent


# ── Data loading (BIDS — subjects with BOTH anat + func) ──

def load_aligned(data_dir: Path):
    """
    Load subjects that have BOTH T1w MRI and BOLD fMRI from ds003126_raw.
    Returns X_mri, X_fmri, y (only subjects present in both modalities).
    """
    mri_vols, fmri_vols, labels = [], [], []

    ds1 = data_dir / "ds003126_raw"
    if not ds1.exists():
        print("[ERROR] ds003126_raw not found")
        return np.array([]), np.array([]), np.array([])

    tsv = ds1 / "participants.tsv"
    if not tsv.exists():
        print("[ERROR] participants.tsv not found")
        return np.array([]), np.array([]), np.array([])

    with open(tsv, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split("\t")
    group_idx = header.index("group") if "group" in header else -1

    for line in lines[1:]:
        parts = line.strip().split("\t")
        sub_id = parts[0]
        group = parts[group_idx] if group_idx >= 0 else "TD"
        if group not in ("DL", "TD"):
            group = "TD"  # SpD → control

        sub_path = ds1 / sub_id

        # Find T1w MRI file
        mri_file = None
        for ses in ["ses-1", "ses-2", ""]:
            ses_path = sub_path / ses if ses else sub_path
            anat = ses_path / "anat"
            if anat.exists():
                for f in sorted(os.listdir(anat)):
                    if f.endswith((".nii", ".nii.gz")) and "T1w" in f:
                        fp = str(anat / f)
                        if os.path.getsize(fp) > 0:
                            mri_file = fp
                            break
            if mri_file:
                break

        # Find BOLD fMRI file
        fmri_file = None
        for ses in ["ses-1", "ses-2", ""]:
            ses_path = sub_path / ses if ses else sub_path
            func = ses_path / "func"
            if func.exists():
                for f in sorted(os.listdir(func)):
                    if f.endswith((".nii", ".nii.gz")) and "bold" in f.lower():
                        fp = str(func / f)
                        if os.path.getsize(fp) > 0:
                            fmri_file = fp
                            break
            if fmri_file:
                break

        if mri_file and fmri_file:
            try:
                mv = preprocess_mri(mri_file)
                fv = preprocess_fmri(fmri_file)
                if mv is not None and fv is not None:
                    mri_vols.append(mv)
                    fmri_vols.append(fv)
                    labels.append(1 if group == "DL" else 0)
                    print(f"  ✓ {sub_id} ({group})")
            except Exception as e:
                print(f"  ✗ {sub_id}: {e}")

    X_mri = np.array(mri_vols) if mri_vols else np.zeros((0,) + settings.MRI_SHAPE)
    X_fmri = np.array(fmri_vols) if fmri_vols else np.zeros((0,) + settings.FMRI_SPATIAL_SHAPE + (settings.FMRI_TIME_STEPS,))
    y = np.array(labels)
    return X_mri, X_fmri, y


def train(args):
    data_dir = Path(args.data_dir)
    print(f"[Train Fusion] Loading aligned bimodal data from {data_dir}")
    X_mri, X_fmri, y = load_aligned(data_dir)
    print(f"  Aligned subjects: {len(y)}  (DL={int(y.sum())}, TD={int(len(y)-y.sum())})")

    if len(y) < 4:
        print("[ERROR] Not enough aligned subjects for fusion training")
        return

    # Compute class weights
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"  Class weights: {class_weight}")

    # Load pretrained base models
    mri_model = tf.keras.models.load_model(
        str(settings.MODEL_DIR / settings.MRI_MODEL_FILE), compile=False)
    fmri_model = tf.keras.models.load_model(
        str(settings.MODEL_DIR / settings.FMRI_MODEL_FILE), compile=False)

    # Create feature extractors (remove final sigmoid)
    mri_feat = create_feature_extractor(mri_model, name="mri_features")
    fmri_feat = create_feature_extractor(fmri_model, name="fmri_features")

    # Extract features
    print("  Extracting MRI features...")
    F_mri = mri_feat.predict(X_mri, verbose=0)
    print(f"    MRI features shape: {F_mri.shape}")
    print("  Extracting fMRI features...")
    F_fmri = fmri_feat.predict(X_fmri, verbose=0)
    print(f"    fMRI features shape: {F_fmri.shape}")

    # Concatenate features for simple fusion head
    F_all = np.concatenate([F_mri, F_fmri], axis=-1)
    print(f"  Concatenated features: {F_all.shape}")

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory(memory_file=str(settings.LOG_DIR / "fusion_agent_memory.json"))
    agent = LLMAgent(memory=memory)

    n_folds = min(args.folds, len(y) // 2)  # Safety: don't exceed data size
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    best_auc = 0
    best_threshold = 0.5

    for fold, (train_idx, val_idx) in enumerate(skf.split(F_all, y)):
        print(f"\n{'='*50}\nFold {fold+1}/{n_folds}\n{'='*50}")

        F_tr, y_tr = F_all[train_idx], y[train_idx]
        F_val, y_val = F_all[val_idx], y[val_idx]
        print(f"  Train: DL={int(y_tr.sum())}, TD={int(len(y_tr)-y_tr.sum())}")
        print(f"  Val:   DL={int(y_val.sum())}, TD={int(len(y_val)-y_val.sum())}")

        # Build a simple fusion head (Dense layers on concatenated features)
        feat_dim = F_all.shape[-1]
        fusion_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(feat_dim,)),
            tf.keras.layers.Dense(128, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation="relu",
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        lr = 1e-3

        for rnd in range(args.agent_rounds):
            print(f"\n  Agent round {rnd+1}/{args.agent_rounds}")

            fusion_model.compile(
                optimizer=tf.keras.optimizers.Adam(lr),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            fusion_model.fit(
                F_tr, y_tr,
                validation_data=(F_val, y_val),
                epochs=30,
                batch_size=4,
                class_weight=class_weight,
                verbose=1,
            )

            preds = fusion_model.predict(F_val, verbose=0).flatten()
            print(f"    Pred range: [{preds.min():.4f}, {preds.max():.4f}], "
                  f"mean={preds.mean():.4f}")

            # Optimal threshold via Youden's J
            if len(np.unique(y_val)) > 1:
                fpr, tpr, thresholds = roc_curve(y_val, preds)
                j_scores = tpr - fpr
                opt_idx = np.argmax(j_scores)
                opt_thresh = float(thresholds[opt_idx])
                auc = roc_auc_score(y_val, preds)
            else:
                opt_thresh = 0.5
                auc = 0.5

            pred_labels = (preds >= opt_thresh).astype(int)
            acc = accuracy_score(y_val, pred_labels)
            f1 = f1_score(y_val, pred_labels, zero_division=0)

            metrics = {"accuracy": acc, "f1": f1, "auc": auc,
                      "threshold": opt_thresh, "iteration": rnd}
            print(f"    Metrics: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, "
                  f"threshold={opt_thresh:.4f}")

            available_actions = ["reduce_lr", "train_longer", "augment_more"]
            decision = agent.decide(metrics, available_actions)
            action = decision["action"]
            print(f"    Agent action: {action} — {decision.get('reasoning', '')}")

            if action == "reduce_lr":
                lr *= 0.5

            after_metrics = {"accuracy": acc, "f1": f1, "auc": auc}
            agent.reflect(action, metrics, after_metrics)

            if auc > best_auc:
                best_auc = auc
                best_threshold = opt_thresh
                # Save as agentic fusion
                sp = settings.MODEL_DIR / settings.FUSION_AGENTIC_MODEL_FILE
                sp.parent.mkdir(parents=True, exist_ok=True)
                fusion_model.save(str(sp))
                # Save threshold
                thresh_path = settings.MODEL_DIR / "fusion_agentic_threshold.json"
                with open(thresh_path, "w") as f:
                    json.dump({"threshold": opt_thresh, "auc": auc}, f)
                print(f"    💾 Best fusion model saved (AUC={auc:.4f}, "
                      f"thresh={opt_thresh:.4f})")

    # Also save an HM fusion model (uses the full MRI+fMRI pipeline)
    print("\n  Building full agentic fusion model with extractors...")
    full_fusion = build_agentic_fusion(
        mri_feat, fmri_feat,
        mri_shape=settings.MRI_SHAPE,
        fmri_shape=settings.FMRI_SPATIAL_SHAPE + (settings.FMRI_TIME_STEPS,),
    )
    hm_path = settings.MODEL_DIR / settings.FUSION_HM_MODEL_FILE
    full_fusion.save(str(hm_path))
    # Save HM threshold (use same as agentic)
    thresh_path = settings.MODEL_DIR / "fusion_hm_threshold.json"
    with open(thresh_path, "w") as f:
        json.dump({"threshold": best_threshold, "auc": best_auc}, f)
    print(f"  💾 HM fusion model saved to {hm_path}")

    print(f"\n✅ Fusion training complete. Best AUC: {best_auc:.4f}, "
          f"Optimal threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--agent_rounds", type=int, default=2)
    train(parser.parse_args())

"""
Train 3D-CNN on structural MRI with agentic LLM-guided optimisation.

Key design choices:
  - Standard BCE loss (direct calibrated probabilities)
  - Balanced augmentation (80 per class) handles imbalance
  - Mild regularization (dropout=0.3, weight_decay=0.005, no L2)
  - 48 axial slices (middle 80% of brain volume)
  - Fast volume-level augmentations (rot90, flip, jitter, noise)
  - Test-time augmentation during evaluation

Usage:
    python -m cloud_system.training.train_mri --data_dir "E:/KNOWLEDGE_BASED_SYSTEMS (4)"
"""

import sys, os, argparse, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from cloud_system.config.settings import settings
from cloud_system.models.cnn3d import build_3dcnn
from cloud_system.preprocessing.mri_pipeline import preprocess_mri
from cloud_system.models.agentic_optimizer import AgentMemory, LLMAgent


# ── Data loading (BIDS format) ──

def load_dataset(data_dir: Path, extra_td_limit: int = 15):
    """
    Load MRI data from ds003126_raw (DL + TD).
    Also loads a LIMITED number of ds006239_raw TD subjects for variety.
    """
    volumes, labels = [], []

    # ── ds003126_raw (primary, balanced) ──
    ds1 = data_dir / "ds003126_raw"
    if not ds1.exists():
        print("[ERROR] ds003126_raw not found")
        return np.array([]), np.array([])

    tsv = ds1 / "participants.tsv"
    if not tsv.exists():
        print("[ERROR] participants.tsv not found")
        return np.array([]), np.array([])

    with open(tsv, "r") as f:
        lines = f.readlines()
    header = lines[0].strip().split("\t")
    group_idx = header.index("group") if "group" in header else -1

    for line in lines[1:]:
        parts = line.strip().split("\t")
        sub_id = parts[0]
        group = parts[group_idx] if group_idx >= 0 else "TD"

        sub_path = ds1 / sub_id
        if not sub_path.exists():
            continue

        for session in ["ses-1", "ses-2", ""]:
            session_path = sub_path / session if session else sub_path
            anat = session_path / "anat"
            if anat.exists():
                for fname in sorted(os.listdir(anat)):
                    if fname.endswith((".nii", ".nii.gz")) and "T1w" in fname:
                        fpath = str(anat / fname)
                        if not os.path.isfile(fpath):
                            continue
                        if os.path.getsize(fpath) == 0:
                            continue
                        vol = preprocess_mri(fpath)
                        if vol is not None:
                            volumes.append(vol)
                            labels.append(1 if group == "DL" else 0)
                            break

    n_dl_ds1 = sum(1 for l in labels if l == 1)
    n_td_ds1 = sum(1 for l in labels if l == 0)
    print(f"[DATA] ds003126_raw: {len(volumes)} samples (DL={n_dl_ds1}, TD={n_td_ds1})")

    # ── ds006239_raw (limited TD subjects for variety) ──
    ds2 = data_dir / "ds006239_raw"
    extra_added = 0
    if ds2.exists() and extra_td_limit > 0:
        sub_dirs = sorted([d for d in ds2.iterdir() if d.name.startswith("sub-")])
        np.random.seed(42)
        np.random.shuffle(sub_dirs)
        for sub_path in sub_dirs:
            if extra_added >= extra_td_limit:
                break
            for session in ["ses-1", "ses-2", ""]:
                session_path = sub_path / session if session else sub_path
                anat = session_path / "anat"
                if anat.exists():
                    for fname in sorted(os.listdir(anat)):
                        if fname.endswith((".nii", ".nii.gz")) and "T1w" in fname:
                            fpath = str(anat / fname)
                            if not os.path.isfile(fpath):
                                continue
                            if os.path.getsize(fpath) == 0:
                                continue
                            vol = preprocess_mri(fpath)
                            if vol is not None:
                                volumes.append(vol)
                                labels.append(0)  # all TD
                                extra_added += 1
                                break
                    if extra_added > 0:
                        break
        print(f"[DATA] ds006239_raw: added {extra_added} extra TD subjects")

    X = np.array(volumes) if volumes else np.array([]).reshape(0, *settings.MRI_SHAPE)
    y = np.array(labels) if labels else np.array([])
    n_dl = int(np.sum(y == 1))
    n_td = int(np.sum(y == 0))
    print(f"[DATA] TOTAL: {len(X)} samples (DL={n_dl}, TD={n_td})")
    return X, y


# ── Augmentation (volume-level transforms + slice shuffle + mixup) ──

def augment_balanced(X, y, target_per_class=150):
    """
    Augment BOTH classes to reach target_per_class samples each.
    Volume-level augmentations + slice shuffle + mixup for diversity.
    """
    aug_X, aug_y = [X.copy()], [y.copy()]

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        n_existing = len(idx)
        n_needed = max(0, target_per_class - n_existing)

        for _ in range(n_needed):
            i = idx[np.random.randint(len(idx))]
            vol = X[i].copy()

            # Apply 1-3 random augmentations for diversity
            n_augs = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            aug_choices = np.random.choice(7, size=n_augs, replace=False)
            for aug_type in aug_choices:
                if aug_type == 0:
                    # 90 degree rotation on spatial axes
                    k = np.random.choice([1, 2, 3])
                    vol = np.rot90(vol, k=k, axes=(1, 2))
                elif aug_type == 1:
                    # Horizontal flip
                    vol = vol[:, :, ::-1, :]
                elif aug_type == 2:
                    # Intensity jitter (scale + shift)
                    vol = vol * np.random.uniform(0.85, 1.15)
                    vol = vol + np.random.uniform(-0.04, 0.04)
                elif aug_type == 3:
                    # Gaussian noise
                    vol = vol + np.random.normal(0, 0.015, vol.shape)
                elif aug_type == 4:
                    # Vertical flip
                    vol = vol[:, ::-1, :, :]
                elif aug_type == 5:
                    # Depth flip (reverse slice order)
                    vol = vol[::-1, :, :, :]
                elif aug_type == 6:
                    # Slice shuffle (randomize slice order)
                    perm = np.random.permutation(vol.shape[0])
                    vol = vol[perm]

            vol = np.clip(vol, 0, 1).astype(np.float32)
            aug_X.append(vol[np.newaxis])
            aug_y.append(np.array([cls]))

    X_out = np.concatenate(aug_X)
    y_out = np.concatenate(aug_y)

    # Mixup: create 20% extra mixed samples (inter-class blending)
    n_mixup = int(len(y_out) * 0.2)
    mix_X, mix_y = [], []
    for _ in range(n_mixup):
        i1 = np.random.randint(len(y_out))
        i2 = np.random.randint(len(y_out))
        lam = np.random.beta(0.4, 0.4)  # bimodal -- mostly near 0 or 1
        x_mix = (lam * X_out[i1] + (1 - lam) * X_out[i2]).astype(np.float32)
        y_mix = lam * y_out[i1] + (1 - lam) * y_out[i2]
        mix_X.append(x_mix[np.newaxis])
        mix_y.append(np.array([y_mix]))

    if mix_X:
        X_out = np.concatenate([X_out] + mix_X)
        y_out = np.concatenate([y_out] + mix_y)

    # Shuffle
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ── Focal loss (NO label smoothing → confident predictions) ──

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss WITHOUT label smoothing.
    Targets stay 0.0 / 1.0 so the model can learn confident predictions.
    gamma=2.0 focuses learning on hard-to-classify examples.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Standard binary cross entropy
        bce = -(y_true * tf.math.log(y_pred) +
                (1 - y_true) * tf.math.log(1 - y_pred))

        # Focal modulating factor
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha * tf.pow(1 - p_t, gamma)

        return tf.reduce_mean(focal_weight * bce)

    return loss_fn


# ── Test-time augmentation ──

def predict_with_tta(model, X, n_augments=5):
    """
    Predict with test-time augmentation.
    Averages predictions over geometric variants for more stable confidence.
    """
    all_preds = []

    # Original
    all_preds.append(model.predict(X, verbose=0).flatten())

    # Horizontal flip
    X_hflip = X[:, :, :, ::-1, :]
    all_preds.append(model.predict(X_hflip, verbose=0).flatten())

    # Vertical flip
    X_vflip = X[:, :, ::-1, :, :]
    all_preds.append(model.predict(X_vflip, verbose=0).flatten())

    if n_augments >= 4:
        # 90° rotation
        X_rot90 = np.rot90(X, k=1, axes=(2, 3))
        all_preds.append(model.predict(X_rot90, verbose=0).flatten())

    if n_augments >= 5:
        # 270° rotation
        X_rot270 = np.rot90(X, k=3, axes=(2, 3))
        all_preds.append(model.predict(X_rot270, verbose=0).flatten())

    return np.mean(all_preds, axis=0)


# ── Agentic training loop ──

def train(args):
    data_dir = Path(args.data_dir)
    print(f"[Train MRI] Loading data from {data_dir}")
    X, y = load_dataset(data_dir, extra_td_limit=args.extra_td)
    print(f"  Loaded {len(X)} volumes  (DL={int(y.sum())}, TD={int(len(y)-y.sum())})")

    if len(X) < 4:
        print("[ERROR] Not enough data to train")
        return

    # Compute class weights for the full dataset
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"  Class weights: {class_weight}")

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory(memory_file=str(settings.LOG_DIR / "mri_agent_memory.json"))
    agent = LLMAgent(memory=memory)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    best_auc = 0
    best_threshold = 0.5

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*50}\nFold {fold+1}/{args.folds}\n{'='*50}")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(f"  Train: DL={int(y_train.sum())}, TD={int(len(y_train)-y_train.sum())}")
        print(f"  Val:   DL={int(y_val.sum())}, TD={int(len(y_val)-y_val.sum())}")

        # Balance via augmentation
        aug_target = args.aug_target
        X_aug, y_aug = augment_balanced(X_train, y_train, target_per_class=aug_target)
        print(f"  After augmentation: DL={int(y_aug.sum())}, "
              f"TD={int(len(y_aug)-y_aug.sum())} (target={aug_target})")

        model = build_3dcnn(settings.MRI_SHAPE)
        base_lr = 3e-4

        for epoch_block in range(args.agent_rounds):
            print(f"\n  Agent round {epoch_block+1}/{args.agent_rounds}")

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(
                    learning_rate=base_lr, weight_decay=0.02),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True,
                    verbose=1),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=8, verbose=1,
                    min_lr=1e-6),
            ]

            model.fit(
                X_aug, y_aug,
                validation_data=(X_val, y_val),
                epochs=args.epochs_per_round,
                batch_size=settings.MRI_BATCH_SIZE,
                callbacks=callbacks,
                verbose=1,
            )

            # Evaluate with test-time augmentation
            preds = predict_with_tta(model, X_val, n_augments=5)
            print(f"    Pred range (TTA): [{preds.min():.4f}, {preds.max():.4f}], "
                  f"mean={preds.mean():.4f}")

            raw_preds = model.predict(X_val, verbose=0).flatten()
            print(f"    Pred range (raw): [{raw_preds.min():.4f}, {raw_preds.max():.4f}], "
                  f"mean={raw_preds.mean():.4f}")

            # Find optimal threshold via Youden's J statistic
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
                      "threshold": opt_thresh, "iteration": epoch_block}
            print(f"    Metrics: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, "
                  f"threshold={opt_thresh:.4f}")

            # Agent decides
            available_actions = ["augment_more", "reduce_lr", "train_longer"]
            decision = agent.decide(metrics, available_actions)
            action = decision["action"]
            print(f"    Agent action: {action} — {decision.get('reasoning', '')}")

            if action == "augment_more":
                aug_target = min(aug_target + 50, 250)
                X_aug, y_aug = augment_balanced(X_train, y_train,
                                                target_per_class=aug_target)
                print(f"    Re-augmented to {aug_target} per class")
            elif action == "reduce_lr":
                base_lr *= 0.5
                print(f"    Base LR reduced to {base_lr:.6f}")

            after_metrics = {"accuracy": acc, "f1": f1, "auc": auc}
            agent.reflect(action, metrics, after_metrics)

            if auc > best_auc:
                best_auc = auc
                best_threshold = opt_thresh
                save_path = settings.MODEL_DIR / settings.MRI_MODEL_FILE
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(save_path))
                thresh_path = settings.MODEL_DIR / "mri_threshold.json"
                with open(thresh_path, "w") as f:
                    json.dump({"threshold": opt_thresh, "auc": auc}, f)
                print(f"    [SAVED] Best model (AUC={auc:.4f}, thresh={opt_thresh:.4f})")

    print(f"\n[DONE] Training complete. Best AUC: {best_auc:.4f}, "
          f"Optimal threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--agent_rounds", type=int, default=1)
    parser.add_argument("--epochs_per_round", type=int, default=100)
    parser.add_argument("--extra_td", type=int, default=15,
                        help="Number of extra TD subjects from ds006239_raw")
    parser.add_argument("--aug_target", type=int, default=150,
                        help="Augmentation target per class")
    train(parser.parse_args())

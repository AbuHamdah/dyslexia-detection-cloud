"""
Train CNN-LSTM on functional MRI with agentic LLM-guided optimisation.
Refactored from codes/fmri.py to use cloud_system modules.

Uses ds003126_raw + limited ds006239_raw with class_weight for balance.
Saves optimal threshold alongside the model.

Usage:
    python -m cloud_system.training.train_fmri --data_dir "E:/KNOWLEDGE_BASED_SYSTEMS (4)"
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
from cloud_system.models.cnn_lstm import build_cnn_lstm
from cloud_system.preprocessing.fmri_pipeline import preprocess_fmri
from cloud_system.models.agentic_optimizer import AgentMemory, LLMAgent


# ── Data loading (BIDS format) ──

def load_dataset(data_dir: Path):
    """
    Load fMRI data from BIDS datasets.
    ds003126_raw: reads participants.tsv for DL/TD labels
    ds006239_raw: limited to 10 subjects as TD
    """
    volumes, labels = [], []

    # ── Dataset 1: ds003126_raw (DL + TD) ──
    ds1 = data_dir / "ds003126_raw"
    if ds1.exists():
        tsv = ds1 / "participants.tsv"
        if tsv.exists():
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

                func_files = []
                for session in ["ses-1", "ses-2", ""]:
                    session_path = sub_path / session if session else sub_path
                    if not session_path.exists():
                        continue
                    func_dir = session_path / "func"
                    if func_dir.exists():
                        for fname in sorted(os.listdir(func_dir)):
                            if fname.endswith((".nii", ".nii.gz")) and "bold" in fname.lower():
                                fpath = str(func_dir / fname)
                                if os.path.getsize(fpath) == 0:
                                    continue
                                func_files.append(fpath)

                for fmri_file in func_files:
                    vol = preprocess_fmri(fmri_file)
                    if vol is not None:
                        volumes.append(vol)
                        labels.append(1 if group == "DL" else 0)

        n1 = len(volumes)
        dl1 = sum(labels)
        print(f"[DATA] ds003126_raw: {n1} samples (DL={dl1}, TD={n1 - dl1})")

    # ── Dataset 2: ds006239_raw (TD only, limited) ──
    ds2 = data_dir / "ds006239_raw"
    if ds2.exists():
        count_before = len(volumes)
        ds2_subjects = 0
        max_ds2_subjects = 10
        for folder in sorted(os.listdir(ds2)):
            if not folder.startswith("sub-"):
                continue
            if ds2_subjects >= max_ds2_subjects:
                break
            sub_path = ds2 / folder

            func_files = []
            for session in ["ses-1", "ses-2", ""]:
                session_path = sub_path / session if session else sub_path
                if not session_path.exists():
                    continue
                func_dir = session_path / "func"
                if func_dir.exists():
                    for fname in sorted(os.listdir(func_dir)):
                        if fname.endswith((".nii", ".nii.gz")) and "bold" in fname.lower():
                            fpath = str(func_dir / fname)
                            if os.path.getsize(fpath) == 0:
                                continue
                            func_files.append(fpath)

            for fmri_file in func_files[:2]:
                vol = preprocess_fmri(fmri_file)
                if vol is not None:
                    volumes.append(vol)
                    labels.append(0)
            ds2_subjects += 1

        print(f"[DATA] ds006239_raw: {len(volumes) - count_before} TD samples")

    X = np.array(volumes) if volumes else np.array([]).reshape(0, 64, 64, 3, 30)
    y = np.array(labels) if labels else np.array([])
    print(f"[DATA] TOTAL: {len(X)} samples (DL={int(np.sum(y==1))}, TD={int(np.sum(y==0))})")
    return X, y


# ── Augmentation (balance both classes) ──

def augment_balanced(X, y, target_per_class=80):
    """Augment both classes to reach target_per_class each."""
    aug_X, aug_y = [X.copy()], [y.copy()]

    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        n_existing = len(idx)
        n_needed = max(0, target_per_class - n_existing)

        for _ in range(n_needed):
            i = idx[np.random.randint(len(idx))]
            vol = X[i].copy()

            aug_type = np.random.randint(3)
            if aug_type == 0:
                vol = vol[::-1, :, :, :]  # flip
            elif aug_type == 1:
                vol = vol * np.random.uniform(0.8, 1.2)  # intensity
            elif aug_type == 2:
                vol = vol + np.random.normal(0, 0.02, vol.shape)  # noise

            vol = np.clip(vol, 0, 1).astype(np.float32)
            aug_X.append(vol[np.newaxis])
            aug_y.append(np.array([cls]))

    X_out = np.concatenate(aug_X)
    y_out = np.concatenate(aug_y)
    perm = np.random.permutation(len(y_out))
    return X_out[perm], y_out[perm]


# ── Agentic training loop ──

def train(args):
    data_dir = Path(args.data_dir)
    print(f"[Train fMRI] Loading data from {data_dir}")
    X, y = load_dataset(data_dir)
    print(f"  Loaded {len(X)} volumes  (DL={int(y.sum())}, TD={int(len(y)-y.sum())})")

    if len(X) < 4:
        print("[ERROR] Not enough data to train")
        return

    # Compute class weights
    classes = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes, y=y)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    print(f"  Class weights: {class_weight}")

    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory(memory_file=str(settings.LOG_DIR / "fmri_agent_memory.json"))
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

        # Balanced augmentation
        X_aug, y_aug = augment_balanced(X_train, y_train, target_per_class=80)
        print(f"  After augmentation: DL={int(y_aug.sum())}, TD={int(len(y_aug)-y_aug.sum())}")

        model = build_cnn_lstm(
            input_shape=settings.FMRI_SPATIAL_SHAPE + (settings.FMRI_TIME_STEPS,),
        )
        lr = settings.FMRI_LR

        for epoch_block in range(args.agent_rounds):
            print(f"\n  Agent round {epoch_block+1}/{args.agent_rounds}")

            model.compile(
                optimizer=tf.keras.optimizers.AdamW(lr, weight_decay=0.02),
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(
                X_aug, y_aug,
                validation_data=(X_val, y_val),
                epochs=args.epochs_per_round,
                batch_size=settings.FMRI_BATCH_SIZE,
                class_weight=class_weight,
                verbose=1,
            )

            preds = model.predict(X_val, verbose=0).flatten()
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
                      "threshold": opt_thresh, "iteration": epoch_block}
            print(f"    Metrics: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}, "
                  f"threshold={opt_thresh:.4f}")

            available_actions = ["augment_more", "reduce_lr", "train_longer"]
            decision = agent.decide(metrics, available_actions)
            action = decision["action"]
            print(f"    Agent action: {action} — {decision.get('reasoning', '')}")

            if action == "augment_more":
                X_aug, y_aug = augment_balanced(X_train, y_train, target_per_class=100)
            elif action == "reduce_lr":
                lr *= 0.5

            after_metrics = {"accuracy": acc, "f1": f1, "auc": auc}
            agent.reflect(action, metrics, after_metrics)

            if auc > best_auc:
                best_auc = auc
                best_threshold = opt_thresh
                save_path = settings.MODEL_DIR / settings.FMRI_MODEL_FILE
                save_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(save_path))
                # Save threshold
                thresh_path = settings.MODEL_DIR / "fmri_threshold.json"
                with open(thresh_path, "w") as f:
                    json.dump({"threshold": opt_thresh, "auc": auc}, f)
                print(f"    💾 Best model saved (AUC={auc:.4f}, thresh={opt_thresh:.4f})")

    print(f"\n✅ Training complete. Best AUC: {best_auc:.4f}, "
          f"Optimal threshold: {best_threshold:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--agent_rounds", type=int, default=2)
    parser.add_argument("--epochs_per_round", type=int, default=30)
    train(parser.parse_args())

# agentic_pretrained_fusion.py
import os
import sys
import json
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.ndimage import zoom, rotate, shift, gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import requests
import nibabel as nib

warnings.filterwarnings('ignore')

# OpenAI API Key — set via environment variable or .env file
# os.environ["OPENAI_API_KEY"] = "your-key-here"  # DO NOT hardcode

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# GPU memory growth
for device in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(device, True)
    except:
        pass

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration"""
    max_iterations: int = 10
    target_accuracy: float = 0.90
    target_f1: float = 0.85
    data_dir: str = "/content/drive/MyDrive/"

    # Pre-trained model paths
    mri_model_path: str = "/content/drive/MyDrive/best_mri_model.h5"
    fmri_model_path: str = "/content/drive/MyDrive/best_fmri_model.h5"

    # MRI shape (must match trained model)
    mri_shape: Tuple[int, int, int, int] = (10, 128, 128, 1)

    # fMRI shape (must match trained model)
    fmri_spatial: Tuple[int, int, int] = (64, 64, 3)
    fmri_timesteps: int = 30

    # Training
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 1e-3
    memory_file: str = "/content/drive/MyDrive/agent_memory_pretrained_fusion.json"


# ============================================================================
# AGENT MEMORY
# ============================================================================

class AgentMemory:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.experiences = self._load()

    def _load(self) -> List[Dict]:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.experiences, f, indent=2)

    def add(self, action: str, f1_before: float, f1_after: float):
        self.experiences.append({
            "action": action,
            "f1_before": f1_before,
            "f1_after": f1_after,
            "improvement": f1_after - f1_before,
            "success": f1_after > f1_before,
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_summary(self) -> str:
        if not self.experiences:
            return "No past experience."

        stats = {}
        for exp in self.experiences:
            a = exp["action"]
            if a not in stats:
                stats[a] = {"success": 0, "total": 0, "avg": 0}
            stats[a]["total"] += 1
            if exp["success"]:
                stats[a]["success"] += 1
            stats[a]["avg"] += exp["improvement"]

        lines = []
        for a, s in stats.items():
            rate = s["success"] / s["total"] * 100
            avg = s["avg"] / s["total"] * 100
            lines.append(f"  • {a}: {rate:.0f}% success ({s['total']} tries), avg: {avg:+.1f}%")
        return "\n".join(lines)


# ============================================================================
# LLM AGENT
# ============================================================================

class LLMAgent:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def query(self, prompt: str) -> str:
        if not self.api_key:
            return "train_fusion"
        try:
            response = requests.post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300, "temperature": 0.3},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except:
            pass
        return "train_fusion"

    def decide_action(self, state: Dict, memory_summary: str) -> Tuple[str, str]:
        prompt = f"""You are an AI agent optimizing a PRETRAINED FUSION model for dyslexia detection.
The model uses frozen pre-trained MRI and fMRI feature extractors, only training the fusion head.

Current State:
- Iteration: {state.get('iteration', 1)}/{state.get('max_iterations', 10)}
- Fusion F1: {state.get('f1', 0):.2%}
- Accuracy: {state.get('accuracy', 0):.2%}
- MRI features: frozen (pre-trained)
- fMRI features: frozen (pre-trained)
- Target F1: {state.get('target_f1', 0.85):.2%}

Learning Summary:
{memory_summary}

Available Actions:
1. train_fusion - Continue training fusion head
2. unfreeze_partial - Unfreeze last layers of both models for fine-tuning
3. reduce_lr - Reduce learning rate
4. add_regularization - Increase dropout in fusion head
5. adjust_weights - Modify fusion layer combination weights

Choose ONE action. Reply with action name on first line, then brief explanation.
"""
        response = self.query(prompt)
        lines = response.strip().split('\n')

        actions = ["train_fusion", "unfreeze_partial", "reduce_lr", "add_regularization", "adjust_weights"]
        action = "train_fusion"
        for a in actions:
            if a in lines[0].lower():
                action = a
                break

        reason = ' '.join(lines[1:]) if len(lines) > 1 else "Based on analysis."
        return action, reason


# ============================================================================
# DATA LOADERS (Same as before, but simplified)
# ============================================================================

class MRIDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.shape = config.mri_shape

    def load_volume(self, filepath: str) -> Optional[np.ndarray]:
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            if data.ndim == 4:
                data = np.mean(data, axis=3)

            target_d, target_h, target_w, _ = self.shape
            factors = [target_d/data.shape[0], target_h/data.shape[1], target_w/data.shape[2]]
            data = zoom(data, factors, order=1)

            v_min, v_max = data.min(), data.max()
            if v_max > v_min:
                data = (data - v_min) / (v_max - v_min)

            return data[..., np.newaxis].astype(np.float32)
        except:
            return None

    def load(self) -> Dict:
        print("[MRI] Loading...")
        X, y = [], []

        # Dataset 1: ds003126_raw (DL + TD)
        ds1 = os.path.join(self.data_dir, "ds003126_raw")
        if os.path.exists(ds1):
            pfile = os.path.join(ds1, "participants.tsv")
            if os.path.exists(pfile):
                with open(pfile, 'r') as f:
                    lines = f.readlines()
                    header = lines[0].strip().split('\t')
                    gidx = header.index('group') if 'group' in header else -1

                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        sub_id = parts[0]
                        group = parts[gidx] if gidx >= 0 else 'TD'

                        for ses in ['ses-1', 'ses-2', '']:
                            anat_dir = os.path.join(ds1, sub_id, ses, "anat") if ses else os.path.join(ds1, sub_id, "anat")
                            if os.path.exists(anat_dir):
                                for fname in os.listdir(anat_dir):
                                    if fname.endswith(('.nii', '.nii.gz')) and 'T1w' in fname:
                                        vol = self.load_volume(os.path.join(anat_dir, fname))
                                        if vol is not None:
                                            X.append(vol)
                                            y.append(1 if group == 'DL' else 0)
                                        break

        # Dataset 2: ds006239_raw (TD only - 83 subjects)
        ds2 = os.path.join(self.data_dir, "ds006239_raw")
        if os.path.exists(ds2):
            for sub in sorted(os.listdir(ds2)):
                if sub.startswith('sub-'):
                    for ses in ['ses-1', 'ses-2', '']:
                        anat_dir = os.path.join(ds2, sub, ses, "anat") if ses else os.path.join(ds2, sub, "anat")
                        if os.path.exists(anat_dir):
                            for fname in os.listdir(anat_dir):
                                if fname.endswith(('.nii', '.nii.gz')) and 'T1w' in fname:
                                    vol = self.load_volume(os.path.join(anat_dir, fname))
                                    if vol is not None:
                                        X.append(vol)
                                        y.append(0)  # TD
                                    break
                            break  # One MRI per subject

        X = np.array(X) if X else np.array([]).reshape(0, *self.shape)
        y = np.array(y)
        print(f"[MRI] {len(X)} samples (DL={np.sum(y==1)}, TD={np.sum(y==0)})")
        return {"X": X, "y": y}

    def augment(self, X, y, factor=6):
        dl_mask = y == 1
        X_dl, y_dl = X[dl_mask], y[dl_mask]
        X_td, y_td = X[~dl_mask], y[~dl_mask]

        X_aug, y_aug = [X_dl], [y_dl]
        for _ in range(factor - 1):
            aug = []
            for vol in X_dl:
                v = vol.copy()
                angle = np.random.uniform(-20, 20)
                for d in range(v.shape[0]):
                    v[d,:,:,0] = rotate(v[d,:,:,0], angle, reshape=False, order=1)
                v = np.clip(v * np.random.uniform(0.85, 1.15), 0, 1)
                aug.append(v)
            X_aug.append(np.array(aug))
            y_aug.append(y_dl.copy())

        X_final = np.concatenate([np.concatenate(X_aug), X_td])
        y_final = np.concatenate([np.concatenate(y_aug), y_td])
        idx = np.random.permutation(len(X_final))
        return X_final[idx], y_final[idx]


class FMRIDataLoader:
    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.spatial = config.fmri_spatial
        self.timesteps = config.fmri_timesteps

    def load_volume(self, filepath: str) -> Optional[np.ndarray]:
        try:
            img = nib.load(filepath)
            data = img.get_fdata()
            if data.ndim == 3:
                data = data[..., np.newaxis]

            n_t = data.shape[3]
            if n_t >= self.timesteps:
                indices = np.linspace(0, n_t-1, self.timesteps, dtype=int)
                data = data[..., indices]
            else:
                pad = np.repeat(data[...,-1:], self.timesteps - n_t, axis=3)
                data = np.concatenate([data, pad], axis=3)

            target_h, target_w, n_slices = self.spatial
            result = np.zeros((target_h, target_w, n_slices, self.timesteps), dtype=np.float32)

            for t in range(self.timesteps):
                vol = data[..., t]
                slices = [
                    vol[:, :, vol.shape[2]//2],
                    vol[:, vol.shape[1]//2, :],
                    vol[vol.shape[0]//2, :, :]
                ]
                for i, s in enumerate(slices):
                    factors = [target_h/s.shape[0], target_w/s.shape[1]]
                    result[:,:,i,t] = zoom(s, factors, order=1)

            v_min, v_max = result.min(), result.max()
            if v_max > v_min:
                result = (result - v_min) / (v_max - v_min)
            return result.astype(np.float32)
        except:
            return None

    def load(self) -> Dict:
        print("[fMRI] Loading...")
        X, y = [], []

        # Dataset 1
        ds1 = os.path.join(self.data_dir, "ds003126_raw")
        if os.path.exists(ds1):
            pfile = os.path.join(ds1, "participants.tsv")
            if os.path.exists(pfile):
                with open(pfile, 'r') as f:
                    lines = f.readlines()
                    header = lines[0].strip().split('\t')
                    gidx = header.index('group') if 'group' in header else -1

                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        sub_id = parts[0]
                        group = parts[gidx] if gidx >= 0 else 'TD'

                        for ses in ['ses-1', 'ses-2', '']:
                            func_dir = os.path.join(ds1, sub_id, ses, "func") if ses else os.path.join(ds1, sub_id, "func")
                            if os.path.exists(func_dir):
                                for fname in sorted(os.listdir(func_dir)):
                                    if fname.endswith(('.nii', '.nii.gz')) and 'bold' in fname.lower():
                                        vol = self.load_volume(os.path.join(func_dir, fname))
                                        if vol is not None:
                                            X.append(vol)
                                            y.append(1 if group == 'DL' else 0)

        # Dataset 2 (TD only)
        ds2 = os.path.join(self.data_dir, "ds006239_raw")
        if os.path.exists(ds2):
            for sub in sorted(os.listdir(ds2)):
                if sub.startswith('sub-'):
                    files = []
                    for ses in ['ses-1', 'ses-2', '']:
                        func_dir = os.path.join(ds2, sub, ses, "func") if ses else os.path.join(ds2, sub, "func")
                        if os.path.exists(func_dir):
                            for fname in sorted(os.listdir(func_dir)):
                                if fname.endswith(('.nii', '.nii.gz')) and 'bold' in fname.lower():
                                    files.append(os.path.join(func_dir, fname))
                    for fp in files[:2]:
                        vol = self.load_volume(fp)
                        if vol is not None:
                            X.append(vol)
                            y.append(0)

        X = np.array(X) if X else np.array([]).reshape(0, *self.spatial, self.timesteps)
        y = np.array(y)
        print(f"[fMRI] {len(X)} samples (DL={np.sum(y==1)}, TD={np.sum(y==0)})")
        return {"X": X, "y": y}

    def augment(self, X, y, factor=5):
        dl_mask = y == 1
        X_dl, y_dl = X[dl_mask], y[dl_mask]
        X_td, y_td = X[~dl_mask], y[~dl_mask]

        X_aug, y_aug = [X_dl], [y_dl]
        for _ in range(factor - 1):
            aug = []
            for vol in X_dl:
                v = vol.copy()
                angle = np.random.uniform(-15, 15)
                intensity = np.random.uniform(0.85, 1.15)
                for t in range(v.shape[-1]):
                    for s in range(v.shape[2]):
                        v[:,:,s,t] = rotate(v[:,:,s,t], angle, reshape=False, order=1)
                v = np.clip(v * intensity, 0, 1)
                aug.append(v)
            X_aug.append(np.array(aug))
            y_aug.append(y_dl.copy())

        X_final = np.concatenate([np.concatenate(X_aug), X_td])
        y_final = np.concatenate([np.concatenate(y_aug), y_td])
        idx = np.random.permutation(len(X_final))
        return X_final[idx], y_final[idx]


# ============================================================================
# PRETRAINED FUSION MODEL
# ============================================================================

def create_feature_extractor(model_path: str, model_type: str) -> keras.Model:
    """Load pre-trained model and create feature extractor (remove last sigmoid layer)"""
    print(f"[LOAD] Loading {model_type} model from {model_path}...")

    model = keras.models.load_model(model_path, compile=False)
    print(f"[LOAD] {model_type} model loaded: {model.count_params():,} params")

    # Find the layer before the final Dense(1, sigmoid)
    # Usually it's the second-to-last layer
    feature_layer = None
    for i, layer in enumerate(model.layers[:-1]):
        if isinstance(layer, layers.Dense):
            feature_layer = layer

    if feature_layer is None:
        # Use second-to-last layer
        feature_layer = model.layers[-2]

    # Create feature extractor
    feature_extractor = keras.Model(
        inputs=model.input,
        outputs=feature_layer.output,
        name=f'{model_type}_features'
    )

    # Freeze all layers
    for layer in feature_extractor.layers:
        layer.trainable = False

    print(f"[LOAD] {model_type} feature extractor: output shape = {feature_extractor.output_shape}")
    return feature_extractor


def build_fusion_model(mri_extractor: keras.Model, fmri_extractor: keras.Model,
                       config: Config, dropout: float = 0.5) -> keras.Model:
    """Build fusion model using pre-trained feature extractors"""

    # Get input shapes from extractors
    mri_input = keras.Input(shape=config.mri_shape, name='mri_input')
    fmri_input = keras.Input(shape=(*config.fmri_spatial, config.fmri_timesteps), name='fmri_input')

    # Extract features (frozen)
    mri_features = mri_extractor(mri_input)
    fmri_features = fmri_extractor(fmri_input)

    # Fusion head (trainable)
    fused = layers.Concatenate(name='fusion')([mri_features, fmri_features])

    x = layers.Dense(128, activation='relu', name='fusion_dense1',
                    kernel_regularizer=keras.regularizers.l2(0.01))(fused)
    x = layers.Dropout(dropout)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation='relu', name='fusion_dense2',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout * 0.6)(x)

    x = layers.Dense(32, activation='relu', name='fusion_dense3',
                    kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout * 0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = keras.Model(inputs=[mri_input, fmri_input], outputs=outputs, name='pretrained_fusion')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )

    return model


# ============================================================================
# PRETRAINED FUSION AGENT
# ============================================================================

class PretrainedFusionAgent:
    """Agent using pre-trained MRI and fMRI models for fusion"""

    def __init__(self, config: Config):
        self.config = config
        self.memory = AgentMemory(config.memory_file)
        self.llm = LLMAgent()

        # Data loaders
        self.mri_loader = MRIDataLoader(config)
        self.fmri_loader = FMRIDataLoader(config)

        # Load pre-trained feature extractors
        self.mri_extractor = create_feature_extractor(config.mri_model_path, "MRI")
        self.fmri_extractor = create_feature_extractor(config.fmri_model_path, "fMRI")

        self.best_f1 = 0.0
        self.best_model = None
        self.dropout = 0.5
        self.mri_aug = 6
        self.fmri_aug = 5

        self.raw_mri = None
        self.raw_fmri = None

    def prepare_data(self) -> Dict:
        """Prepare aligned MRI + fMRI data"""

        X_mri, y_mri = self.mri_loader.augment(self.raw_mri["X"], self.raw_mri["y"], self.mri_aug)
        X_fmri, y_fmri = self.fmri_loader.augment(self.raw_fmri["X"], self.raw_fmri["y"], self.fmri_aug)

        print(f"[DATA] MRI: {len(X_mri)} (DL={np.sum(y_mri==1)}, TD={np.sum(y_mri==0)})")
        print(f"[DATA] fMRI: {len(X_fmri)} (DL={np.sum(y_fmri==1)}, TD={np.sum(y_fmri==0)})")

        # Align: take minimum of each class
        n_dl = min(np.sum(y_mri==1), np.sum(y_fmri==1))
        n_td = min(np.sum(y_mri==0), np.sum(y_fmri==0))

        mri_dl = np.where(y_mri==1)[0][:n_dl]
        mri_td = np.where(y_mri==0)[0][:n_td]
        fmri_dl = np.where(y_fmri==1)[0][:n_dl]
        fmri_td = np.where(y_fmri==0)[0][:n_td]

        X_mri_aligned = np.concatenate([X_mri[mri_dl], X_mri[mri_td]])
        X_fmri_aligned = np.concatenate([X_fmri[fmri_dl], X_fmri[fmri_td]])
        y_aligned = np.concatenate([np.ones(n_dl), np.zeros(n_td)])

        idx = np.random.permutation(len(y_aligned))
        X_mri_aligned = X_mri_aligned[idx]
        X_fmri_aligned = X_fmri_aligned[idx]
        y_aligned = y_aligned[idx]

        print(f"[ALIGNED] {len(y_aligned)} pairs (DL={n_dl}, TD={n_td})")

        # Split
        indices = np.arange(len(y_aligned))
        train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=y_aligned, random_state=42)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=y_aligned[temp_idx], random_state=42)

        print(f"[SPLIT] Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

        return {
            "X_mri_train": X_mri_aligned[train_idx], "X_fmri_train": X_fmri_aligned[train_idx], "y_train": y_aligned[train_idx],
            "X_mri_val": X_mri_aligned[val_idx], "X_fmri_val": X_fmri_aligned[val_idx], "y_val": y_aligned[val_idx],
            "X_mri_test": X_mri_aligned[test_idx], "X_fmri_test": X_fmri_aligned[test_idx], "y_test": y_aligned[test_idx]
        }

    def train_model(self, data: Dict) -> Tuple[Dict, keras.Model]:
        """Train fusion head"""

        print(f"[MODEL] Building Fusion Model (pretrained extractors frozen)...")
        model = build_fusion_model(self.mri_extractor, self.fmri_extractor, self.config, self.dropout)

        trainable = sum([np.prod(w.shape) for w in model.trainable_weights])
        total = model.count_params()
        print(f"[MODEL] Total: {total:,} params, Trainable: {trainable:,} params (fusion head only)")

        n_pos = np.sum(data["y_train"] == 1)
        n_neg = np.sum(data["y_train"] == 0)
        class_weight = {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]

        print(f"[TRAIN] Epochs={self.config.epochs}, LR={self.config.learning_rate:.2e}")

        history = model.fit(
            [data["X_mri_train"], data["X_fmri_train"]], data["y_train"],
            validation_data=([data["X_mri_val"], data["X_fmri_val"]], data["y_val"]),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_prob = model.predict([data["X_mri_val"], data["X_fmri_val"]]).flatten()

        best_thresh, best_f1_val = 0.5, 0
        for thresh in np.arange(0.3, 0.7, 0.05):
            yp = (y_prob >= thresh).astype(int)
            f1 = f1_score(data["y_val"], yp, zero_division=0)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_thresh = thresh

        y_pred = (y_prob >= best_thresh).astype(int)

        results = {
            "accuracy": accuracy_score(data["y_val"], y_pred),
            "f1": f1_score(data["y_val"], y_pred),
            "precision": precision_score(data["y_val"], y_pred, zero_division=0),
            "recall": recall_score(data["y_val"], y_pred, zero_division=0),
            "threshold": best_thresh
        }

        print(f"[RESULTS] Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}, "
              f"P={results['precision']:.2%}, R={results['recall']:.2%}")

        if results["f1"] > self.best_f1:
            self.best_f1 = results["f1"]
            self.best_model = model
            model.save("/content/drive/MyDrive/best_pretrained_fusion.h5")
            print(f"[BEST] 🏆 New best F1: {self.best_f1:.2%}")

        return results, model

    def execute_action(self, action: str):
        """Execute agent action"""

        if action == "train_fusion":
            print(f"[ACTION] Continue training fusion head")

        elif action == "unfreeze_partial":
            print(f"[ACTION] Unfreezing last layers of feature extractors...")
            # Unfreeze last few layers of each extractor
            for layer in self.mri_extractor.layers[-3:]:
                layer.trainable = True
            for layer in self.fmri_extractor.layers[-3:]:
                layer.trainable = True
            self.config.learning_rate *= 0.1  # Lower LR for fine-tuning

        elif action == "reduce_lr":
            self.config.learning_rate *= 0.5
            print(f"[ACTION] Reduced LR to {self.config.learning_rate:.2e}")

        elif action == "add_regularization":
            self.dropout = min(0.7, self.dropout + 0.1)
            print(f"[ACTION] Increased dropout to {self.dropout}")

        elif action == "adjust_weights":
            self.mri_aug += 1
            self.fmri_aug += 1
            print(f"[ACTION] Adjusted augmentation: MRI={self.mri_aug}x, fMRI={self.fmri_aug}x")

    def run(self):

        self.raw_mri = self.mri_loader.load()
        self.raw_fmri = self.fmri_loader.load()

        if len(self.raw_mri["X"]) == 0 or len(self.raw_fmri["X"]) == 0:
            print("[ERROR] No data found!")
            return None

        prev_f1 = 0.0

        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"[ITERATION {iteration}/{self.config.max_iterations}]")
            print(f"{'='*70}")

            data = self.prepare_data()
            results, model = self.train_model(data)

            if results["accuracy"] >= self.config.target_accuracy and results["f1"] >= self.config.target_f1:
                print(f"\n🎉 TARGET ACHIEVED! Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}")
                break

            # Agent decision
            print(f"\n🤖 [AGENT] Analyzing...")
            print(f"📊 Learning Summary:\n{self.memory.get_summary()}")

            state = {
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
                "accuracy": results["accuracy"],
                "f1": results["f1"],
                "target_f1": self.config.target_f1
            }

            action, reason = self.llm.decide_action(state, self.memory.get_summary())

            print(f"\n[DECISION] 🤖 LLM: {action}")
            print(f"[REASON] {reason[:200]}")

            self.execute_action(action)
            self.memory.add(action, prev_f1, results["f1"])

            if results["f1"] > prev_f1:
                print(f"[REFLECT] ✅ Improved F1 by {(results['f1'] - prev_f1)*100:+.2f}%")
            else:
                print(f"[REFLECT] ➖ Minimal effect")

            prev_f1 = results["f1"]

        # Final test evaluation
        print(f"\n{'='*70}")
        print("[FINAL RESULTS - TEST SET]")
        print(f"{'='*70}")

        if self.best_model is not None:
            data = self.prepare_data()
            y_prob = self.best_model.predict([data["X_mri_test"], data["X_fmri_test"]]).flatten()
            y_pred = (y_prob >= 0.5).astype(int)

            test_acc = accuracy_score(data["y_test"], y_pred)
            test_f1 = f1_score(data["y_test"], y_pred)
            test_prec = precision_score(data["y_test"], y_pred, zero_division=0)
            test_rec = recall_score(data["y_test"], y_pred, zero_division=0)

            print(f"Test Accuracy:  {test_acc:.2%}")
            print(f"Test F1-Score:  {test_f1:.2%}")
            print(f"Test Precision: {test_prec:.2%}")
            print(f"Test Recall:    {test_rec:.2%}")

            cm = confusion_matrix(data["y_test"], y_pred)
            print(f"\nConfusion Matrix:\n{cm}")

            print(f"\n📊 Learning Summary:\n{self.memory.get_summary()}")

            return {"test_accuracy": test_acc, "test_f1": test_f1, "test_precision": test_prec, "test_recall": test_rec}

        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Config(
        max_iterations=10,
        target_accuracy=0.90,
        target_f1=0.85,
        mri_model_path="/content/drive/MyDrive/best_mri_model.h5",
        fmri_model_path="/content/drive/MyDrive/best_fmri_model.h5",
        mri_shape=(10, 128, 128, 1),
        fmri_spatial=(64, 64, 3),
        fmri_timesteps=30,
        batch_size=4,
        epochs=50,
        learning_rate=1e-3
    )

    agent = PretrainedFusionAgent(config)
    results = agent.run()

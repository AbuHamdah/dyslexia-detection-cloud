
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
import re
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Simple configuration"""
    max_iterations: int = 10
    target_accuracy: float = 0.85
    target_f1: float = 0.80
    data_dir: str = "/content/drive/MyDrive/"
    spatial_shape: Tuple[int, int, int] = (64, 64, 3)  # (H, W, 3) - 3 slices: axial, coronal, sagittal
    time_steps: int = 30  # Reduced from 120 to save GPU memory
    batch_size: int = 4  # Smaller batch for memory efficiency
    epochs: int = 100
    learning_rate: float = 5e-4
    memory_file: str = "/content/drive/MyDrive/agent_memory_fmri.json"


# ============================================================================
# SIMPLE MEMORY - Learns from past experiences
# ============================================================================

class SimpleMemory:
    """Simple JSON-based memory that learns from past experiences"""

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

    def add(self, action: str, f1_before: float, f1_after: float, params: Dict = None):
        """Record an experience"""
        self.experiences.append({
            "action": action,
            "f1_before": f1_before,
            "f1_after": f1_after,
            "improvement": f1_after - f1_before,
            "success": f1_after > f1_before,
            "params": params or {},
            "timestamp": datetime.now().isoformat()
        })
        self.save()

    def get_action_stats(self) -> Dict:
        """Get success statistics for each action"""
        stats = {}
        for exp in self.experiences:
            action = exp["action"]
            if action not in stats:
                stats[action] = {"success": 0, "total": 0, "avg_improvement": 0}
            stats[action]["total"] += 1
            if exp["success"]:
                stats[action]["success"] += 1
            stats[action]["avg_improvement"] += exp["improvement"]

        # Calculate averages
        for action in stats:
            if stats[action]["total"] > 0:
                stats[action]["avg_improvement"] /= stats[action]["total"]
                stats[action]["success_rate"] = stats[action]["success"] / stats[action]["total"]

        return stats

    def get_summary(self) -> str:
        """Get learning summary for LLM"""
        stats = self.get_action_stats()
        if not stats:
            return "No past experience available."

        summary = []
        for action, s in stats.items():
            summary.append(f"  • {action}: {s['success_rate']*100:.0f}% success ({s['total']} tries), avg: {s['avg_improvement']*100:+.1f}%")
        return "\n".join(summary)


# ============================================================================
# LLM AGENT - Real reasoning with GPT
# ============================================================================

class LLMAgent:
    """Simple LLM agent using OpenAI API"""

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def query(self, prompt: str) -> str:
        """Query OpenAI API"""
        if not self.api_key:
            return "augment_more"  # Default fallback

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.3
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[LLM] API Error: {e}")
        return "augment_more"

    def decide_action(self, state: Dict, memory_summary: str) -> Tuple[str, str]:
        """Decide next action based on current state and past experience"""

        prompt = f"""You are an AI agent optimizing a CNN-LSTM model for fMRI dyslexia detection.

Current State:
- Iteration: {state.get('iteration', 1)}/{state.get('max_iterations', 10)}
- Current F1: {state.get('f1', 0):.2%}
- Current Accuracy: {state.get('accuracy', 0):.2%}
- Target F1: {state.get('target_f1', 0.80):.2%}
- Target Accuracy: {state.get('target_accuracy', 0.85):.2%}
- Samples: DL={state.get('n_dl', 0)}, TD={state.get('n_td', 0)}

Learning Summary from past iterations:
{memory_summary}

Available Actions:
1. augment_more - Increase data augmentation (helps with limited data)
2. train_longer - Increase epochs (helps if model hasn't converged)
3. reduce_lr - Reduce learning rate (helps with fine-tuning)
4. adjust_threshold - Optimize classification threshold

Choose ONE action. Reply with ONLY the action name on the first line, then explain briefly.
"""

        response = self.query(prompt)
        lines = response.strip().split('\n')

        # Extract action from response
        action = "augment_more"
        for a in ["augment_more", "train_longer", "reduce_lr", "adjust_threshold"]:
            if a in lines[0].lower():
                action = a
                break

        reason = ' '.join(lines[1:]) if len(lines) > 1 else "Based on current state analysis."
        return action, reason


# ============================================================================
# DATA LOADER - Loads and preprocesses 4D fMRI data
# ============================================================================

class FMRIDataLoader:
    """Loads and preprocesses 4D fMRI data"""

    def __init__(self, config: Config):
        self.config = config
        self.data_dir = config.data_dir
        self.spatial_shape = config.spatial_shape
        self.time_steps = config.time_steps

    def load_fmri_volume(self, filepath: str) -> Optional[np.ndarray]:
        """Load and preprocess a 4D fMRI volume

        Each fMRI scan is structured as a time series of 120 volumes.
        Volumes are resized to 64×64 pixels and normalized to [0,1].
        """
        try:
            img = nib.load(filepath)
            data = img.get_fdata()

            # Handle 3D images (treat as single time point)
            if data.ndim == 3:
                data = data[..., np.newaxis]

            # Get dimensions
            n_timepoints = data.shape[3] if data.ndim == 4 else 1

            # Sample or pad time points to get exactly self.time_steps (120)
            if n_timepoints >= self.time_steps:
                indices = np.linspace(0, n_timepoints - 1, self.time_steps, dtype=int)
                data = data[..., indices]
            else:
                pad_size = self.time_steps - n_timepoints
                padding = np.repeat(data[..., -1:], pad_size, axis=3)
                data = np.concatenate([data, padding], axis=3)

            # Resize each volume to 64x64 with 3 orthogonal slices (axial, coronal, sagittal)
            target_h, target_w, n_slices = self.spatial_shape  # (64, 64, 3)
            resized_data = np.zeros((target_h, target_w, n_slices, self.time_steps), dtype=np.float32)

            for t in range(self.time_steps):
                vol_3d = data[..., t]

                # Extract 3 orthogonal slices for better spatial coverage
                # Axial slice (middle Z)
                mid_z = vol_3d.shape[2] // 2
                axial = vol_3d[:, :, mid_z]

                # Coronal slice (middle Y)
                mid_y = vol_3d.shape[1] // 2
                coronal = vol_3d[:, mid_y, :]

                # Sagittal slice (middle X)
                mid_x = vol_3d.shape[0] // 2
                sagittal = vol_3d[mid_x, :, :]

                # Resize all slices to target size
                for i, slice_2d in enumerate([axial, coronal, sagittal]):
                    factors = [target_h / slice_2d.shape[0], target_w / slice_2d.shape[1]]
                    resized_slice = zoom(slice_2d, factors, order=1)
                    resized_data[:, :, i, t] = resized_slice

            # Normalize to [0,1]
            v_min, v_max = resized_data.min(), resized_data.max()
            if v_max > v_min:
                resized_data = (resized_data - v_min) / (v_max - v_min)

            return resized_data.astype(np.float32)

        except Exception as e:
            print(f"[ERROR] Loading {filepath}: {e}")
            return None

    def load(self) -> Dict:
        """Load the complete fMRI dataset"""
        print("[DataLoader] Loading fMRI Dataset...")

        X_data = []
        y_data = []

        # Dataset 1: ds003126_raw (DL + TD)
        ds1_path = os.path.join(self.data_dir, "ds003126_raw")
        if os.path.exists(ds1_path):
            participants_file = os.path.join(ds1_path, "participants.tsv")
            if os.path.exists(participants_file):
                with open(participants_file, 'r') as f:
                    lines = f.readlines()
                    header = lines[0].strip().split('\t')
                    group_idx = header.index('group') if 'group' in header else -1

                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        sub_id = parts[0]
                        group = parts[group_idx] if group_idx >= 0 else 'TD'

                        sub_path = os.path.join(ds1_path, sub_id)
                        if not os.path.exists(sub_path):
                            continue

                        func_files = []
                        for session in ['ses-1', 'ses-2', '']:
                            session_path = os.path.join(sub_path, session) if session else sub_path
                            if not os.path.exists(session_path):
                                continue

                            func_dir = os.path.join(session_path, "func")
                            if os.path.exists(func_dir):
                                for fname in sorted(os.listdir(func_dir)):
                                    if fname.endswith(('.nii', '.nii.gz')) and 'bold' in fname.lower():
                                        func_files.append(os.path.join(func_dir, fname))

                        for fmri_file in func_files:
                            volume = self.load_fmri_volume(fmri_file)
                            if volume is not None:
                                X_data.append(volume)
                                y_data.append(1 if group == 'DL' else 0)

                print(f"[DataLoader] Dataset 1: {len(X_data)} samples (DL={sum(y_data)}, TD={len(y_data)-sum(y_data)})")

        # Dataset 2: ds006239_raw (TD only) - first 2 per subject
        ds2_path = os.path.join(self.data_dir, "ds006239_raw")
        if os.path.exists(ds2_path):
            count_before = len(X_data)
            for sub_folder in sorted(os.listdir(ds2_path)):
                if sub_folder.startswith('sub-'):
                    sub_path = os.path.join(ds2_path, sub_folder)

                    func_files = []
                    for session in ['ses-1', 'ses-2', '']:
                        session_path = os.path.join(sub_path, session) if session else sub_path
                        if not os.path.exists(session_path):
                            continue

                        func_dir = os.path.join(session_path, "func")
                        if os.path.exists(func_dir):
                            for fname in sorted(os.listdir(func_dir)):
                                if fname.endswith(('.nii', '.nii.gz')) and 'bold' in fname.lower():
                                    func_files.append(os.path.join(func_dir, fname))

                    for fmri_file in func_files[:2]:  # Only first 2
                        volume = self.load_fmri_volume(fmri_file)
                        if volume is not None:
                            X_data.append(volume)
                            y_data.append(0)  # TD

            print(f"[DataLoader] Dataset 2: {len(X_data) - count_before} TD samples")

        X = np.array(X_data)
        y = np.array(y_data)

        n_dl = np.sum(y == 1)
        n_td = np.sum(y == 0)

        print(f"[DataLoader] TOTAL: {len(X)} samples (DL={n_dl}, TD={n_td})")
        print(f"[DataLoader] Shape: {X.shape}")

        return {"X": X, "y": y, "n_dl": n_dl, "n_td": n_td}

    def augment(self, X: np.ndarray, y: np.ndarray, factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Augment ONLY minority class (DL) to balance with majority (TD)"""

        dl_mask = y == 1
        td_mask = y == 0

        X_dl, y_dl = X[dl_mask], y[dl_mask]
        X_td, y_td = X[td_mask], y[td_mask]

        print(f"[AUGMENT] Before: DL={len(X_dl)}, TD={len(X_td)}")

        # Augment DL class
        X_dl_aug = [X_dl]
        y_dl_aug = [y_dl]

        for i in range(factor - 1):
            augmented = []
            for vol in X_dl:
                aug = self._apply_augmentation(vol)
                augmented.append(aug)
            X_dl_aug.append(np.array(augmented))
            y_dl_aug.append(y_dl.copy())

        X_dl_final = np.concatenate(X_dl_aug, axis=0)
        y_dl_final = np.concatenate(y_dl_aug, axis=0)

        # Combine with original TD
        X_final = np.concatenate([X_dl_final, X_td], axis=0)
        y_final = np.concatenate([y_dl_final, y_td], axis=0)

        # Shuffle
        idx = np.random.permutation(len(X_final))

        print(f"[AUGMENT] After: DL={np.sum(y_final==1)}, TD={np.sum(y_final==0)}")
        return X_final[idx], y_final[idx]

    def _apply_augmentation(self, volume: np.ndarray) -> np.ndarray:
        """Apply augmentation to a 4D fMRI volume"""
        aug = volume.copy()

        angle = np.random.uniform(-15, 15)
        shifts = [np.random.uniform(-3, 3), np.random.uniform(-3, 3)]
        intensity = np.random.uniform(0.85, 1.15)

        for t in range(aug.shape[-1]):
            for s in range(aug.shape[2]):
                slice_2d = aug[:, :, s, t]
                slice_2d = rotate(slice_2d, angle, reshape=False, order=1, mode='nearest')
                slice_2d = shift(slice_2d, shifts, order=1, mode='nearest')
                slice_2d = slice_2d * intensity
                if np.random.random() > 0.5:
                    slice_2d = gaussian_filter(slice_2d, sigma=np.random.uniform(0, 0.5))
                aug[:, :, s, t] = np.clip(slice_2d, 0, 1)

        return aug.astype(np.float32)

    def prepare(self, X: np.ndarray, y: np.ndarray, aug_factor: int = 5) -> Dict:
        """Prepare data: augment and split"""
        if len(X) == 0:
            raise ValueError("No data loaded!")

        X_aug, y_aug = self.augment(X, y, aug_factor)

        # Split: Train 70%, Val 15%, Test 15%
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
        )

        print(f"[SPLIT] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }


# ============================================================================
# MODEL BUILDER - CNN-LSTM
# ============================================================================

def build_model(config: Config, params: Dict = None) -> keras.Model:
    """Build 2D-CNN-LSTM model with anti-overfitting measures"""
    params = params or {}
    lr = params.get("learning_rate", config.learning_rate)
    dropout = params.get("dropout", 0.5)
    l2_reg = 0.01

    # Input shape: (H, W, 1, time_steps) = (64, 64, 1, 120)
    inputs = keras.Input(shape=(*config.spatial_shape, config.time_steps))

    # Permute to (time_steps, H, W, 1)
    x = layers.Permute((4, 1, 2, 3))(inputs)

    # TimeDistributed 2D-CNN - Extract spatial features
    # CNN Block 1
    x = layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.3))(x)

    # CNN Block 2
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Dropout(dropout * 0.4))(x)

    # CNN Block 3
    x = layers.TimeDistributed(
        layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))
    )(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    # LSTM for temporal features - shape now (batch, time_steps, 128)
    # Note: recurrent_dropout removed for cuDNN compatibility (faster training)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Add explicit dropout instead of recurrent_dropout

    x = layers.Bidirectional(layers.LSTM(32, return_sequences=False, dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    # Classification head
    x = layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout * 0.5)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )

    return model


# ============================================================================
# FMRI AGENT - Main agentic loop
# ============================================================================

class FMRIAgent:
    """Agentic CNN-LSTM for fMRI dyslexia detection"""

    def __init__(self, config: Config):
        self.config = config
        self.loader = FMRIDataLoader(config)
        self.memory = SimpleMemory(config.memory_file)
        self.llm = LLMAgent()

        self.best_f1 = 0.0
        self.best_model = None
        self.aug_factor = 5
        self.raw_data = None

    def train_model(self, data: Dict, params: Dict = None) -> Tuple[Dict, keras.Model]:
        """Train the model and return results"""
        params = params or {}
        epochs = params.get("epochs", self.config.epochs)

        # Class weights for imbalance
        n_pos = np.sum(data["y_train"] == 1)
        n_neg = np.sum(data["y_train"] == 0)
        class_weight = {0: 1.0, 1: n_neg / n_pos if n_pos > 0 else 1.0}

        print(f"[MODEL] Building CNN-LSTM...")
        model = build_model(self.config, params)
        print(f"[MODEL] Parameters: {model.count_params():,}")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            )
        ]

        print(f"[TRAIN] Epochs={epochs}, LR={params.get('learning_rate', self.config.learning_rate):.2e}")

        history = model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate with optimal threshold
        y_prob = model.predict(data["X_val"]).flatten()
        best_thresh, best_f1_val = 0.5, 0
        for thresh in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(data["y_val"], y_pred, zero_division=0)
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

        # Save best
        if results["f1"] > self.best_f1:
            self.best_f1 = results["f1"]
            self.best_model = model
            model.save("/content/drive/MyDrive/best_fmri_model.h5")
            print(f"[BEST] New best F1: {self.best_f1:.2%}")

        return results, model

    def execute_action(self, action: str, data: Dict) -> Dict:
        """Execute an action and return updated data"""

        if action == "augment_more":
            self.aug_factor += 2
            print(f"[ACTION] Re-augmenting from original data with factor={self.aug_factor}")
            data = self.loader.prepare(self.raw_data["X"], self.raw_data["y"], self.aug_factor)

        elif action == "train_longer":
            self.config.epochs = min(150, self.config.epochs + 30)
            print(f"[ACTION] Increased epochs to {self.config.epochs}")

        elif action == "reduce_lr":
            self.config.learning_rate *= 0.5
            print(f"[ACTION] Reduced LR to {self.config.learning_rate:.2e}")

        elif action == "adjust_threshold":
            print(f"[ACTION] Will optimize threshold during evaluation")

        return data

    def run(self):
        """Run the agentic training loop"""
        print("""
╔══════════════════════════════════════════════════════════════╗
║  SIMPLIFIED AGENTIC CNN-LSTM FOR fMRI DYSLEXIA DETECTION  ║
╠══════════════════════════════════════════════════════════════╣
║  ✅ Real learning from past experiences                      ║
║  ✅ LLM-guided decision making                               ║
║  ✅ CNN-LSTM for spatial-temporal features                   ║
╚══════════════════════════════════════════════════════════════╝
""")

        # Load data
        raw = self.loader.load()
        if len(raw["X"]) == 0:
            print("[ERROR] No data found!")
            return None

        self.raw_data = raw

        # Prepare initial data
        data = self.loader.prepare(raw["X"], raw["y"], self.aug_factor)

        prev_f1 = 0.0

        for iteration in range(1, self.config.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"[ITERATION {iteration}/{self.config.max_iterations}]")
            print(f"{'='*60}")

            # Train model
            results, model = self.train_model(data, {
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.epochs
            })

            # Check if target achieved
            if results["accuracy"] >= self.config.target_accuracy and \
               results["f1"] >= self.config.target_f1:
                print(f"\n TARGET ACHIEVED! Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}")
                break

            # Get LLM decision
            print(f"\n [AGENT] Analyzing with learned experience...")
            print(f"Learning Summary:\n{self.memory.get_summary()}")

            state = {
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
                "accuracy": results["accuracy"],
                "f1": results["f1"],
                "target_accuracy": self.config.target_accuracy,
                "target_f1": self.config.target_f1,
                "n_dl": np.sum(data["y_train"] == 1),
                "n_td": np.sum(data["y_train"] == 0)
            }

            action, reason = self.llm.decide_action(state, self.memory.get_summary())

            print(f"\n[DECISION] LLM: {action}")
            print(f"[REASON] {reason[:200]}")

            # Execute action
            print(f"[ACTION] Executing: {action}")
            data = self.execute_action(action, data)

            # Record experience
            self.memory.add(action, prev_f1, results["f1"])

            if results["f1"] > prev_f1:
                print(f"[REFLECT] {action} improved F1 by {(results['f1'] - prev_f1)*100:+.2f}%")
            else:
                print(f"[REFLECT] {action} had minimal effect")

            prev_f1 = results["f1"]

        # Final evaluation on test set
        print(f"\n{'='*60}")
        print("[FINAL RESULTS]")
        print(f"{'='*60}")

        if self.best_model is not None:
            y_prob = self.best_model.predict(data["X_test"]).flatten()
            y_pred = (y_prob >= 0.5).astype(int)

            test_acc = accuracy_score(data["y_test"], y_pred)
            test_f1 = f1_score(data["y_test"], y_pred)
            test_prec = precision_score(data["y_test"], y_pred, zero_division=0)
            test_rec = recall_score(data["y_test"], y_pred, zero_division=0)

            print(f"Test Accuracy: {test_acc:.2%}")
            print(f"Test F1: {test_f1:.2%}")
            print(f"Test Precision: {test_prec:.2%}")
            print(f"Test Recall: {test_rec:.2%}")

            cm = confusion_matrix(data["y_test"], y_pred)
            print(f"\nConfusion Matrix:\n{cm}")

            print(f"\nLearning Summary:\n{self.memory.get_summary()}")

            return {
                "test_accuracy": test_acc,
                "test_f1": test_f1,
                "test_precision": test_prec,
                "test_recall": test_rec
            }

        return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    config = Config(
        max_iterations=10,
        target_accuracy=0.85,
        target_f1=0.80,
        spatial_shape=(64, 64, 3),  # 3 slices: axial, coronal, sagittal
        time_steps=30,  # Reduced for GPU memory efficiency
        batch_size=4,   # Smaller batch for stability
        epochs=100
    )

    agent = FMRIAgent(config)
    results = agent.run()

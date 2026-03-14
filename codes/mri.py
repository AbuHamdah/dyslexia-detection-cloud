"""
SIMPLIFIED AGENTIC 3D-CNN FOR MRI DYSLEXIA DETECTION
=====================================================
"""

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
from scipy.ndimage import zoom, rotate, shift
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import requests
import re

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

# Set random seeds for different results
np.random.seed(123)
tf.random.set_seed(123)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Simple configuration"""
    max_iterations: int = 10
    target_accuracy: float = 0.96
    target_f1: float = 0.92
    data_dir: str = "/content/drive/MyDrive/"
    mri_shape: Tuple[int, int, int, int] = (10, 128, 128, 1)
    batch_size: int = 8  # 🟢 Smaller batch for better generalization
    epochs: int = 50  # 🟢 More epochs with early stopping
    learning_rate: float = 5e-4  # 🟢 Lower LR for stability
    memory_file: str = "/content/drive/MyDrive/agent_memory.json"


# ============================================================================
# AGENT MEMORY - Simple but Effective
# ============================================================================

class AgentMemory:
    """Simple persistent memory that ACTUALLY learns"""

    def __init__(self, memory_file: str = "/content/drive/MyDrive/agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.experiences = self._load()
        self.action_stats = {}  # action -> {success: int, fail: int, avg_improvement: float}
        self._compute_stats()
        print(f"[MEMORY] 🧠 Loaded {len(self.experiences)} past experiences")

    def _load(self) -> List[Dict]:
        if self.memory_file.exists():
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def _save(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.experiences[-100:], f, indent=2)  # Keep last 100

    def _compute_stats(self):
        """Compute statistics from experiences - THIS IS REAL LEARNING"""
        self.action_stats = {}
        for exp in self.experiences:
            action = exp.get("action", "unknown")
            improvement = exp.get("improvement", 0)

            if action not in self.action_stats:
                self.action_stats[action] = {"success": 0, "fail": 0, "total_improvement": 0, "count": 0}

            self.action_stats[action]["count"] += 1
            self.action_stats[action]["total_improvement"] += improvement

            if improvement > 0:
                self.action_stats[action]["success"] += 1
            else:
                self.action_stats[action]["fail"] += 1

    def save_experience(self, action: str, before: Dict, after: Dict):
        """Save and learn from experience"""
        improvement = after.get("f1", 0) - before.get("f1", 0)

        exp = {
            "action": action,
            "before": {"accuracy": before.get("accuracy", 0), "f1": before.get("f1", 0)},
            "after": {"accuracy": after.get("accuracy", 0), "f1": after.get("f1", 0)},
            "improvement": improvement,
            "timestamp": datetime.now().isoformat()
        }

        self.experiences.append(exp)
        self._save()
        self._compute_stats()

        status = "✅" if improvement > 0 else "❌"
        print(f"[MEMORY] {status} Saved: {action} → {improvement:+.2%}")

    def get_best_action(self, available_actions: List[str]) -> Tuple[str, str]:
        """Get best action based on REAL past experience - TRUE LEARNING"""
        if not self.action_stats:
            return available_actions[0], "No past experience, trying first action"

        # Calculate success rate and average improvement for each action
        scores = {}
        for action in available_actions:
            if action in self.action_stats:
                stats = self.action_stats[action]
                count = stats["count"]
                if count > 0:
                    success_rate = stats["success"] / count
                    avg_improvement = stats["total_improvement"] / count
                    # Score = success_rate * 0.5 + normalized_improvement * 0.5
                    scores[action] = success_rate * 0.5 + min(avg_improvement * 10, 1) * 0.5

        if scores:
            best_action = max(scores, key=scores.get)
            stats = self.action_stats[best_action]
            reason = f"Learned from {stats['count']} tries: {stats['success']} success, {stats['fail']} fail"
            return best_action, reason

        # Try untested action
        untested = [a for a in available_actions if a not in self.action_stats]
        if untested:
            return untested[0], "Exploring untested action"

        return available_actions[0], "Default action"

    def get_summary(self) -> str:
        """Get learning summary"""
        if not self.action_stats:
            return "No learning yet"

        lines = ["📊 Learning Summary:"]
        for action, stats in sorted(self.action_stats.items(),
                                   key=lambda x: x[1]["success"], reverse=True):
            rate = stats["success"] / stats["count"] * 100 if stats["count"] > 0 else 0
            avg = stats["total_improvement"] / stats["count"] * 100 if stats["count"] > 0 else 0
            lines.append(f"  • {action}: {rate:.0f}% success ({stats['count']} tries), avg: {avg:+.1f}%")

        return "\n".join(lines)


# ============================================================================
# LLM AGENT - Simple and Effective
# ============================================================================

class LLMAgent:
    """Simple LLM Agent with REAL learning from memory"""

    def __init__(self, config: Config):
        self.config = config
        self.memory = AgentMemory(config.memory_file)
        self.provider = self._setup_provider()

    def _setup_provider(self) -> str:
        """Setup LLM provider"""
        # Try OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key and len(api_key) > 20:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print("[LLM] ✅ Using OpenAI GPT")
                return "openai"
            except:
                pass

        # Try Ollama
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                print("[LLM] ✅ Using Ollama (local)")
                return "ollama"
        except:
            pass

        print("[LLM] ⚠️ No LLM available, using memory-based decisions")
        return "memory"

    def _call_llm(self, prompt: str) -> str:
        """Call LLM"""
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                return response.choices[0].message.content

            elif self.provider == "ollama":
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2", "prompt": prompt, "stream": False},
                    timeout=60
                )
                if r.status_code == 200:
                    return r.json().get("response", "")
        except Exception as e:
            print(f"[LLM] Error: {e}")

        return ""

    def decide(self, context: Dict, actions: List[str]) -> Dict:
        """Make decision using LLM + Memory"""

        # 1. Get memory recommendation (REAL LEARNING)
        memory_action, memory_reason = self.memory.get_best_action(actions)
        memory_summary = self.memory.get_summary()

        print(f"\n🤖 ═══════════════════════════════════════════════════════")
        print(f"🤖 [AGENT] Analyzing with learned experience...")
        print(memory_summary)
        print(f"🤖 ═══════════════════════════════════════════════════════\n")

        # 2. If no LLM, use memory directly
        if self.provider == "memory":
            print(f"[DECISION] 📚 Memory-based: {memory_action}")
            print(f"[REASON] {memory_reason}")
            return {"action": memory_action, "reasoning": memory_reason}

        # 3. Ask LLM with memory context
        prompt = f"""You are optimizing a 3D-CNN model for dyslexia detection.

Current Performance:
- Accuracy: {context.get('accuracy', 0):.2%} (target: 96%)
- F1-Score: {context.get('f1', 0):.2%} (target: 92%)
- Recall: {context.get('recall', 0):.2%}
- Precision: {context.get('precision', 0):.2%}
- Iteration: {context.get('iteration', 0)}/{self.config.max_iterations}

{memory_summary}

Memory recommends: "{memory_action}" because: {memory_reason}

Available actions: {actions}

Choose ONE action. Consider what memory learned. Respond in JSON:
{{"action": "action_name", "reasoning": "why this action"}}"""

        response = self._call_llm(prompt)

        # Parse response
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                action = result.get("action", memory_action)
                reasoning = result.get("reasoning", memory_reason)

                # Validate action
                if action not in actions:
                    action = memory_action

                print(f"[DECISION] 🤖 LLM: {action}")
                print(f"[REASON] {reasoning}")
                return {"action": action, "reasoning": reasoning}
        except:
            pass

        # Fallback to memory
        print(f"[DECISION] 📚 Fallback to memory: {memory_action}")
        return {"action": memory_action, "reasoning": memory_reason}

    def reflect(self, action: str, before: Dict, after: Dict):
        """Reflect and save to memory"""
        self.memory.save_experience(action, before, after)

        improvement = after.get("f1", 0) - before.get("f1", 0)
        if improvement > 0.01:
            print(f"[REFLECT] ✅ {action} worked! +{improvement:.2%}")
        elif improvement < -0.01:
            print(f"[REFLECT] ❌ {action} hurt performance: {improvement:.2%}")
        else:
            print(f"[REFLECT] ➖ {action} had minimal effect")


# ============================================================================
# DATA LOADER
# ============================================================================

class DataLoader:
    """Simple MRI data loader"""

    def __init__(self, config: Config):
        self.config = config
        self.shape = config.mri_shape

    def load_volume(self, filepath: str) -> Optional[np.ndarray]:
        """Load MRI volume - 10 evenly spaced slices"""
        try:
            import nibabel as nib
            img = nib.load(filepath)
            data = img.get_fdata()

            # Get 10 evenly spaced axial slices
            n_slices = data.shape[2]
            indices = np.linspace(0, n_slices - 1, 10, dtype=int)
            slices = data[:, :, indices]

            # Resize to 128x128
            zoom_factors = (128 / slices.shape[0], 128 / slices.shape[1], 1)
            slices = zoom(slices, zoom_factors, order=1)

            # Normalize
            slices = (slices - slices.min()) / (slices.max() - slices.min() + 1e-8)

            # Reshape to (10, 128, 128, 1)
            slices = np.transpose(slices, (2, 0, 1))
            slices = slices[..., np.newaxis]

            return slices.astype(np.float32)
        except Exception as e:
            return None

    def load_dataset(self) -> Dict:
        """Load complete dataset"""
        print("[DATA] Loading MRI dataset...")

        X, y = [], []

        # Dataset 1: ds003126_raw (DL + TD) - uses participants.tsv
        ds1 = os.path.join(self.config.data_dir, "ds003126_raw")
        if os.path.exists(ds1):
            participants_file = os.path.join(ds1, "participants.tsv")
            if os.path.exists(participants_file):
                with open(participants_file, 'r') as f:
                    lines = f.readlines()
                    header = lines[0].strip().split('\t')
                    group_idx = header.index('group') if 'group' in header else -1

                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        sub_id = parts[0]
                        group = parts[group_idx] if group_idx >= 0 else 'TD'

                        sub_path = os.path.join(ds1, sub_id)
                        if not os.path.exists(sub_path):
                            continue

                        # Look for T1w in anat folder (with or without session)
                        for session in ['ses-1', 'ses-2', '']:
                            session_path = os.path.join(sub_path, session) if session else sub_path
                            anat = os.path.join(session_path, "anat")

                            if os.path.exists(anat):
                                for f in sorted(os.listdir(anat)):
                                    if f.endswith(('.nii', '.nii.gz')) and 'T1w' in f:
                                        vol = self.load_volume(os.path.join(anat, f))
                                        if vol is not None:
                                            X.append(vol)
                                            y.append(1 if group == 'DL' else 0)
                                        break

                print(f"[DATA] Dataset 1: {len(X)} samples (DL={sum(y)}, TD={len(y)-sum(y)})")

        # Dataset 2: ds006239_raw (TD only)
        ds2 = os.path.join(self.config.data_dir, "ds006239_raw")
        if os.path.exists(ds2):
            count_before = len(X)
            for folder in sorted(os.listdir(ds2)):
                if not folder.startswith("sub-"):
                    continue

                sub_path = os.path.join(ds2, folder)

                for session in ['ses-1', 'ses-2', '']:
                    session_path = os.path.join(sub_path, session) if session else sub_path
                    anat = os.path.join(session_path, "anat")

                    if os.path.exists(anat):
                        for f in sorted(os.listdir(anat)):
                            if f.endswith(('.nii', '.nii.gz')) and 'T1w' in f:
                                vol = self.load_volume(os.path.join(anat, f))
                                if vol is not None:
                                    X.append(vol)
                                    y.append(0)  # TD only
                                break

            print(f"[DATA] Dataset 2: {len(X) - count_before} TD samples")

        X = np.array(X) if X else np.array([]).reshape(0, *self.shape)
        y = np.array(y) if y else np.array([])

        print(f"[DATA] TOTAL: {len(X)} samples (DL={np.sum(y==1)}, TD={np.sum(y==0)})")
        return {"X": X, "y": y}

    def augment(self, X: np.ndarray, y: np.ndarray, factor: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Augment minority class (DL) only"""
        if len(X) == 0:
            print("[AUGMENT] ⚠️ No data to augment!")
            return X, y

        dl_mask = (y == 1)
        X_dl, y_dl = X[dl_mask], y[dl_mask]
        X_td, y_td = X[~dl_mask], y[~dl_mask]

        if len(X_dl) == 0:
            print("[AUGMENT] ⚠️ No DL samples to augment!")
            return X, y

        X_aug_list = [X_dl]
        y_aug_list = [y_dl]

        for _ in range(factor - 1):
            X_new = []
            for vol in X_dl:
                aug = vol.copy()
                angle = np.random.uniform(-20, 20)
                for i in range(aug.shape[0]):
                    aug[i, :, :, 0] = rotate(aug[i, :, :, 0], angle, reshape=False, mode='nearest')
                    aug[i, :, :, 0] *= np.random.uniform(0.85, 1.15)
                X_new.append(aug)
            X_aug_list.append(np.array(X_new))
            y_aug_list.append(np.ones(len(X_new)))

        X_aug = np.vstack(X_aug_list)
        y_aug = np.hstack(y_aug_list)

        # Combine with TD
        X_final = np.vstack([X_td, X_aug])
        y_final = np.hstack([y_td, y_aug])

        # Shuffle
        idx = np.random.permutation(len(X_final))

        print(f"[AUGMENT] {len(X_final)} samples (DL={np.sum(y_final==1)}, TD={np.sum(y_final==0)})")
        return X_final[idx], y_final[idx]

    def prepare(self, X: np.ndarray, y: np.ndarray, aug_factor: int = 6) -> Dict:
        """Prepare data: augment and split"""
        if len(X) == 0:
            raise ValueError("❌ No data loaded! Check data_dir path and dataset structure.")

        X_aug, y_aug = self.augment(X, y, aug_factor)

        if len(X_aug) < 10:
            raise ValueError(f"❌ Not enough data after augmentation ({len(X_aug)} samples). Need at least 10.")

        X_train, X_temp, y_train, y_temp = train_test_split(
            X_aug, y_aug, test_size=0.3, stratify=y_aug, random_state=123
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=123
        )

        print(f"[SPLIT] Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test
        }


# ============================================================================
# MODEL BUILDER
# ============================================================================

def build_model(config: Config, params: Dict = None) -> keras.Model:
    """Build 3D-CNN model with anti-overfitting measures"""
    params = params or {}
    lr = params.get("learning_rate", config.learning_rate)
    dropout = params.get("dropout", 0.5)  # 🟢 Higher default dropout
    l2_reg = 0.01  # 🟢 L2 regularization

    inputs = keras.Input(shape=config.mri_shape)

    # 🟢 SIMPLER architecture to prevent overfitting
    # Conv block 1
    x = layers.Conv3D(16, 3, activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(dropout * 0.3)(x)  # 🟢 Spatial dropout for conv

    # Conv block 2
    x = layers.Conv3D(32, 3, activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(2)(x)
    x = layers.SpatialDropout3D(dropout * 0.5)(x)

    # Conv block 3 (smaller)
    x = layers.Conv3D(64, 3, activation='relu', padding='same',
                      kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    # 🟢 Simpler dense layers
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.02),  # 🟢 Higher weight decay
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    print(f"[MODEL] Built with {model.count_params():,} parameters (anti-overfitting)")
    return model


# ============================================================================
# MAIN AGENT
# ============================================================================

class MRIAgent:
    """Simple, effective agentic system"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.loader = DataLoader(self.config)
        self.agent = LLMAgent(self.config)

        self.actions = [
            "augment_more",      # Add more augmented data
            "reduce_lr",         # Reduce learning rate
            "increase_dropout",  # More regularization
            "train_longer",      # More epochs
            "adjust_weights"     # Change class weights
        ]

        self.params = {
            "learning_rate": self.config.learning_rate,
            "dropout": 0.3,
            "epochs": self.config.epochs,
            "class_weights": {0: 1.0, 1: 1.5}
        }

        self.best_f1 = 0
        self.best_model = None
        self.aug_factor = 6  # Starting augmentation factor
        self.raw_data = None  # Store original data for re-augmentation

    def execute_action(self, action: str, data: Dict) -> Dict:
        """Execute action - modifies params or data"""
        print(f"[ACTION] ⚡ Executing: {action}")

        if action == "augment_more":
            # Increase factor and re-prepare from ORIGINAL data (not cumulative)
            self.aug_factor = min(15, self.aug_factor + 2)
            print(f"  → Re-augmenting from original data with factor={self.aug_factor}")
            data = self.loader.prepare(self.raw_data["X"], self.raw_data["y"], self.aug_factor)

        elif action == "reduce_lr":
            self.params["learning_rate"] *= 0.5
            print(f"  → LR: {self.params['learning_rate']:.2e}")

        elif action == "increase_dropout":
            self.params["dropout"] = min(0.6, self.params["dropout"] + 0.1)
            print(f"  → Dropout: {self.params['dropout']}")

        elif action == "train_longer":
            self.params["epochs"] = min(100, self.params["epochs"] + 20)
            print(f"  → Epochs: {self.params['epochs']}")

        elif action == "adjust_weights":
            self.params["class_weights"] = {0: 1.0, 1: 3.0}
            print(f"  → Weights: {self.params['class_weights']}")

        return data

    def train(self, data: Dict) -> Tuple[Dict, keras.Model]:
        """Train model and return results"""
        model = build_model(self.config, self.params)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy', patience=20, restore_best_weights=True,
                min_delta=0.01  # 🟢 Need at least 1% improvement
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
            )
        ]

        print(f"[TRAIN] Epochs={self.params['epochs']}, LR={self.params['learning_rate']:.2e}")

        model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            epochs=self.params["epochs"],
            batch_size=self.config.batch_size,
            class_weight=self.params["class_weights"],
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        y_pred_prob = model.predict(data["X_val"]).flatten()

        # Find best threshold
        best_f1, best_thresh = 0, 0.5
        for thresh in np.arange(0.3, 0.7, 0.05):
            y_pred = (y_pred_prob >= thresh).astype(int)
            f1 = f1_score(data["y_val"], y_pred)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

        y_pred = (y_pred_prob >= best_thresh).astype(int)

        results = {
            "accuracy": accuracy_score(data["y_val"], y_pred),
            "f1": f1_score(data["y_val"], y_pred),
            "precision": precision_score(data["y_val"], y_pred, zero_division=0),
            "recall": recall_score(data["y_val"], y_pred, zero_division=0),
            "threshold": best_thresh
        }

        print(f"[RESULTS] Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}, "
              f"P={results['precision']:.2%}, R={results['recall']:.2%}")

        # Save best (overwrite single file)
        if results["f1"] > self.best_f1:
            self.best_f1 = results["f1"]
            self.best_model = model
            model.save("/content/drive/MyDrive/best_mri_model.h5")
            print(f"[BEST] 🏆 New best F1: {self.best_f1:.2%}")

        return results, model

    def run(self):

        # Load data
        raw = self.loader.load_dataset()
        self.raw_data = raw  # Store for re-augmentation
        data = self.loader.prepare(raw["X"], raw["y"], aug_factor=self.aug_factor)

        prev_results = {"accuracy": 0, "f1": 0}

        for iteration in range(self.config.max_iterations):
            print(f"\n{'='*60}")
            print(f"[ITERATION {iteration + 1}/{self.config.max_iterations}]")
            print(f"{'='*60}")

            # Train
            results, model = self.train(data)

            # Check if target reached
            if results["accuracy"] >= self.config.target_accuracy and \
               results["f1"] >= self.config.target_f1:
                print(f"\n🎉 TARGET ACHIEVED! Acc={results['accuracy']:.2%}, F1={results['f1']:.2%}")
                break

            # Agent decides next action
            context = {
                "accuracy": results["accuracy"],
                "f1": results["f1"],
                "precision": results["precision"],
                "recall": results["recall"],
                "iteration": iteration + 1
            }

            decision = self.agent.decide(context, self.actions)
            action = decision["action"]

            # Execute action
            data = self.execute_action(action, data)

            # Reflect and learn
            self.agent.reflect(action, prev_results, results)

            prev_results = results

        # Final results
        print(f"\n{'='*60}")
        print("[FINAL RESULTS]")
        print(f"{'='*60}")

        if self.best_model:
            y_pred_prob = self.best_model.predict(data["X_test"]).flatten()
            y_pred = (y_pred_prob >= 0.5).astype(int)

            print(f"Test Accuracy: {accuracy_score(data['y_test'], y_pred):.2%}")
            print(f"Test F1: {f1_score(data['y_test'], y_pred):.2%}")
            print(f"Test Precision: {precision_score(data['y_test'], y_pred, zero_division=0):.2%}")
            print(f"Test Recall: {recall_score(data['y_test'], y_pred, zero_division=0):.2%}")

            print("\nConfusion Matrix:")
            print(confusion_matrix(data["y_test"], y_pred))

        # Show what agent learned
        print(f"\n{self.agent.memory.get_summary()}")

        return self.best_model

if __name__ == "__main__":
    agent = MRIAgent()
    agent.run()

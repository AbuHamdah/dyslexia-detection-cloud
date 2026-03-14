"""
Agentic Optimizer — Unified Agent Memory + LLM Agent.
Merged from AgentMemory (codes/mri.py) + LLMAgent (codes/mri.py)
+ SimpleMemory (codes/fmri.py) + AgentMemory (codes/fusion.py).
"""

import os
import re
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# ════════════════════════════════════════════════════════════════════
# Agent Memory — persistent JSON-based learning from past experiences
# ════════════════════════════════════════════════════════════════════

class AgentMemory:
    """
    Persistent memory that learns from past optimisation decisions.
    Tracks which actions improved metrics and which did not.
    From codes/mri.py AgentMemory + codes/fmri.py SimpleMemory.
    """

    def __init__(self, memory_file: str = "agent_memory.json"):
        self.memory_file = Path(memory_file)
        self.experiences: List[Dict] = self._load()
        self.action_stats: Dict = {}
        self._compute_stats()

    # ── persistence ──

    def _load(self) -> List[Dict]:
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, "w") as f:
            json.dump(self.experiences[-100:], f, indent=2)  # keep last 100

    # ── learning ──

    def _compute_stats(self):
        self.action_stats = {}
        for exp in self.experiences:
            action = exp.get("action", "unknown")
            improvement = exp.get("improvement", 0)

            if action not in self.action_stats:
                self.action_stats[action] = {
                    "success": 0, "fail": 0,
                    "total_improvement": 0, "count": 0,
                }

            self.action_stats[action]["count"] += 1
            self.action_stats[action]["total_improvement"] += improvement
            if improvement > 0:
                self.action_stats[action]["success"] += 1
            else:
                self.action_stats[action]["fail"] += 1

    def save_experience(self, action: str, before: Dict, after: Dict):
        improvement = after.get("f1", 0) - before.get("f1", 0)
        self.experiences.append({
            "action": action,
            "before": {"accuracy": before.get("accuracy", 0), "f1": before.get("f1", 0)},
            "after": {"accuracy": after.get("accuracy", 0), "f1": after.get("f1", 0)},
            "improvement": improvement,
            "timestamp": datetime.now().isoformat(),
        })
        self._save()
        self._compute_stats()

    def add(self, action: str, f1_before: float, f1_after: float):
        """Shortcut used by fMRI / fusion training scripts."""
        self.save_experience(
            action,
            {"f1": f1_before},
            {"f1": f1_after},
        )

    # ── decision support ──

    def get_best_action(self, available_actions: List[str]) -> Tuple[str, str]:
        if not self.action_stats:
            return available_actions[0], "No past experience, trying first action"

        scores = {}
        for action in available_actions:
            if action in self.action_stats:
                stats = self.action_stats[action]
                count = stats["count"]
                if count > 0:
                    success_rate = stats["success"] / count
                    avg_improvement = stats["total_improvement"] / count
                    scores[action] = success_rate * 0.5 + min(avg_improvement * 10, 1) * 0.5

        if scores:
            best = max(scores, key=scores.get)
            s = self.action_stats[best]
            return best, f"Learned from {s['count']} tries: {s['success']} success, {s['fail']} fail"

        untested = [a for a in available_actions if a not in self.action_stats]
        if untested:
            return untested[0], "Exploring untested action"

        return available_actions[0], "Default action"

    def get_summary(self) -> str:
        if not self.action_stats:
            return "No learning yet"

        lines = []
        for action, stats in sorted(self.action_stats.items(),
                                    key=lambda x: x[1]["success"], reverse=True):
            rate = stats["success"] / stats["count"] * 100 if stats["count"] else 0
            avg = stats["total_improvement"] / stats["count"] * 100 if stats["count"] else 0
            lines.append(f"  {action}: {rate:.0f}% success ({stats['count']} tries), avg: {avg:+.1f}%")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════
# LLM Agent — OpenAI / Ollama / Memory fallback
# ════════════════════════════════════════════════════════════════════

class LLMAgent:
    """
    LLM-guided decision maker with memory fallback.
    From codes/mri.py LLMAgent (OpenAI + Ollama + memory).
    """

    def __init__(self, memory: AgentMemory, openai_key: str = ""):
        self.memory = memory
        self.provider = self._setup(openai_key)

    def _setup(self, openai_key: str) -> str:
        key = openai_key or os.environ.get("OPENAI_API_KEY", "")
        if key and len(key) > 20:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=key)
                return "openai"
            except Exception:
                pass

        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1)
            if r.status_code == 200:
                return "ollama"
        except Exception:
            pass

        print("[LLM] Using memory-based decisions (no LLM available)")
        return "memory"

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.provider == "openai":
                resp = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300, temperature=0.7,
                )
                return resp.choices[0].message.content

            if self.provider == "ollama":
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2", "prompt": prompt, "stream": False},
                    timeout=60,
                )
                if r.status_code == 200:
                    return r.json().get("response", "")
        except Exception:
            pass
        return ""

    # ── public API ──

    def decide(self, context: Dict, actions: List[str]) -> Dict:
        """Choose next optimisation action using LLM + memory."""
        memory_action, memory_reason = self.memory.get_best_action(actions)
        summary = self.memory.get_summary()

        if self.provider == "memory":
            return {"action": memory_action, "reasoning": memory_reason}

        prompt = f"""You are optimizing a deep learning model for dyslexia detection.

Current Performance:
- Accuracy: {context.get('accuracy', 0):.2%}
- F1-Score: {context.get('f1', 0):.2%}
- Recall:   {context.get('recall', 0):.2%}
- Precision:{context.get('precision', 0):.2%}
- Iteration:{context.get('iteration', 0)}

Learning summary:
{summary}

Memory recommends: "{memory_action}" — {memory_reason}

Available actions: {actions}

Choose ONE action. Respond in JSON: {{"action": "name", "reasoning": "why"}}"""

        response = self._call_llm(prompt)
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                result = json.loads(match.group())
                action = result.get("action", memory_action)
                if action not in actions:
                    action = memory_action
                return {"action": action, "reasoning": result.get("reasoning", "")}
        except Exception:
            pass

        return {"action": memory_action, "reasoning": memory_reason}

    def reflect(self, action: str, before: Dict, after: Dict):
        self.memory.save_experience(action, before, after)

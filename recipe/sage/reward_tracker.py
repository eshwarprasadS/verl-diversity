"""
Reward tracking module for monitoring data point rewards across training steps.
"""

import json
import os
from collections import defaultdict
from typing import Dict
import torch


class RewardTracker:
    """Track rewards for each unique data point across training steps.

    This class maintains a history of rewards for each data point identified by its index,
    allowing analysis of reward progression and identification of consistently zero-reward samples.
    """

    def __init__(self):
        """Initialize the reward tracker with empty storage."""
        # Store full history: index -> list of (step, reward) tuples
        self.index_to_reward_history: Dict[str, list] = defaultdict(list)

        # Store generated hints: index -> list of dicts(step, level, hint, used)
        self.index_to_hint_history: Dict[str, list] = defaultdict(list)
        # Store raw hint text
        self.index_to_hint_raw_history: Dict[str, list] = defaultdict(list)

        # Store the next-iteration hint level for each index
        self.index_to_hint_level: Dict[str, str] = {}
        # Store accuracy per index per level: index -> list of dicts(step, level, accuracy)
        self.index_to_hint_accuracy_history: Dict[str, list] = defaultdict(list)
        # Store last-iteration accuracy for each index
        self.index_to_last_hint_accuracy: Dict[str, float] = {}
        # Store last successful hint payloads for fallback: index -> {"level_1": ..., ...}
        self.index_to_last_hint_payloads: Dict[str, dict] = {}

    def update(self, batch, global_step: int):
        """Update reward tracking with a new batch of data.

        Args:
            batch: DataProto containing 'index' in non_tensor_batch and rewards in batch
            global_step: Current training step number
        """
        if "index" not in batch.non_tensor_batch:
            return

        indexes = batch.non_tensor_batch["index"]

        # Extract rewards - sum token-level rewards to get sequence-level reward
        if "token_level_rewards" in batch.batch:
            rewards = batch.batch["token_level_rewards"].sum(dim=-1)
        elif "token_level_scores" in batch.batch:
            rewards = batch.batch["token_level_scores"].sum(dim=-1)
        else:
            return

        if torch.is_tensor(rewards):
            rewards = rewards.detach().cpu().numpy()

        for index, reward in zip(indexes, rewards):
            index_str = str(index)
            reward_float = float(reward)
            self.index_to_reward_history[index_str].append((global_step, reward_float))

    def log_hint_payloads(self, index_array, hint_payloads: dict, global_step: int, used: bool, failed: bool = False):
        """Log generated hints for given indices."""
        if index_array is None:
            return
        for batch_idx, payload in hint_payloads.items():
            if batch_idx >= len(index_array):
                continue
            index_str = str(index_array[batch_idx])
            for level_key in ("level_1", "level_2", "level_3"):
                hint_text = payload.get(level_key)
                if hint_text and str(hint_text).strip():
                    self.index_to_hint_history[index_str].append(
                        {
                            "step": global_step,
                            "level": level_key,
                            "hint": str(hint_text).strip(),
                            "used": used,
                            "failed": failed,
                        }
                    )

    def log_hint_raw(self, index_array, raw_hints: dict, global_step: int, used: bool, failed: bool = False):
        """Log raw hint strings."""
        if index_array is None:
            return
        for batch_idx, raw in raw_hints.items():
            if batch_idx >= len(index_array):
                continue
            index_str = str(index_array[batch_idx])
            self.index_to_hint_raw_history[index_str].append(
                {
                    "step": global_step,
                    "raw": str(raw),
                    "used": used,
                    "failed": failed,
                }
            )

    def log_hint_accuracy(self, index_array, levels_by_batch_idx: dict, accuracies, global_step: int):
        """Log per-prompt accuracy for the hint setting used in the current iteration."""
        if index_array is None:
            return
        for batch_idx, acc in enumerate(accuracies):
            if batch_idx >= len(index_array):
                continue
            index_str = str(index_array[batch_idx])
            level = levels_by_batch_idx.get(batch_idx, "no_hint")
            self.index_to_hint_accuracy_history[index_str].append(
                {
                    "step": global_step,
                    "level": level,
                    "accuracy": float(acc),
                }
            )

    def get_hint_level(self, index_str: str, default: str = "no_hint") -> str:
        return self.index_to_hint_level.get(index_str, default)

    def set_hint_level(self, index_str: str, level: str):
        self.index_to_hint_level[index_str] = level

    def get_last_hint_accuracy(self, index_str: str):
        return self.index_to_last_hint_accuracy.get(index_str)

    def set_last_hint_accuracy(self, index_str: str, accuracy: float):
        self.index_to_last_hint_accuracy[index_str] = float(accuracy)

    def get_last_hint_payload(self, index_str: str):
        return self.index_to_last_hint_payloads.get(index_str)

    def set_last_hint_payload(self, index_str: str, payload: dict):
        if payload:
            self.index_to_last_hint_payloads[index_str] = payload

    def get_zero_reward_stats(self) -> Dict[str, float]:
        """Calculate statistics about data points with zero rewards."""
        total_indexes = len(self.index_to_reward_history)

        if total_indexes == 0:
            return {
                "reward_tracking/num_total_indexes": 0,
            }

        reward_chunks = [[] for _ in range(7)]
        intervals = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for index, rewards in self.index_to_reward_history.items():
            avg_reward = sum(r for _, r in rewards) / len(rewards) if rewards else 0.0

            if avg_reward == 0.0:
                reward_chunks[0].append(index)
            elif avg_reward == 1.0:
                reward_chunks[-1].append(index)
            else:
                for i in range(1, len(intervals)):
                    if intervals[i - 1] < avg_reward <= intervals[i]:
                        reward_chunks[i].append(index)
                        break

        interval_labels = [
            "0",
            "(0.0,0.2]",
            "(0.2,0.4]",
            "(0.4,0.6]",
            "(0.6,0.8]",
            "(0.8,1.0)",
            "1",
        ]
        results = {
            "reward_tracking/num_total_indexes": total_indexes,
        }
        for label, chunk in zip(interval_labels, reward_chunks):
            results[f"reward_tracking/pct_indexes_in_{label}"] = len(chunk) / total_indexes

        return results

    def save_checkpoint(self, checkpoint_dir: str):
        """Save reward tracking state to disk."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, "reward_tracker.json")

        data = {
            "index_to_reward_history": {
                index: history for index, history in self.index_to_reward_history.items()
            },
            "index_to_hint_history": {
                index: history for index, history in self.index_to_hint_history.items()
            },
            "index_to_hint_raw_history": {
                index: history for index, history in self.index_to_hint_raw_history.items()
            },
            "index_to_hint_level": self.index_to_hint_level,
            "index_to_last_hint_accuracy": self.index_to_last_hint_accuracy,
            "index_to_last_hint_payloads": self.index_to_last_hint_payloads,
            "index_to_hint_accuracy_history": {
                index: history for index, history in self.index_to_hint_accuracy_history.items()
            },
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved reward tracker to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_dir: str) -> bool:
        """Load reward tracking state from disk."""
        checkpoint_path = os.path.join(checkpoint_dir, "reward_tracker.json")

        if not os.path.exists(checkpoint_path):
            print(f"No reward tracker checkpoint found at {checkpoint_path}")
            return False

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        self.index_to_reward_history = defaultdict(list, data["index_to_reward_history"])
        self.index_to_hint_history = defaultdict(list, data.get("index_to_hint_history", {}))
        self.index_to_hint_raw_history = defaultdict(list, data.get("index_to_hint_raw_history", {}))
        self.index_to_hint_level = data.get("index_to_hint_level", {})
        self.index_to_last_hint_accuracy = data.get("index_to_last_hint_accuracy", {})
        self.index_to_last_hint_payloads = data.get("index_to_last_hint_payloads", {})
        self.index_to_hint_accuracy_history = defaultdict(list, data.get("index_to_hint_accuracy_history", {}))

        num_zero_reward = 0
        for k, v in self.index_to_reward_history.items():
            if all(reward == 0.0 for step, reward in v):
                num_zero_reward += 1

        print(f"Loaded reward tracker from {checkpoint_path}")
        print(f"  Total indexes tracked: {len(self.index_to_reward_history)}")
        print(f"  Indexes with zero reward history: {num_zero_reward}")

        return True

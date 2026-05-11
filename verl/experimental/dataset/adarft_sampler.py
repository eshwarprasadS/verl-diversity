"""AdaRFT curriculum sampler for verl.

Adaptive Curriculum Reinforcement Finetuning (Shi et al., 2025).
Dynamically adjusts target difficulty based on recent reward signals.

Requires: dataset must have 'difficulty' in extra_info (precomputed, float 0-100).

Config parameters (passed via data_config):
    adarft_beta (float): Target success rate equilibrium. Default 0.5.
    adarft_alpha (float): Sensitivity of tanh response. Default 2.0.
    adarft_eta (float): Step size scaling reward to difficulty space. Default 50.0.
    adarft_d_min (float): Minimum target difficulty. Default 0.0.
    adarft_d_max (float): Maximum target difficulty. Default 100.0.
    adarft_initial_target (float): Starting target difficulty. Default 50.0.
"""

import math
from collections.abc import Sized

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class AdaRFTSampler(AbstractCurriculumSampler):

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.batch_size = getattr(data_config, "train_batch_size", 128)

        self.beta = getattr(data_config, "adarft_beta", 0.5)
        self.alpha = getattr(data_config, "adarft_alpha", 2.0)
        self.eta = getattr(data_config, "adarft_eta", 50.0)
        self.d_min = getattr(data_config, "adarft_d_min", 0.0)
        self.d_max = getattr(data_config, "adarft_d_max", 100.0)
        self.target_difficulty = getattr(data_config, "adarft_initial_target", 50.0)

        self.difficulties = self._extract_difficulties()
        self.epoch = 0

    def _extract_difficulties(self) -> np.ndarray:
        """Extract difficulty scores from the dataset."""
        difficulties = []
        for i in range(len(self.data_source)):
            item = self.data_source[i]
            extra = item.get("extra_info", {})
            if isinstance(extra, str):
                import json
                extra = json.loads(extra)
            diff = extra.get("difficulty", 50.0)
            difficulties.append(float(diff))
        return np.array(difficulties)

    def __iter__(self):
        distances = np.abs(self.difficulties - self.target_difficulty)
        indices = np.argsort(distances)[: self.batch_size]
        np.random.shuffle(indices)
        yield from indices.tolist()

    def __len__(self):
        return self.batch_size

    def update(self, batch: DataProto) -> None:
        """Update target difficulty based on batch reward signal."""
        if "token_level_scores" in batch.batch:
            rewards = batch.batch["token_level_scores"].sum(dim=-1).float()
        elif "rewards" in batch.batch:
            rewards = batch.batch["rewards"].float()
        else:
            return

        mean_reward = rewards.mean().item()
        delta = self.eta * math.tanh(self.alpha * (mean_reward - self.beta))
        self.target_difficulty = max(
            self.d_min, min(self.d_max, self.target_difficulty + delta)
        )
        self.epoch += 1

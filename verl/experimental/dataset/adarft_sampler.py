"""AdaRFT curriculum sampler for verl.

Faithful implementation of Algorithm 1 from:
  Shi et al., 2025 — "Efficient Reinforcement Finetuning via
  Adaptive Curriculum Learning" (arXiv:2504.05520)

At every training step, selects the B problems closest to the current
target difficulty T. After the policy update, T is adjusted based on
the batch's mean reward.

Requires: dataset must have 'difficulty' in extra_info (precomputed, float 0-100).
"""

import json
import math
from collections.abc import Sized

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import Sampler

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler


class AdaRFTSampler(AbstractCurriculumSampler):

    # Paper's exact hyperparameters (Section 4.3, arXiv:2504.05520)
    BETA = 0.5   # target success rate equilibrium
    ALPHA = 2.0  # tanh sensitivity
    ETA = 50.0   # step size (maps reward delta to difficulty-space delta)
    D_MIN = 0.0
    D_MAX = 100.0
    T_INIT = 0.0  # start at easiest problems

    def __init__(self, data_source: Sized, data_config: DictConfig):
        self.data_source = data_source
        self.batch_size = getattr(data_config, "train_batch_size", 128)
        self.target_difficulty = self.T_INIT
        self.difficulties = self._extract_difficulties()
        self._step = 0

    def _extract_difficulties(self) -> np.ndarray:
        difficulties = []
        for i in range(len(self.data_source)):
            item = self.data_source[i]
            extra = item.get("extra_info", {})
            if isinstance(extra, str):
                extra = json.loads(extra)
            diff = extra.get("difficulty", 50.0)
            difficulties.append(float(diff))
        return np.array(difficulties)

    def __iter__(self):
        # Per-step re-selection: yield the B closest problems to T.
        # Between batches, update() adjusts T, so the next batch uses
        # the updated target. This loop runs indefinitely; the training
        # loop stops via trainer.total_training_steps.
        while True:
            distances = np.abs(self.difficulties - self.target_difficulty)
            top_b = np.argsort(distances)[:self.batch_size]
            yield from top_b.tolist()

    def __len__(self):
        return len(self.data_source)

    def update(self, batch: DataProto) -> None:
        # Equation from Section 3.2:
        #   T' = clip(T + eta * tanh(alpha * (R_avg - beta)), d_min, d_max)
        # Called AFTER the policy update (step 6 in Algorithm 1).
        if "token_level_scores" in batch.batch:
            rewards = batch.batch["token_level_scores"].sum(dim=-1).float()
        elif "rewards" in batch.batch:
            rewards = batch.batch["rewards"].float()
        else:
            return

        mean_reward = rewards.mean().item()
        delta = self.ETA * math.tanh(self.ALPHA * (mean_reward - self.BETA))
        self.target_difficulty = max(
            self.D_MIN, min(self.D_MAX, self.target_difficulty + delta)
        )
        self._step += 1

import logging
from pathlib import Path

import numpy as np


class AlignedRewardSet:
    def __init__(self, path: Path, true_reward: np.ndarray) -> None:
        self.diffs = np.load(path)

        assert np.all(true_reward @ self.diffs.T >= 0)
        self.diffs = self.diffs[true_reward @ self.diffs.T > 1e-16]
        assert np.all(true_reward @ self.diffs.T > 1e-16)

        self.true_reward = true_reward

    def prob_aligned(self, rewards: np.ndarray, densities: np.ndarray) -> np.ndarray:
        assert (
            densities.shape[0] == rewards.shape[0]
        ), f"Unequal amount of densities and rewards {densities.shape=}, {rewards.shape=}"
        assert len(densities.shape) <= 2

        aligned_reward_indices = np.all((rewards @ self.diffs.T) > 0, axis=1)
        prob_aligned = np.sum(densities[aligned_reward_indices], axis=0)
        if np.sum(aligned_reward_indices) == 0:
            logging.warning("No aligned rewards")
        elif np.allclose(prob_aligned, 0, atol=0.001):
            logging.debug("There are some aligned rewards, but all likelihoods are 0")

        return prob_aligned

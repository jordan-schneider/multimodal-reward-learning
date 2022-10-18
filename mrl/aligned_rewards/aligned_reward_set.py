import logging
from pathlib import Path

import numpy as np


class AlignedRewardSet:
    def __init__(
        self, diffs: np.ndarray, true_reward: np.ndarray, epsilon: float = 1e-16
    ) -> None:
        """The space of rewards which agree with a given ground-truth reward on all preferences over linear features.

        Args:
            diffs (np.ndarray): Differences in reward features between two objects e.g. states, trajectories, actions.
            true_reward (np.ndarray): The ground truth reward the reward set agrees with.
            epsilon (float, optional): A conservative preference tolerance. The ground truth reward/return of an object must be at least this much larger than the other object to count as agreeing. Defaults to 1e-16.
        """
        self.diffs = diffs
        self.epsilon = epsilon

        assert np.all(true_reward @ self.diffs.T >= 0)
        self.diffs = self.diffs[true_reward @ self.diffs.T > self.epsilon]
        assert np.all(true_reward @ self.diffs.T > self.epsilon)

        self.true_reward = true_reward

    def prob_aligned(self, rewards: np.ndarray, densities: np.ndarray) -> np.ndarray:
        """The odds that a reward sampled according to some density is aligned with the ground truth reward.

        Args:
            rewards (np.ndarray): The set of reward samples that forms the support of the didstribution.
            densities (np.ndarray): The density assigned to each reward sample.

        Returns:
            np.ndarray: _description_
        """
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

        import pdb

        pdb.set_trace()

        return prob_aligned

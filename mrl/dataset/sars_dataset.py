from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

# TODO: Fix this the next time I do policy or value learning.


class SarsDataset:
    data: Dict[str, np.ndarray]  # type: ignore

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        firsts: np.ndarray,
    ) -> None:

        if len(self) == 0 and len(states) > 0:
            logging.warning("Every timestep is first")
            self.index_map = np.empty(0, dtype=int)
        else:
            self.index_map = self._compute_index_map(firsts)

    def _compute_index_map(self, firsts: np.ndarray) -> np.ndarray:
        index_map = np.empty(len(self), dtype=int)
        j = 0
        for i in range(len(self)):
            while firsts[j + 1]:
                j += 1
            index_map[i] = j
            assert j < firsts.shape[0]
            j += 1

        return index_map

    def __len__(self) -> int:
        assert self.data["states"] is not None and self.data["firsts"] is not None
        # There are (n-1) possible transitions from n states. Each time an episode resets, we don't know the
        # terminal states, so we subtract out that many transitions. There are sum(firsts) - 1 resets
        # (the first reset doesn't count. (n-1)-(sum(f)-1) = n-sum(f)
        return len(self.data["states"]) - np.sum(self.data["firsts"])

    def __getitem__(
        self, i: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        j = self.index_map[i]
        return (
            self.data["states"][j],
            self.data["actions"][j],
            self.data["rewards"][j],
            self.data["states"][j + 1],
        )

    def make_sars(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = self.data["states"][self.index_map]
        actions = self.data["actions"][self.index_map]
        rewards = self.data["rewards"][self.index_map]
        next_states = self.data["states"][self.index_map + 1]

        return states, actions, rewards, next_states

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> SarsDataset:
        """Builds RLDataset from Roller output dict"""
        states = data["ob"].reshape(
            (-1, *data["ob"].shape[2:])
        )  # (num, T, H, W, C) -> (num * T, H, W, C)
        actions = data["ac"].flatten()  # (num, T) -> (num * T)
        rewards = data["reward"].flatten()  # (num, T) -> (num * T)
        firsts = data["first"].flatten()  # (num, T) -> (num * T)

        return cls(states=states, actions=actions, rewards=rewards, firsts=firsts)

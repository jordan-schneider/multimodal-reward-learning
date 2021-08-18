from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import torch


class SarsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        super().__init__()
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

        self.index_map = self._compute_index_map(self.dones)

    def _compute_index_map(self, dones: torch.Tensor) -> np.ndarray:
        index_map = np.empty((len(self)), dtype=np.uint64)
        j = 0
        for i in range(len(self)):
            if dones[j]:
                j += 1
            index_map[i] = j
            j += 1

        return index_map

    def __len__(self) -> int:
        # Each time we end an episode, the final state cannot be used
        return len(self.states) - np.sum(self.dones) - 1

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.index_map[i]
        return self.states[j], self.actions[j], self.rewards[j], self.states[j + 1]

    @classmethod
    def from_dict(cls, data: dict) -> SarsDataset:
        states = cast(torch.Tensor, data["ob"]).flatten(
            0, 1
        )  # (num, T, H, W, C) -> (num * T, H, W, C)
        actions = cast(torch.Tensor, data["ac"]).flatten()  # (num, T) -> (num * T)
        rewards = cast(torch.Tensor, data["reward"]).flatten()  # (num, T) -> (num * T)
        firsts = cast(torch.Tensor, data["first"]).flatten()  # (num, T) -> (num * T)
        dones = np.concatenate(firsts[1:], [False])

        return cls(states, actions, rewards, dones)

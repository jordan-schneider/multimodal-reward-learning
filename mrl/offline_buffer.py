from __future__ import annotations

from typing import Generic, Tuple, TypeVar

import numpy as np
import torch


class SarsDataset(torch.utils.data.Dataset):
    def __init__(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor) -> None:
        super().__init__()
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def __len__(self) -> int:
        return len(self.states) - 1

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[i], self.actions[i], self.rewards[i], self.states[i + 1]

    @staticmethod
    def from_dict(data: dict) -> SarsDataset:
        # TODO: Figure out what's going on with this dict
        print(data.keys())
        pass

from __future__ import annotations

from typing import Generator, Iterator, Tuple, cast

import numpy as np
import torch


class RLDataset:
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones

    @classmethod
    def from_gym3(
        cls, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> RLDataset:
        """Builds RLDataset from procgen_rollout output arrays"""
        # (T, num, H, W, C) -> (num * T, H, W, C)
        flat_states = torch.tensor(states).flatten(0, 1)
        flat_actions = torch.tensor(actions).flatten()  # (num, T) -> (num * T)
        flat_rewards = torch.tensor(rewards).flatten()  # (num, T) -> (num * T)
        flat_firsts = torch.tensor(firsts).flatten()  # (num, T) -> (num * T)
        dones = flat_firsts[1:]

        return cls(flat_states, flat_actions, flat_rewards, dones)

    @classmethod
    def from_dict(cls, data: dict) -> RLDataset:
        """Builds RLDataset from Roller output dict"""
        states = cast(torch.Tensor, data["ob"]).flatten(
            0, 1
        )  # (num, T, H, W, C) -> (num * T, H, W, C)
        actions = cast(torch.Tensor, data["ac"]).flatten()  # (num, T) -> (num * T)
        rewards = cast(torch.Tensor, data["reward"]).flatten()  # (num, T) -> (num * T)
        firsts = cast(torch.Tensor, data["first"]).flatten()  # (num, T) -> (num * T)
        dones = firsts[1:]

        return cls(states, actions, rewards, dones)

    def trajs(
        self, include_incomplete: bool = True
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        done_indices = self.dones.nonzero(as_tuple=True)[0] + 1

        start = 0
        for done_index in done_indices:
            yield self.states[start:done_index], self.actions[start : done_index - 1], self.rewards[
                start:done_index
            ]
            start = done_index

        if include_incomplete:
            yield self.states[start:], self.actions[start:], self.rewards[start:]


class SarsDataset(torch.utils.data.Dataset, RLDataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        super().__init__(states, actions, rewards, dones)

        self.index_map = self._compute_index_map(self.dones)

    def _compute_index_map(self, dones: torch.Tensor) -> np.ndarray:
        index_map = np.empty((len(self)), dtype=int)
        j = 0
        for i in range(len(self)):
            if dones[j]:
                j += 1
            index_map[i] = j
            j += 1

        return index_map

    def __len__(self) -> int:
        # Each time we end an episode, the final state cannot be used
        return len(self.states) - torch.sum(self.dones) - 1

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.index_map[i]
        return self.states[j], self.actions[j], self.rewards[j], self.states[j + 1]

    @classmethod
    def from_rl_dataset(cls, data: RLDataset) -> SarsDataset:
        return cls(data.states, data.actions, data.rewards, data.dones)

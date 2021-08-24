from __future__ import annotations

import logging
from typing import Generator, List, Tuple, cast

import numpy as np
import torch
from torch.functional import Tensor


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

        self.states.requires_grad = False
        self.actions.requires_grad = False
        self.rewards.requires_grad = False
        self.dones.requires_grad = False

    @staticmethod
    def process_gym3(
        states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # (T, num, H, W, C) -> (num * T, H, W, C)
        flat_states = torch.tensor(states).flatten(0, 1)
        flat_actions = torch.tensor(actions).flatten()  # (num, T) -> (num * T)
        flat_rewards = torch.tensor(rewards).flatten()  # (num, T) -> (num * T)
        flat_firsts = torch.tensor(firsts).flatten()  # (num, T) -> (num * T)
        dones = flat_firsts[1:]
        return flat_states, flat_actions, flat_rewards, dones

    @classmethod
    def from_gym3(
        cls, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> RLDataset:
        """Builds RLDataset from procgen_rollout output arrays"""
        return cls(*RLDataset.process_gym3(states, actions, rewards, firsts))

    def append_gym3(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> RLDataset:
        s, a, r, d = self.process_gym3(states, actions, rewards, firsts)
        self.states = torch.cat((self.states, s), dim=0)
        self.actions = torch.cat((self.actions, a))
        self.rewards = torch.cat((self.rewards, r))
        self.dones = torch.cat((self.dones, d))
        return self

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

    def truncated_returns(
        self, horizon: int, discount_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert discount_rate >= 0.0 and discount_rate <= 1.0

        discounts = torch.pow(
            torch.ones(horizon, dtype=self.rewards.dtype) * discount_rate, torch.arange(horizon)
        )

        done_indices = self.dones.nonzero(as_tuple=True)[0] + 1
        returns: List[torch.Tensor] = []
        start_index = 0
        for done_index in done_indices:
            logging.debug(
                f"done_index={done_index}, start_index={start_index}, n_returns={len(returns)}"
            )
            while start_index < done_index:
                reward_batch = self.rewards[start_index : min(start_index + horizon, done_index)]
                assert (
                    reward_batch.dtype == discounts.dtype
                ), f"dtype mismatch, reward_batch={reward_batch.dtype}, discounts={discounts.dtype}"
                returns.append(reward_batch @ discounts[: len(reward_batch)])
                start_index += 1

        logging.debug(f"start_index={start_index}, n_returns={len(returns)}")

        if len(done_indices) > 0:
            assert (
                start_index == done_indices.max(dim=0)[0]
            ), f"final start_index={start_index} but max done_index={done_indices.max(dim=0)[0]}"

        # After the last done, we don't want to use any fewer than k rewards, because the episode
        # hasn't actually ended, the k-step return can't be computed.
        for i in range(start_index, len(self.rewards) - horizon):
            reward_batch = self.rewards[i : i + horizon]
            assert (
                reward_batch.dtype == discounts.dtype
            ), f"dtype mismatch, reward_batch={reward_batch.dtype}, discounts={discounts.dtype}"
            returns.append(reward_batch @ discounts)

        last_index = max(start_index, len(self.rewards) - horizon)
        states, actions = self.states[:last_index], self.actions[:last_index]

        assert len(states) == len(returns), f"{len(states)} states but {len(returns)} returns"

        returns_torch = torch.stack(returns)

        return states, actions, returns_torch


class SarsDataset(RLDataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        super().__init__(states, actions, rewards, dones)

        if len(self) < 0:
            raise ValueError("dones cannot be all True")

        self.index_map = self._compute_index_map(self.dones)

    def _compute_index_map(self, dones: torch.Tensor) -> np.ndarray:
        index_map = np.empty((len(self)), dtype=int)
        j = 0
        for i in range(len(self)):
            while dones[j]:
                j += 1
            index_map[i] = j
            j += 1

        return index_map

    def __len__(self) -> int:
        # Each time we end an episode, the final state cannot be used
        return len(self.states) - torch.sum(self.dones) - 1

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.index_map[i]
        logging.debug(f"Mapping {i} to {j}")
        return self.states[j], self.actions[j], self.rewards[j], self.states[j + 1]

    def make_sars(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The torch index typing doesn't handle numpy arrays for some reason.
        states = self.states[self.index_map]  # type: ignore
        actions = self.actions[self.index_map]  # type: ignore
        rewards = self.rewards[self.index_map]  # type: ignore
        next_states = self.states[self.index_map + 1]  # type: ignore

        return states, actions, rewards, next_states

    @classmethod
    def from_rl_dataset(cls, data: RLDataset) -> SarsDataset:
        return cls(data.states, data.actions, data.rewards, data.dones)

    @classmethod
    def from_gym3(
        cls, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> SarsDataset:
        return cls(*RLDataset.process_gym3(states, actions, rewards, firsts))

    def append_gym3(
        self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, firsts: np.ndarray
    ) -> SarsDataset:
        """This function exists entirely to fix the return type."""
        return cast(SarsDataset, super().append_gym3(states, actions, rewards, firsts))

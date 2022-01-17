from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple, cast

import numpy as np
import torch


class RlDataset:
    def __init__(
        self,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.dones = dones
        self.features = features

    @staticmethod
    def process_gym3(
        *,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> Dict[str, torch.Tensor]:
        out = {}

        if states is not None:
            # (T, num, H, W, C) -> (num * T, H, W, C)
            out["states"] = torch.tensor(states).flatten(0, 1)
        if actions is not None:
            out["actions"] = torch.tensor(actions).flatten()  # (num, T) -> (num * T)
        if rewards is not None:
            out["rewards"] = torch.tensor(rewards).flatten()  # (num, T) -> (num * T)
        if firsts is not None:
            flat_firsts = torch.tensor(firsts).flatten()  # (num, T) -> (num * T)
            out["dones"] = flat_firsts[1:]
        if features is not None:
            out["features"] = torch.tensor(features).flatten(0, 1)

        return out

    def __len__(self) -> int:
        if self.states is not None:
            return self.states.shape[0]
        if self.actions is not None:
            return self.actions.shape[0] + 1
        if self.rewards is not None:
            return self.rewards.shape[0]
        if self.dones is not None:
            return self.dones.shape[0] + 1
        if self.features is not None:
            return self.features.shape[0]
        raise ValueError("No data in dataset.")

    @classmethod
    def from_gym3(
        cls,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> RlDataset:
        """Builds RLDataset from procgen_rollout output arrays"""
        return cls(
            **RlDataset.process_gym3(
                states=states,
                actions=actions,
                rewards=rewards,
                firsts=firsts,
                features=features,
            )
        )

    def append_gym3(
        self,
        *,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> RlDataset:
        data = RlDataset.process_gym3(
            states=states,
            actions=actions,
            rewards=rewards,
            firsts=firsts,
            features=features,
        )

        if self.states is not None and data["states"] is not None:
            self.states = torch.cat((self.states, data["states"]), dim=0)
        if self.actions is not None and data["actions"] is not None:
            self.actions = torch.cat((self.actions, data["actions"]))
        if self.rewards is not None and data["rewards"] is not None:
            self.rewards = torch.cat((self.rewards, data["rewards"]))
        if self.dones is not None and data["dones"] is not None:
            self.dones = torch.cat((self.dones, data["dones"]))
        if self.features is not None and data["features"] is not None:
            self.features = torch.cat((self.features, data["features"]), dim=0)
        return self

    @classmethod
    def from_dict(cls, data: dict) -> RlDataset:
        """Builds RLDataset from Roller output dict"""
        states = cast(torch.Tensor, data["ob"]).flatten(
            0, 1
        )  # (num, T, H, W, C) -> (num * T, H, W, C)
        actions = cast(torch.Tensor, data["ac"]).flatten()  # (num, T) -> (num * T)
        rewards = cast(torch.Tensor, data["reward"]).flatten()  # (num, T) -> (num * T)
        firsts = cast(torch.Tensor, data["first"]).flatten()  # (num, T) -> (num * T)
        dones = firsts[1:]

        return cls(states=states, actions=actions, rewards=rewards, dones=dones)

    def get_bytes(self) -> int:
        def get_tensor_bytes(t: Optional[torch.Tensor]) -> int:
            if t is None:
                return 0
            else:
                return t.numel() * t.element_size()

        return (
            get_tensor_bytes(self.states)
            + get_tensor_bytes(self.actions)
            + get_tensor_bytes(self.rewards)
            + get_tensor_bytes(self.dones)
            + get_tensor_bytes(self.features)
        )

    @dataclass
    class Traj:
        states: Optional[torch.Tensor] = None
        features: Optional[torch.Tensor] = None
        actions: Optional[torch.Tensor] = None
        rewards: Optional[torch.Tensor] = None

    def trajs(
        self,
        *,
        include_incomplete: bool = False,
    ) -> Generator[Traj, None, None]:
        assert self.dones is not None, "Must supply dones to get trajs."

        if len(self.dones) == 0:
            # Only one timestep provided
            yield self.Traj(
                states=self.states, actions=self.actions, rewards=self.rewards
            )
            return

        # We want to include the state for which done=True, and slicing logic requires that we add an additional one
        # We've also trimmed the first timestep from dones, so we net only add one. See test_trajs()
        # in test_dataset.py for proof.
        done_indices = self.dones.nonzero(as_tuple=True)[0] + 1

        start = 0
        for done_index in done_indices:
            yield RlDataset.Traj(
                states=self.states[start:done_index]
                if self.states is not None
                else None,
                actions=self.actions[start:done_index]
                if self.actions is not None
                else None,
                rewards=self.rewards[start:done_index]
                if self.rewards is not None
                else None,
                features=self.features[start:done_index]
                if self.features is not None
                else None,
            )
            start = done_index

        if include_incomplete:
            yield RlDataset.Traj(
                states=self.states[start:] if self.states is not None else None,
                actions=self.actions[start:] if self.actions is not None else None,
                rewards=self.rewards[start:] if self.rewards is not None else None,
                features=self.features[start:] if self.features is not None else None,
            )

    def truncated_returns(
        self, horizon: int, discount_rate: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert discount_rate >= 0.0 and discount_rate <= 1.0
        assert (
            self.states is not None
            and self.actions is not None
            and self.rewards is not None
            and self.dones is not None
        )

        discounts = torch.pow(
            torch.ones(horizon, dtype=self.rewards.dtype) * discount_rate,
            torch.arange(horizon),
        )

        done_indices = self.dones.nonzero(as_tuple=True)[0] + 1
        returns: List[torch.Tensor] = []
        start_index = 0
        for done_index in done_indices:
            while start_index < done_index:
                reward_batch = self.rewards[
                    start_index : min(start_index + horizon, done_index)
                ]
                assert (
                    reward_batch.dtype == discounts.dtype
                ), f"dtype mismatch, reward_batch={reward_batch.dtype}, discounts={discounts.dtype}"
                returns.append(reward_batch @ discounts[: len(reward_batch)])
                start_index += 1

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

        assert len(states) == len(
            returns
        ), f"{len(states)} states but {len(returns)} returns"

        returns_torch = torch.stack(returns)

        return states, actions, returns_torch


class SarsDataset(RlDataset):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(states, actions, rewards, dones, features)

        if len(self) == 0 and len(states) > 0:
            logging.warning("Every timestep is done")
            self.index_map = np.empty(0, dtype=int)
        else:
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
        assert self.states is not None and self.dones is not None
        return len(self.states) - torch.sum(self.dones) - 1

    def __getitem__(
        self, i: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            self.states is not None
            and self.actions is not None
            and self.rewards is not None
        )
        j = self.index_map[i]
        return self.states[j], self.actions[j], self.rewards[j], self.states[j + 1]

    def make_sars(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # The torch index typing doesn't handle numpy arrays for some reason.
        states = self.states[self.index_map]  # type: ignore
        actions = self.actions[self.index_map]  # type: ignore
        rewards = self.rewards[self.index_map]  # type: ignore
        next_states = self.states[self.index_map + 1]  # type: ignore

        return states, actions, rewards, next_states

    @classmethod
    def from_rl_dataset(cls, data: RlDataset) -> SarsDataset:
        assert (
            data.states is not None
            and data.actions is not None
            and data.rewards is not None
            and data.dones is not None
        )
        return cls(data.states, data.actions, data.rewards, data.dones)

    @classmethod
    def from_gym3(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        firsts: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> SarsDataset:
        return cls(
            **RlDataset.process_gym3(
                states=states,
                actions=actions,
                rewards=rewards,
                firsts=firsts,
                features=features,
            )
        )

    def append_gym3(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        firsts: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> SarsDataset:
        """This function exists entirely to fix the return type."""
        return cast(
            SarsDataset,
            super().append_gym3(
                states=states,
                actions=actions,
                rewards=rewards,
                firsts=firsts,
                features=features,
            ),
        )

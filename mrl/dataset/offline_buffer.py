from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import torch


class RlDataset:
    def __init__(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.firsts = firsts
        self.features = features

    @staticmethod
    def process_gym3(
        *,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        keep_incomplete: bool = False,
    ) -> Dict[str, np.ndarray]:
        out = {}

        time = get_first(
            lambda x: x.shape[0], (states, actions, rewards, firsts, features)
        )
        n_envs = get_first(
            lambda x: x.shape[1], (states, actions, rewards, firsts, features)
        )

        if not keep_incomplete:
            if firsts is None:
                raise ValueError("Must supply firsts to remove incomplete trajs.")
            last_first = np.stack(
                [np.nonzero(firsts[:, env])[0][-1] for env in range(n_envs)]
            )
        else:
            last_first = np.ones(n_envs, dtype=int) * time

        if states is not None:
            # (T, num, H, W, C) -> (num * T, H, W, C)
            out["states"] = np.concatenate(
                [states[:l, env] for env, l in enumerate(last_first)]
            )
        if actions is not None:
            out["actions"] = np.concatenate(
                [actions[:l, env] for env, l in enumerate(last_first)]
            )
        if rewards is not None:
            out["rewards"] = np.concatenate(
                [rewards[:l, env] for env, l in enumerate(last_first)]
            )
        if firsts is not None:
            out["firsts"] = np.concatenate(
                [firsts[:l, env] for env, l in enumerate(last_first)]
            )
        if features is not None:
            # (T, num, features) -> (num * T, features)
            out["features"] = np.concatenate(
                [features[:l, env] for env, l in enumerate(last_first)]
            )

        return out

    def __len__(self) -> int:
        return get_first(
            lambda x: x.shape[0],
            (self.states, self.actions, self.rewards, self.firsts, self.features),
        )

    @classmethod
    def from_gym3(
        cls,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        keep_incomplete: bool = False,
    ) -> RlDataset:
        """Builds RLDataset from procgen_rollout output arrays"""
        return cls(
            **RlDataset.process_gym3(
                states=states,
                actions=actions,
                rewards=rewards,
                firsts=firsts,
                features=features,
                keep_incomplete=keep_incomplete,
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
            self.states = np.concatenate((self.states, data["states"]), axis=0)
        if self.actions is not None and data["actions"] is not None:
            self.actions = np.concatenate((self.actions, data["actions"]), axis=0)
        if self.rewards is not None and data["rewards"] is not None:
            self.rewards = np.concatenate((self.rewards, data["rewards"]), axis=0)
        if self.firsts is not None and data["firsts"] is not None:
            self.firsts = np.concatenate((self.firsts, data["firsts"]), axis=0)
        if self.features is not None and data["features"] is not None:
            self.features = np.concatenate((self.features, data["features"]), axis=0)
        return self

    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> RlDataset:
        """Builds RLDataset from Roller output dict"""
        states = data["ob"].reshape(
            (-1, *data["ob"].shape[2:])
        )  # (num, T, H, W, C) -> (num * T, H, W, C)
        actions = data["ac"].flatten()  # (num, T) -> (num * T)
        rewards = data["reward"].flatten()  # (num, T) -> (num * T)
        firsts = data["first"].flatten()  # (num, T) -> (num * T)

        return cls(states=states, actions=actions, rewards=rewards, firsts=firsts)

    def get_bytes(self) -> int:
        total = 0
        if self.states is not None:
            total += self.states.nbytes
        if self.actions is not None:
            total += self.actions.nbytes
        if self.rewards is not None:
            total += self.rewards.nbytes
        if self.firsts is not None:
            total += self.firsts.nbytes
        if self.features is not None:
            total += self.features.nbytes
        return total

    @dataclass
    class Traj:
        states: Optional[np.ndarray] = None
        features: Optional[np.ndarray] = None
        actions: Optional[np.ndarray] = None
        rewards: Optional[np.ndarray] = None

    def trajs(
        self,
        *,
        include_last: bool = True,
    ) -> Generator[Traj, None, None]:
        assert self.firsts is not None, "Must supply firsts to get trajs."

        if self.firsts.shape[0] == 1:
            # Only one timestep provided
            yield self.Traj(
                states=self.states,
                features=self.features,
                actions=self.actions,
                rewards=self.rewards,
            )
            return

        first_indices = self.firsts.nonzero()[0][1:]

        start = 0
        for end in first_indices:
            yield RlDataset.Traj(
                states=self.states[start:end] if self.states is not None else None,
                actions=self.actions[start:end] if self.actions is not None else None,
                rewards=self.rewards[start:end] if self.rewards is not None else None,
                features=self.features[start:end]
                if self.features is not None
                else None,
            )
            start = end

        if include_last:
            yield RlDataset.Traj(
                states=self.states[start:] if self.states is not None else None,
                actions=self.actions[start:] if self.actions is not None else None,
                rewards=self.rewards[start:] if self.rewards is not None else None,
                features=self.features[start:] if self.features is not None else None,
            )

    def truncated_returns(
        self, horizon: int, discount_rate: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert discount_rate >= 0.0 and discount_rate <= 1.0
        assert (
            self.states is not None
            and self.actions is not None
            and self.rewards is not None
            and self.firsts is not None
        )

        discounts = torch.pow(
            torch.ones(horizon, dtype=self.rewards.dtype) * discount_rate,
            torch.arange(horizon),
        )

        first_indices = self.firsts.nonzero()[0]
        returns: List[np.ndarray] = []
        start_index = 0
        for end_index in first_indices:
            while start_index < end_index:
                reward_batch = self.rewards[
                    start_index : min(start_index + horizon, end_index)
                ]
                assert (
                    reward_batch.dtype == discounts.dtype
                ), f"dtype mismatch, reward_batch={reward_batch.dtype}, discounts={discounts.dtype}"
                returns.append(reward_batch @ discounts[: len(reward_batch)])
                start_index += 1

        if len(first_indices) > 0:
            assert (
                start_index == first_indices.max(axis=0)[0]
            ), f"final start_index={start_index} but max first_index={first_indices.max(axis=0)[0]}"

        # After the last first, we don't want to use any fewer than k rewards, because the episode
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

        returns_np = np.stack(returns)

        return states, actions, returns_np


class SarsDataset(RlDataset):
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    firsts: np.ndarray

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        firsts: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(
            states=states,
            actions=actions,
            rewards=rewards,
            firsts=firsts,
            features=features,
        )

        if len(self) == 0 and len(states) > 0:
            logging.warning("Every timestep is first")
            self.index_map = np.empty(0, dtype=int)
        else:
            self.index_map = self._compute_index_map(self.firsts)

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
        assert self.states is not None and self.firsts is not None
        # There are (n-1) possible transitions from n states. Each time an episode resets, we don't know the
        # terminal states, so we subtract out that many transitions. There are sum(firsts) - 1 resets
        # (the first reset doesn't count. (n-1)-(sum(f)-1) = n-sum(f)
        return len(self.states) - np.sum(self.firsts)

    def __getitem__(
        self, i: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert (
            self.states is not None
            and self.actions is not None
            and self.rewards is not None
        )
        j = self.index_map[i]
        return self.states[j], self.actions[j], self.rewards[j], self.states[j + 1]

    def make_sars(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = self.states[self.index_map]
        actions = self.actions[self.index_map]
        rewards = self.rewards[self.index_map]
        next_states = self.states[self.index_map + 1]

        return states, actions, rewards, next_states

    @classmethod
    def from_rl_dataset(cls, data: RlDataset) -> SarsDataset:
        assert (
            data.states is not None
            and data.actions is not None
            and data.rewards is not None
            and data.firsts is not None
        )
        return cls(data.states, data.actions, data.rewards, data.firsts)

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


X = TypeVar("X")
T = TypeVar("T")


def get_first(f: Callable[[X], T], args: Sequence[Optional[X]]) -> T:
    for arg in args:
        if arg is not None:
            return f(arg)
    raise ValueError("No non-None argument")

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Final,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import numpy as np
import torch


class TrajectoryDataset:
    Attrs = Literal["states", "actions", "rewards", "firsts", "features"]
    NAMES: Final[Tuple[Attrs, ...]] = (
        "states",
        "actions",
        "rewards",
        "firsts",
        "features",
    )

    def __init__(
        self,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        dones: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
    ) -> None:
        self.data: Dict[TrajectoryDataset.Attrs, Optional[np.ndarray]] = {
            name: value
            for name, value in zip(
                self.NAMES, (states, actions, rewards, firsts, dones, features)
            )
        }

    def __len__(self) -> int:
        def get_len(arr: np.ndarray) -> int:
            return arr.shape[0]

        return get_first(
            get_len,
            self.data.values(),
        )

    @staticmethod
    def process_gym3(
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        remove_incomplete: bool = True,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        data = (states, actions, rewards, firsts, features)

        time = get_first(lambda x: x.shape[0], data)
        n_envs = get_first(lambda x: x.shape[1], data)

        if remove_incomplete:
            if firsts is None:
                raise ValueError("Must supply firsts to remove incomplete trajs.")
            if np.sum(firsts) == 0:
                raise ValueError("No complete trajectories.")
            bounds_list: List[Tuple[int, int]] = []
            for env in range(n_envs):
                firsts_index = np.nonzero(firsts[:, env])[0]
                bounds_list.append((firsts_index[0], firsts_index[-1]))

            bounds = np.array(bounds_list)
        else:
            bounds = np.stack(
                (np.zeros(n_envs, dtype=int), np.ones(n_envs, dtype=int) * time)
            ).T

        for name, arr in zip(TrajectoryDataset.NAMES, data):
            if arr is None:
                continue
            out[name] = np.concatenate(
                [arr[start:end, env] for env, (start, end) in enumerate(bounds)], axis=0
            )
        return out

    @classmethod
    def from_gym3(
        cls,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        remove_incomplete: bool = True,
    ) -> TrajectoryDataset:
        """Builds RLDataset from procgen_rollout output arrays"""

        return cls(
            **TrajectoryDataset.process_gym3(
                states=states,
                actions=actions,
                rewards=rewards,
                firsts=firsts,
                features=features,
                remove_incomplete=remove_incomplete,
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
    ) -> TrajectoryDataset:
        data = TrajectoryDataset.process_gym3(
            states=states,
            actions=actions,
            rewards=rewards,
            firsts=firsts,
            features=features,
        )

        for name, arr in data.items():
            name = cast(TrajectoryDataset.Attrs, name)
            if arr is None or self.data[name] is None:
                continue
            self.data[name] = np.concatenate((self.data[name], arr), axis=0)
        return self

    def get_bytes(self) -> int:
        total = 0
        for arr in self.data.values():
            if arr is not None:
                total += arr.nbytes
        return total

    @dataclass
    class Traj:
        states: Optional[np.ndarray] = None
        features: Optional[np.ndarray] = None
        actions: Optional[np.ndarray] = None
        rewards: Optional[np.ndarray] = None

    def get_traj(self, start: int, end: int) -> Traj:
        return self.Traj(
            states=self.data["states"][start:end]
            if self.data["states"] is not None
            else None,
            features=self.data["features"][start:end]
            if self.data["features"] is not None
            else None,
            actions=self.data["actions"][start:end]
            if self.data["actions"] is not None
            else None,
            rewards=self.data["rewards"][start:end]
            if self.data["rewards"] is not None
            else None,
        )

    def trajs(self) -> Generator[Traj, None, None]:
        firsts = self.data["firsts"]
        if firsts is None:
            raise ValueError("Must supply firsts to generate trajectories.")

        if firsts.shape[0] == 1:
            # Only one timestep provided
            yield self.Traj(
                states=self.data["states"],
                features=self.data["features"],
                actions=self.data["actions"],
                rewards=self.data["rewards"],
            )
            return

        end_indices = firsts.nonzero()[0][1:]
        if len(end_indices) == 0:
            yield self.Traj(
                states=self.data["states"],
                features=self.data["features"],
                actions=self.data["actions"],
                rewards=self.data["rewards"],
            )
            return

        start = 0
        for end in end_indices:
            yield self.get_traj(start, end)
            start = end

        yield TrajectoryDataset.Traj(
            states=self.data["states"][end:]
            if self.data["states"] is not None
            else None,
            actions=self.data["actions"][end:]
            if self.data["actions"] is not None
            else None,
            rewards=self.data["rewards"][end:]
            if self.data["rewards"] is not None
            else None,
            features=self.data["features"][end:]
            if self.data["features"] is not None
            else None,
        )


X = TypeVar("X")
T = TypeVar("T")


def get_first(f: Callable[[X], T], args: Iterable[Optional[X]]) -> T:
    for arg in args:
        if arg is not None:
            return f(arg)
    raise ValueError("No non-None argument")

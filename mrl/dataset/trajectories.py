from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import (Callable, Dict, Final, Generator, Iterable, List, Optional,
                    Tuple, TypeVar, cast)

import numpy as np
import torch


class TrajectoryDataset:
    NAMES: Final[Tuple[str, ...]] = (
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
        features: Optional[np.ndarray] = None,
        extras: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.data: Dict[str, Optional[np.ndarray]] = {
            name: value
            for name, value in zip(
                self.NAMES, (states, actions, rewards, firsts, features)
            )
        }
        if extras is not None:
            self.data.update(extras)

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
        extras: Optional[Dict[str, np.ndarray]] = None,
        remove_incomplete: bool = True,
    ) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}

        data: Tuple[Optional[np.ndarray], ...] = (
            states,
            actions,
            rewards,
            firsts,
            features,
        )
        if extras is not None:
            data += tuple(extras.values())

        time = get_first(lambda x: x.shape[0], data)
        n_envs = get_first(lambda x: x.shape[1], data)

        if remove_incomplete:
            if firsts is None:
                raise ValueError("Must supply firsts to remove incomplete trajs.")
            if np.sum(firsts) == 0:
                raise ValueError("No complete trajectories.")
            bounds_list: List[Tuple[int, int]] = []
            for env in range(n_envs):
                firsts_env = firsts[:, env]
                firsts_index = np.nonzero(firsts_env)[0]
                if len(firsts_index) == 0:
                    if time >= 1000:
                        logging.warning(
                            "No env restart in entire rollout but 1000 timesteps passed."
                        )
                    bounds_list.append((0, 0))
                    continue
                bounds_list.append((firsts_index[0], firsts_index[-1]))

            bounds = np.array(bounds_list)
        else:
            bounds = np.stack(
                (np.zeros(n_envs, dtype=int), np.ones(n_envs, dtype=int) * time)
            ).T

        for name, arr in chain(
            zip(TrajectoryDataset.NAMES, data),
            extras.items() if extras is not None else (),
        ):
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
        extras: Optional[Dict[str, np.ndarray]] = None,
        remove_incomplete: bool = True,
    ) -> TrajectoryDataset:
        """Builds RLDataset from procgen_rollout output arrays"""
        data = TrajectoryDataset.process_gym3(
            states=states,
            actions=actions,
            rewards=rewards,
            firsts=firsts,
            features=features,
            extras=extras,
            remove_incomplete=remove_incomplete,
        )
        return cls(
            states=data.pop("states", None),
            actions=data.pop("actions", None),
            rewards=data.pop("rewards", None),
            firsts=data.pop("firsts", None),
            features=data.pop("features", None),
            extras=data,
        )

    def append_gym3(
        self,
        *,
        states: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        rewards: Optional[np.ndarray] = None,
        firsts: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        extras: Optional[Dict[str, np.ndarray]] = None,
    ) -> TrajectoryDataset:
        data = TrajectoryDataset.process_gym3(
            states=states,
            actions=actions,
            rewards=rewards,
            firsts=firsts,
            features=features,
            extras=extras,
        )

        for name, arr in data.items():
            cur_data = self.data[name]
            if arr is None or cur_data is None:
                continue

            self.data[name] = np.concatenate((cur_data, arr), axis=0)
        return self

    def get_bytes(self) -> int:
        total = 0
        for arr in self.data.values():
            if arr is not None:
                total += arr.nbytes
        return total

    def _get_extras(self) -> Optional[Dict[str, np.ndarray]]:
        has_extras = len(self.NAMES) < len(self.data)

        return (
            {
                name: arr
                for name, arr in self.data.items()
                if name not in self.NAMES and arr is not None
            }
            if has_extras
            else None
        )

    @dataclass
    class Traj:
        states: Optional[np.ndarray] = None
        actions: Optional[np.ndarray] = None
        rewards: Optional[np.ndarray] = None
        features: Optional[np.ndarray] = None
        extras: Optional[Dict[str, np.ndarray]] = None

    def get_traj(self, start: int, end: int) -> Traj:
        extras = self._get_extras()
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
            extras={name: arr[start:end] for name, arr in extras.items()}
            if extras is not None
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
                actions=self.data["actions"],
                rewards=self.data["rewards"],
                features=self.data["features"],
                extras=self._get_extras(),
            )
            return

        end_indices = firsts.nonzero()[0][1:]
        if len(end_indices) == 0:
            yield self.Traj(
                states=self.data["states"],
                features=self.data["features"],
                actions=self.data["actions"],
                rewards=self.data["rewards"],
                extras=self._get_extras(),
            )
            return

        start = 0
        for end in end_indices:
            yield self.get_traj(start, end)
            start = end

        extras = self._get_extras()
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
            extras={name: arr[end:] for name, arr in extras.items()}
            if extras is not None
            else None,
        )


X = TypeVar("X")
T = TypeVar("T")


def get_first(f: Callable[[X], T], args: Iterable[Optional[X]]) -> T:
    for arg in args:
        if arg is not None:
            return f(arg)
    raise ValueError("No non-None argument")

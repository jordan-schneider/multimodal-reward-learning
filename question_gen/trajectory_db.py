from __future__ import annotations

from copy import deepcopy
from functools import partial
from itertools import product
from typing import Dict, Final, List, Literal, Optional, Sequence, Tuple, Union, cast

import arrow
import numpy as np
import pandas as pd  # type: ignore


def clip_np(arr: Optional[np.ndarray], max_length: int) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if max_length == 0:
        if len(arr.shape) == 1:
            return np.empty((0,))
        else:
            return np.empty((0, *arr.shape[1:]))
    return arr[-max_length:]


def clip_np_actions(
    arr: Optional[np.ndarray], max_length: int, keep_last_action: bool
) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if max_length == 0:
        if len(arr.shape) == 1:
            return np.empty((0,))
        else:
            return np.empty((0, *arr.shape[1:]))
    last_index = len(arr) if keep_last_action else -1
    return arr[-max_length:last_index]


def count_items(arr: np.ndarray, item_shape: Tuple[int, ...]) -> int:
    if len(arr.shape) == len(item_shape) + 1 and arr.shape[1:] == item_shape:
        return arr.shape[0]
    elif arr.shape == item_shape:
        return 0

    raise ValueError(f"arr shape={arr.shape} incompatible with item shape={item_shape}")


def not_all_equal(x: Sequence) -> bool:
    return any(x[0] != x[i] for i in range(1, len(x)))


def batch(
    xs: Sequence[Union[Sequence[np.ndarray], np.ndarray, None]]
) -> List[Optional[Sequence[np.ndarray]]]:
    outs: List[Optional[Sequence[np.ndarray]]] = []
    for x in xs:
        if isinstance(x, np.ndarray):
            outs.append([x])
        else:
            outs.append(x)
    return outs


class FeatureDataset:
    BASE_COLS: Final[List[str]] = [
        "policy",
        "datetime",
        "length",
        "states",
        "features",
        "actions",
        "total_feature",
    ]

    def __init__(
        self, rng: np.random.Generator, extra_cols: Optional[Sequence[str]] = None
    ):
        cols = list(self.BASE_COLS)
        if extra_cols is not None:
            cols += list(extra_cols)
        self.df = pd.DataFrame(columns=cols)
        # Paths to policy models -> indices into df
        self.states: Dict[str, Optional[np.ndarray]] = {}

        self.rng = rng

    @staticmethod
    def from_pickled(data: FeatureDataset) -> FeatureDataset:
        out = FeatureDataset(data.rng, data.df.columns[len(FeatureDataset.BASE_COLS) :])
        out.df = deepcopy(data.df)
        out.states = deepcopy(data.states)
        return out

    def init_df(self):
        self.df["policy"] = self.df["policy"].astype("string")
        self.df["length"] = self.df["length"].astype("int")

    def append(
        self,
        policy: str,
        time: arrow.Arrow,
        states: Union[Sequence[np.ndarray], np.ndarray, None] = None,
        state_features: Union[Sequence[np.ndarray], np.ndarray, None] = None,
        actions: Union[Sequence[np.ndarray], np.ndarray, None] = None,
        extras: Union[
            Dict[str, Sequence[np.ndarray]], Dict[str, np.ndarray], None
        ] = None,
    ) -> None:
        if (
            states is None
            and state_features is None
            and actions is None
            and extras is None
        ):
            raise ValueError(
                "Must provide at least one of states, state_features, actions, or extras."
            )
        if extras is None:
            extras = cast(Dict[str, Sequence[np.ndarray]], {})
        types = [
            type(a)
            for a in (states, state_features, actions, *extras.values())
            if a is not None
        ]
        if not_all_equal(types):
            raise ValueError(f"Provided types are not all the same: {types}")

        state_by_traj, feature_by_traj, action_by_traj = batch(
            (states, state_features, actions)
        )

        extras_by_traj = cast(
            Dict[str, np.ndarray],
            {name: batch((arr,))[0] for name, arr in extras.items()},
        )

        lens = [
            len(a)
            for a in (
                state_by_traj,
                feature_by_traj,
                action_by_traj,
                *extras_by_traj.values(),
            )
            if a is not None
        ]
        if not_all_equal(lens):
            raise ValueError(
                f"Was given unequal number of trajectory components: {lens}"
            )
        n_trajs = lens[0]

        init = len(self.df) == 0

        for traj in range(n_trajs):
            traj_state = state_by_traj[traj] if state_by_traj is not None else None
            traj_feature = (
                feature_by_traj[traj] if feature_by_traj is not None else None
            )
            traj_action = action_by_traj[traj] if action_by_traj is not None else None
            traj_extras = {name: arr[traj] for name, arr in extras_by_traj.items()}

            lens = [
                len(arr)
                for arr in (
                    traj_state,
                    traj_feature,
                    traj_action,
                    *traj_extras.values(),
                )
                if arr is not None
            ]

            if not_all_equal(lens):
                raise ValueError(f"Length of trajectory {traj} not consistent: {lens}")

            self.df.loc[len(self.df)] = (
                policy,
                time.datetime,
                lens[0],
                traj_state,
                traj_feature,
                traj_action,
                np.sum(traj_feature, axis=0) if traj_feature is not None else None,
                *[traj_extras[key] for key in self.df.columns[len(self.BASE_COLS) :]],
            )
            self.states[policy] = None

        if init:
            self.init_df()

    def get_state_comparisons(
        self, n: int, how: Literal["even"] = "even"
    ) -> np.ndarray:
        if n <= 0:
            return np.array([])

        if how == "even":
            return self.get_even_states(n)
        else:
            raise NotImplementedError("Only even sampling supported at the moment")

    def get_trajectory_comparisons(
        self, n: int, how: Literal["even"] = "even"
    ) -> np.ndarray:
        if n <= 0:
            return np.array([])

        if how == "even":
            return self.get_even_trajs(n)

    def get_policy_states(self, policy: str) -> np.ndarray:
        if (out := self.states.get(policy, None)) is not None:
            return out

        states_of_policy = self.df[self.df["policy"] == policy]
        out = np.concatenate(states_of_policy["features"].to_numpy(), axis=0)

        self.states[policy] = out
        return out

    def get_even_states(self, n: int) -> np.ndarray:
        policies = self.df["policy"].unique()
        n_policies = len(policies)
        states_per_model_pair = max(1, n // (n_policies**2))

        out = []
        for first, second in product(policies, policies):
            firsts = self.rng.choice(
                self.get_policy_states(first),
                size=states_per_model_pair,
                replace=False,
            )
            seconds = self.rng.choice(
                self.get_policy_states(second),
                size=states_per_model_pair,
                replace=False,
            )
            out.append(firsts - seconds)
        return np.concatenate(out, axis=0)

    def get_even_trajs(self, n: int) -> np.ndarray:
        policies = self.df["policy"].unique()
        n_policies = len(policies)
        trajs_per_model_pair = max(1, n // (n_policies**2))

        out = []
        for first, second in product(policies, policies):
            firsts = self.df[self.df["policy"] == first]["total_feature"].sample(
                n=trajs_per_model_pair, random_state=self.rng
            )
            seconds = self.df[self.df["policy"] == second]["total_feature"].sample(
                n=trajs_per_model_pair, random_state=self.rng
            )
            out.append(firsts - seconds)
        return np.stack(out)

    def clip(self, max_length: int, keep_last_action: bool = True) -> FeatureDataset:

        out = deepcopy(self)
        to_clip = out.df["length"] > max_length

        out.df.loc[to_clip, "states"] = out.df.loc[to_clip, "states"].apply(
            partial(clip_np, max_length=max_length)
        )
        out.df.loc[to_clip, "features"] = out.df.loc[to_clip, "features"].apply(
            partial(clip_np, max_length=max_length)
        )
        out.df.loc[to_clip, "actions"] = out.df.loc[to_clip, "actions"].apply(
            partial(
                clip_np_actions,
                max_length=max_length,
                keep_last_action=keep_last_action,
            )
        )

        out.df.loc[to_clip, "total_feature"] = out.df.loc[to_clip, "features"].apply(
            np.sum, axis=0
        )

        # TODO: The extras to clip are hard coded. In principle I would like to have a numpy array type with named
        # dimensions, but all of the options are overly complicated. Struct/record arrays in numpy or switch everything
        # to pandas.
        out.df.loc[to_clip, "grids"] = out.df.loc[to_clip, "grids"].apply(
            partial(clip_np, max_length=max_length)
        )
        out.df.loc[to_clip, "cstates"] = out.df.loc[to_clip, "cstates"].apply(
            partial(clip_np, max_length=max_length)
        )

        out.df.loc[to_clip, "length"] = max_length

        return out

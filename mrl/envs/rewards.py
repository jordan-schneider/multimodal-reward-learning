from itertools import product
from math import floor
from typing import Sequence, Tuple

import numpy as np
from linear_procgen import Maze, Miner
from linear_procgen.feature_envs import FeatureEnv
from mrl.util import normalize_vecs


def make_reward_weights(env: FeatureEnv, n_rewards: int) -> np.ndarray:
    if isinstance(env, Miner):
        return _make_miner_reward(env, n_rewards)
    elif isinstance(env, Maze):
        return _make_maze_reward(env, n_rewards)
    else:
        raise ValueError(
            f"Unsupported environment type {type(env)}. Supported types are: Miner, Maze."
        )


def _make_miner_reward(
    env: Miner,
    n_rewards: int,
    feature_ranges: Sequence[Tuple[float, float]] = [
        (-1, 0),
        (-0.1, 0),
        (-0.1, 0),
    ],
) -> np.ndarray:
    n_dim = len(feature_ranges)
    values_per_dim = floor(n_rewards ** (1 / n_dim))
    rewards = np.empty((values_per_dim ** n_dim, env.n_features))
    features = [
        np.linspace(start, stop, num=values_per_dim) for start, stop in feature_ranges
    ]
    # Reverse for backward compatibility reasons.
    rewards = np.stack(list(product(*features)))[::-1]

    return normalize_vecs(rewards)


def _make_maze_reward(
    env: Maze, n_rewards: int, feature_ranges: Sequence[Tuple[float, float]] = [(-1, 0)]
) -> np.ndarray:
    raise NotImplementedError

from pathlib import Path

import numpy as np
from numpy.random import Generator

from mrl.envs import Miner


def make_original_reward() -> np.ndarray:
    reward = np.array([1, 10, 0, 0, 0])
    reward = reward / np.linalg.norm(reward)
    return reward


def make_reward(
    path: Path, rng: Generator, fix_sign: bool = False, use_original: bool = False
) -> np.ndarray:

    if use_original:
        reward = make_original_reward()
    else:
        reward = rng.standard_normal(size=(Miner.N_FEATURES,))
        reward = reward / np.linalg.norm(reward)
        if fix_sign:
            pos_reward = np.abs(reward)
            reward = pos_reward * [1.0, 1.0, -1.0, -1.0, -1.0]

    np.save(path / "reward.npy", reward)
    return reward


def load_reward(
    path: Path,
    rng: Generator,
    overwrite: bool = False,
    fix_reward_sign: bool = False,
    use_original: bool = False,
) -> np.ndarray:
    if overwrite or not (path / "reward.npy").exists():
        return make_reward(path, rng, fix_reward_sign, use_original)
    return np.load(path / "reward.npy")

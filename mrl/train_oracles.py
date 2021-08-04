from pathlib import Path

import fire
import gym3
import numpy as np
from mpi4py import MPI
from numpy.random import Generator
from phasic_policy_gradient.train import train_fn

from mrl.envs import Miner


def make_reward(path: Path, rng: Generator) -> np.ndarray:
    reward = rng.standard_normal(size=(Miner.N_FEATURES,))
    reward = reward / np.linalg.norm(reward)
    np.save(path / "reward.npy", reward)
    return reward


def load_reward(path: Path, rng: Generator, overwrite: bool = False) -> np.ndarray:
    if overwrite or not (path / "reward.npy").exists():
        make_reward(path, rng)
    return np.load(path / "reward.npy")


def train(path: Path, overwrite: bool = False, seed: int = 0, n_parallel_envs: int = 1) -> None:
    path = Path(path)

    rng = np.random.default_rng(seed)
    reward = load_reward(path, rng, overwrite)

    env = Miner(reward_weights=reward, num=n_parallel_envs, rand_seed=seed)
    env = gym3.ExtractDictObWrapper(env, "rgb")

    comm = MPI.COMM_WORLD

    train_fn(arch="detach", n_epoch_vf=6, log_dir=path, comm=comm, venv=env)


if __name__ == "__main__":
    fire.Fire({"train": train})

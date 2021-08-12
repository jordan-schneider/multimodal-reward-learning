from pathlib import Path

import fire  # type: ignore
import gym3  # type: ignore
import numpy as np
from mpi4py import MPI  # type: ignore
from phasic_policy_gradient.train import train_fn  # type: ignore

from mrl.envs import Miner
from mrl.rewards import load_reward


def train(
    path: Path,
    overwrite: bool = False,
    seed: int = 0,
    n_parallel_envs: int = 1,
    fix_reward_sign: bool = False,
    use_original_reward: bool = False,
    port=29500,
) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    reward = load_reward(path, rng, overwrite, fix_reward_sign, use_original_reward)

    env = Miner(reward_weights=reward, num=n_parallel_envs, rand_seed=seed)
    env = gym3.ExtractDictObWrapper(env, "rgb")

    comm = MPI.COMM_WORLD

    train_fn(arch="detach", n_epoch_vf=6, log_dir=path, comm=comm, venv=env, port=port)


if __name__ == "__main__":
    fire.Fire({"train": train})

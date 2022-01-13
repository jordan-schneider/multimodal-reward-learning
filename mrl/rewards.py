from itertools import product
from pathlib import Path

import numpy as np
from mpi4py.MPI import Comm  # type: ignore
from numpy.random import Generator

from mrl.envs import Miner
from mrl.envs.util import ENV_NAMES


def make_original_reward() -> np.ndarray:
    reward = np.array([1, 10, 0, 0, 0])
    reward = reward / np.linalg.norm(reward)
    return reward


def make_miner_near_original_rewards(rootdir: Path, replications: int) -> None:
    n = replications ** (1.0 / 3.0)
    assert (
        n.is_integer()
    ), "If using near original rewards, you must have a perfect cube of replications"
    n = int(n)

    # TODO: Unhardcode these
    danger = np.linspace(0.0, -1.0, n)
    dists = np.linspace(0.0, -0.1, n)
    diamonds = np.linspace(0.0, -0.1, n)

    for i, vals in enumerate(product(danger, dists, diamonds)):
        reward = np.array([1.0, 10.0, *vals])
        reward = reward / np.linalg.norm(reward)

        repl_dir = rootdir / str(i)
        repl_dir.mkdir(parents=True, exist_ok=True)
        np.save(repl_dir / "reward.npy", reward)


def make_maze_near_original_rewards(rootdir: Path, replications: int):
    # TODO: Implement
    raise NotImplementedError()


def make_miner_reward(
    path: Path,
    rng: Generator,
    fix_sign: bool = False,
    use_original: bool = False,
) -> None:
    if use_original:
        reward = make_original_reward()
    else:
        reward = rng.standard_normal(size=(Miner.N_FEATURES,))
        reward = reward / np.linalg.norm(reward)
        if fix_sign:
            pos_reward = np.abs(reward)
            reward = pos_reward * [1.0, 1.0, -1.0, -1.0, -1.0]

    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "reward.npy", reward)


def make_maze_rewards(
    path: Path,
    rng: Generator,
    fix_sign: bool = False,
    use_original: bool = False,
):
    # TODO: implement
    raise NotImplementedError()


def make_rewards(
    path: Path,
    env: ENV_NAMES,
    rng: Generator,
    comm: Comm,
    overwrite: bool = False,
    fix_reward_sign: bool = False,
    use_original: bool = False,
    use_near_original: bool = False,
    replications: int = 1,
) -> None:
    if comm.rank == 0:
        if use_near_original:
            if overwrite or not (path / "0" / "reward.npy").exists():
                if env == "miner":
                    make_miner_near_original_rewards(path, replications)
                elif env == "maze":
                    make_maze_near_original_rewards(path, replications)

        else:
            for repl in range(replications):
                if overwrite or not (path / str(repl) / "reward.npy").exists():
                    if env == "miner":
                        make_miner_reward(
                            path / str(repl), rng, fix_reward_sign, use_original
                        )
                    elif env == "maze":
                        make_maze_rewards(
                            path / str(repl), rng, fix_reward_sign, use_original
                        )
    comm.Barrier()


def load_reward(path: Path, comm: Comm) -> np.ndarray:
    if comm.rank == 0:
        reward = np.load(path / "reward.npy")
    else:
        reward = None
    reward = comm.bcast(reward, root=0)
    comm.Barrier()
    return reward

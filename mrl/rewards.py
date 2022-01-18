from itertools import product
from pathlib import Path

import fire  # type: ignore
import numpy as np


def make_miner_rewards(rootdir: Path, replications: int) -> None:
    rootdir = Path(rootdir)
    n = replications ** (1.0 / 3.0)
    assert (
        n.is_integer()
    ), "If using near original rewards, you must have a perfect cube of replications"
    n = int(n)

    danger = np.linspace(0.0, -1.0, n)
    dists = np.linspace(0.0, -0.1, n)
    diamonds = np.linspace(0.0, -0.1, n)

    for i, vals in enumerate(product(danger, dists, diamonds)):
        reward = np.array([1.0, 10.0, *vals])
        reward = reward / np.linalg.norm(reward)

        repl_dir = rootdir / str(i)
        repl_dir.mkdir(parents=True, exist_ok=True)
        np.save(repl_dir / "reward.npy", reward)


def make_maze_rewards(
    rootdir: Path, replications: int, min_ratio: float, max_ratio: float
) -> None:
    rootdir = Path(rootdir)
    ratio = np.logspace(min_ratio, max_ratio, num=replications)

    distance = np.ones(replications)
    correct_step = distance * ratio

    for i in range(replications):
        reward = np.array([-distance[i], correct_step[i]])
        reward = reward / np.linalg.norm(reward)

        repl_dir = rootdir / str(i)
        repl_dir.mkdir(parents=True, exist_ok=True)
        np.save(repl_dir / "reward.npy", reward)


if __name__ == "__main__":
    fire.Fire({"maze": make_maze_rewards, "miner": make_miner_rewards})

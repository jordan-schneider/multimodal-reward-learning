import logging
from itertools import tee
from pathlib import Path
from typing import List, Optional

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mrl.dataset.roller import procgen_rollout_dataset
from mrl.dataset.trajectories import TrajectoryDataset
from linear_procgen.util import make_env
from mrl.util import get_policy, is_redundant


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def main(
    n_envs: int = 100,
    n_trajs: int = 10_000,
    outdir: Optional[Path] = None,
    verbosity: str = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)
    reward = np.ones(4)
    env = make_env(name="miner", num=n_envs, reward=reward)

    logging.info("Generating trajs")
    dataset = procgen_rollout_dataset(
        env=env,
        policy=get_policy(path=None, env=env),
        n_trajs=n_trajs,
        flags=["reward", "first", "feature"],
        remove_incomplete=True,
        tqdm=True,
    )

    logging.info("Finding nonredundant and recording lengths.")
    lengths: List[int] = []
    diffs: List[np.ndarray] = []
    for traj_1, traj_2 in pairwise(dataset.trajs()):
        assert isinstance(traj_1, TrajectoryDataset.Traj) and isinstance(
            traj_2, TrajectoryDataset.Traj
        )
        assert traj_1.rewards is not None and traj_1.features is not None
        assert traj_2.rewards is not None and traj_2.features is not None
        diff = np.sum(traj_1.features, axis=0) - np.sum(traj_2.features, axis=0)
        logging.debug(f"{diff.shape=}")
        if reward @ diff > 0:
            diff *= -1
        if len(diffs) < 2 or not is_redundant(diff, np.array(diffs)):
            diffs.append(diff)
            lengths.append(traj_1.rewards.shape[0])
            lengths.append(traj_2.rewards.shape[0])
    print(f"{len(lengths)} trajs")
    print(f"{np.mean(lengths)=}")
    print(f"{np.std(lengths)=}")
    print(f"{np.min(lengths)=}")
    print(f"{np.max(lengths)=}")

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(exist_ok=True, parents=True)
        plt.hist(lengths)
        plt.xlabel("Trajectory length")
        plt.title("Histogram of trajectory lengths")
        plt.savefig(outdir / "lengths.png")


if __name__ == "__main__":
    fire.Fire(main)

from pathlib import Path
from typing import Optional

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mrl.dataset.roller import procgen_rollout_dataset
from mrl.envs.util import make_env
from mrl.util import get_policy


def main(
    n_envs: int = 100, n_trajs: int = 10_000, outdir: Optional[Path] = None
) -> None:
    env = make_env(name="miner", num=n_envs, reward=1)

    dataset = procgen_rollout_dataset(
        env=env,
        policy=get_policy(path=None, env=env),
        n_trajs=n_trajs,
        flags=["reward", "first"],
        remove_incomplete=True,
        tqdm=True,
    )

    lengths = []
    for traj in dataset.trajs():
        assert traj.rewards is not None
        lengths.append(traj.rewards.shape[0])
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

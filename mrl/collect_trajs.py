import pickle as pkl
from pathlib import Path
from typing import List

import argh  # type: ignore
import arrow
import numpy as np
import torch
from argh import arg  # type: ignore
from gym3 import ExtractDictObWrapper  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel  # Type: ignore
from phasic_policy_gradient.train import make_model

from mrl.dataset.trajectory_db import FeatureDataset
from mrl.envs import Miner
from mrl.random_policy import RandomPolicy
from mrl.util import find_best_gpu, procgen_rollout_dataset


@arg("--policies", type=Path, nargs="+")
@arg("--outdir", type=Path)
def main(
    policies: List[Path] = [],
    outdir: Path = Path(),
    timesteps: int = int(1e7),
    use_random: bool = True,
    n_envs: int = 100,
    seed: int = 0,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "features.pkl"

    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)

    if outpath.exists():
        dataset = pkl.load(outpath.open("rb"))
        dataset.rng = rng
    else:
        dataset = FeatureDataset(rng=rng)

    env = Miner(reward_weights=np.zeros(5), num=n_envs)
    env = ExtractDictObWrapper(env, "rgb")

    for policy_path in policies:
        device = find_best_gpu()
        policy = make_model(env, arch="detach")
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy = policy.to(device)

        trajs = procgen_rollout_dataset(
            env=env, policy=policy, timesteps=timesteps, flags=["feature", "first"]
        ).trajs()
        for traj in trajs:
            assert traj.features is not None
            dataset.append(
                policy=policy_path,
                time=arrow.now(),
                state_features=traj.features.numpy(),
            )
        del policy

    if use_random:
        policy = RandomPolicy(actype=env.ac_space, num=n_envs)
        trajs = procgen_rollout_dataset(
            env=env, policy=policy, timesteps=timesteps, flags=["feature", "first"]
        ).trajs()
        for traj in trajs:
            assert traj.features is not None
            dataset.append(
                policy="random",
                time=arrow.now(),
                state_features=traj.features.numpy(),
            )

    pkl.dump(dataset, outpath.open("wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)

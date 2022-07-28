import logging
import pickle as pkl
from pathlib import Path
from typing import List

import argh  # type: ignore
import arrow
import numpy as np
import torch
from argh import arg  # type: ignore
from linear_procgen.util import make_env
from mrl.dataset.random_policy import RandomPolicy
from mrl.dataset.roller import procgen_rollout_dataset
from mrl.dataset.trajectory_db import FeatureDataset
from mrl.util import find_best_gpu
from phasic_policy_gradient.train import make_model


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

    env = make_env(name="miner", num=n_envs, reward=0)

    for policy_path in policies:
        logging.info(f"Collecting {timesteps} steps from policy at {policy_path}")
        device = find_best_gpu()
        policy = make_model(env, arch="shared")
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
                state_features=traj.features,
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
                state_features=traj.features,
            )

    pkl.dump(dataset, outpath.open("wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)

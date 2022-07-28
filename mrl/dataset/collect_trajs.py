import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def grid_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
) -> np.ndarray:
    return np.array([i["grid"] for i in info])


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
    outpath = outdir / "trajectories.pkl"

    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)

    if outpath.exists():
        dataset = pkl.load(outpath.open("rb"))
        dataset.rng = rng
    else:
        dataset = FeatureDataset(rng=rng, extra_cols=["grid"])

    env = make_env(name="miner", num=n_envs, reward=0)
    grid_shape = env.get_info()[0]["grid"].shape

    for policy_path in policies:
        logging.info(f"Collecting {timesteps} steps from policy at {policy_path}")
        device = find_best_gpu()
        policy = make_model(env, arch="shared")
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy = policy.to(device)

        trajs = procgen_rollout_dataset(
            env=env,
            policy=policy,
            timesteps=timesteps,
            flags=["action", "first", "feature"],
            extras=[(grid_hook, "grid", grid_shape)],
        ).trajs()
        for traj in trajs:
            assert (
                traj.features is not None
                and traj.actions is not None
                and traj.extras is not None
            )
            dataset.append(
                policy=policy_path,
                time=arrow.now(),
                state_features=traj.features,
                actions=traj.actions,
                extras=traj.extras,
            )
        del policy

    if use_random:
        policy = RandomPolicy(actype=env.ac_space, num=n_envs)
        trajs = procgen_rollout_dataset(
            env=env,
            policy=policy,
            timesteps=timesteps,
            flags=["action", "first", "feature"],
            extras=[(grid_hook, "grid", grid_shape)],
        ).trajs()
        for traj in trajs:
            assert (
                traj.features is not None
                and traj.actions is not None
                and traj.extras is not None
            )
            dataset.append(
                policy="random",
                time=arrow.now(),
                state_features=traj.features,
                actions=traj.actions,
                extras=traj.extras,
            )

    pkl.dump(dataset, outpath.open("wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)

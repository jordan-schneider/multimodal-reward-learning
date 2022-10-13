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
from mrl.dataset.lint_trajs import check_traj
from mrl.dataset.random_policy import RandomPolicy
from mrl.dataset.roller import procgen_rollout_dataset
from mrl.dataset.trajectories import TrajectoryDataset
from mrl.dataset.trajectory_db import FeatureDataset
from mrl.util import find_best_gpu
from phasic_policy_gradient.train import make_model
from procgen import ProcgenGym3Env


def grid_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
    cstate: List[bytes],
) -> np.ndarray:
    return np.array([i["grid"] for i in info])


def grid_shape_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
    cstate: List[bytes],
) -> np.ndarray:
    return np.array([i["grid_size"] for i in info])


def agent_pos_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
    cstate: List[bytes],
) -> np.ndarray:
    return np.array([i["agent_pos"] for i in info])


def exit_pos_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
    cstate: List[bytes],
) -> np.ndarray:
    return np.array([i["exit_pos"] for i in info])


def cstate_hook(
    state: np.ndarray,
    action: Optional[np.ndarray],
    reward: np.ndarray,
    first: np.ndarray,
    info: List[Dict[str, Any]],
    cstate: List[bytes],
) -> np.ndarray:
    return np.array(cstate)


def assert_no_fire_before_end(grids: np.ndarray) -> None:
    assert not np.any(grids[:-1] == 12)


def set_seeds(seed):
    rng = np.random.default_rng(seed=seed)
    torch.manual_seed(seed)
    return rng


def traj_dataset_from_policy(
    env: ProcgenGym3Env,
    policy,
    policy_name: str,
    timesteps: int,
    extras,
    rng: np.random.Generator,
) -> FeatureDataset:
    dataset = FeatureDataset(
        rng=rng,
        extra_cols=["cstates", "grids", "grid_shapes", "agent_pos", "exit_pos"],
    )

    trajs = procgen_rollout_dataset(
        env=env,
        policy=policy,
        timesteps=timesteps,
        flags=["action", "first", "feature"],
        extras=extras,
    ).trajs()
    del policy
    for traj in trajs:
        assert (
            traj.features is not None
            and traj.actions is not None
            and traj.extras is not None
        )

        assert_no_fire_before_end(traj.extras["grids"])
        dataset.append(
            policy=policy_name,
            time=arrow.now(),
            state_features=traj.features,
            actions=traj.actions,
            extras=traj.extras,
        )
        assert_no_fire_before_end(dataset.df["grids"].iloc[-1])

    return dataset


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

    # TODO: Extract outpath stuff to separate class
    out_shard = 0
    outpath = outdir / f"trajectories_{out_shard}.pkl"
    while outpath.exists():
        out_shard += 1
        outpath = outdir / f"trajectories_{out_shard}.pkl"

    rng = set_seeds(seed)

    env = make_env(name="miner", num=n_envs, reward=0)
    grid_shape = env.get_info()[0]["grid"].shape

    extras = [
        (cstate_hook, "cstates", tuple(), np.dtype("S22326")),
        (grid_hook, "grids", grid_shape, np.dtype(np.uint8)),
        (grid_shape_hook, "grid_shapes", (2,), np.dtype(np.uint8)),
        (agent_pos_hook, "agent_pos", (2,), np.dtype(np.uint8)),
        (exit_pos_hook, "exit_pos", (2,), np.dtype(np.uint8)),
    ]

    for policy_path in policies:
        logging.info(f"Collecting {timesteps} steps from policy at {policy_path}")

        device = find_best_gpu()
        policy = make_model(env, arch="shared")
        policy.load_state_dict(torch.load(policy_path, map_location=device))
        policy = policy.to(device)

        dataset = traj_dataset_from_policy(
            env, policy, str(policy_path), timesteps, extras, rng
        )

        for index, row in dataset.df.iterrows():
            print(f"Checking traj {index}")
            actions = row["actions"]
            cstates = row["cstates"]
            grids = row["grids"]
            agent_pos = row["agent_pos"]
            exit_pos = row["exit_pos"]
            check_traj(actions, cstates, grids, agent_pos, exit_pos, use_pdb=True)

        pkl.dump(dataset, outpath.open("wb"))
        del dataset
        out_shard += 1
        outpath = outdir / f"trajectories_{out_shard}.pkl"

    if use_random:
        policy = RandomPolicy(actype=env.ac_space, num=n_envs)
        dataset = traj_dataset_from_policy(
            env, policy, "random", timesteps, extras, rng
        )
        pkl.dump(dataset, outpath.open("wb"))


if __name__ == "__main__":
    argh.dispatch_command(main)

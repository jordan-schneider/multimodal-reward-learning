import logging
import time
from itertools import product
from pathlib import Path
from typing import Optional

import fire  # type: ignore
import numpy as np
import torch
from gym3 import ExtractDictObWrapper  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from scipy.optimize import linprog  # type: ignore

from mrl.envs.miner import Miner
from mrl.preferences import get_policy
from mrl.util import (procgen_rollout_dataset, procgen_rollout_features,
                      setup_logging)


def get_features(
    n_states: int,
    n_trajs: int,
    env: Miner,
    policy: PhasicValueModel,
    outdir: Optional[Path],
    tqdm: bool = False,
) -> np.ndarray:
    if outdir is not None and (outdir / "ars_features.npy").is_file():
        logging.info("Loading features from file")
        features = np.load(outdir / "ars_features.npy")
        if len(features) >= n_states + n_trajs:
            logging.info(f"n_states={n_states} n_trajs={n_trajs}")
            return np.concatenate((features[:n_states], features[-n_trajs:]))

    logging.info("Generating states")
    state_features = procgen_rollout_features(
        env=env,
        policy=policy,
        timesteps=n_states // env.num,
        tqdm=tqdm,
    ).reshape(-1, 5)
    assert (
        state_features.shape[0] == n_states
    ), f"state features shape={state_features.shape} when {n_states} states requested"

    logging.info("Generating trajs")
    trajs = procgen_rollout_dataset(
        env=env,
        policy=policy,
        n_trajs=n_trajs,
        flags=["feature", "first"],
        tqdm=tqdm,
    ).trajs()
    traj_features = np.stack([np.sum(traj.features.numpy(), axis=0) for traj in trajs])  # type: ignore
    assert (
        len(traj_features.shape) == 2
    ), f"traj feature has wrong dimension {traj_features.shape}"
    assert (
        traj_features.shape[0] >= n_trajs
    ), f"traj features {traj_features.shape} not expected length {n_trajs}"
    assert (
        traj_features.shape[1] == state_features.shape[1]
    ), f"traj and state feature dims don't match {traj_features.shape}, {state_features.shape}"

    features = np.concatenate(
        (state_features[:n_states], traj_features[:n_trajs]), axis=0
    )
    if outdir is not None:
        np.save(outdir / "ars_features.npy", features)

    return features


def make_aligned_reward_set(
    reward: np.ndarray,
    n_states: int,
    n_trajs: int,
    env: Miner,
    policy: PhasicValueModel,
    tqdm: bool = False,
    outdir: Optional[Path] = None,
) -> np.ndarray:
    features = get_features(n_states, n_trajs, env, policy, outdir, tqdm)
    logging.info(f"Features shape={features.shape}")

    logging.info("Finding non-redundant constraint set")
    start = time.time()

    total = len(features) * len(features) - len(features)
    logging.info(f"{total} total comparisons")

    iterations = 0
    diffs = np.empty((0, features.shape[1]))
    if outdir is not None and (outdir / "aligned_reward_set.npy").is_file():
        diffs = np.load(outdir / "aligned_reward_set.npy")
        iterations = len(diffs)

    np.random.shuffle(features)

    order1 = np.arange(len(features))
    np.random.shuffle(order1)
    order2 = np.arange(len(features))
    np.random.shuffle(order2)

    last_new = 0
    for i, j in product(order1, order2):
        if i == j:
            continue
        iterations += 1

        diff = features[i] - features[j]
        return_diff = reward @ diff
        opinion = np.sign(return_diff)
        if opinion == 0 or np.abs(return_diff) < 1e-16:
            continue
        diff *= opinion

        try:
            if len(diffs) < 2 or is_redundant(diff, diffs):
                diffs = np.append(diffs, [diff], axis=0)
                last_new = iterations
                if outdir is not None:
                    np.save(outdir / "aligned_reward_set.npy", diffs)
                logging.info(f"{len(diffs)} total diffs")
        except Exception as e:
            logging.warning("Unable to solve LP, adding item to set anyway")
            diffs = np.append(diffs, [diff], axis=0)
            last_new = iterations
            if outdir is not None:
                np.save(outdir / "aligned_reward_set.npy", diffs)
            logging.info(f"{len(diffs)} total diffs") 

        if iterations == 1000:
            stop = time.time()
            duration = stop - start
            logging.info(
                f"First 1000 iterations took {duration:0.1f} seconds. {total} total iters expected to take {duration * total / 1000: 0.1f} seconds."
            )
        if iterations % (total // 1000) == 0:
            logging.info(
                f"{iterations}/{total} pairs considered ({iterations / total * 100 : 0.2f}%)"
            )

        if iterations - last_new > 1e7:
            logging.info(f"1e7 iterations since last new diff, stopping.")
            break

    return diffs


def is_redundant(
    halfspace: np.ndarray, halfspaces: np.ndarray, epsilon: float = 1e-4
) -> bool:
    # Let h be a halfspace constraint in the set of contraints H.
    # We have a constraint c^T w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to
    # add c back into H.
    # Thus, we want to minimize c^T w subject to Hw >= 0.
    # First we need to change this into the form min c^T x subject to Ax <= b.
    # Our problem is equivalent to min c^T w subject to  -H w <= 0.
    if np.any(np.linalg.norm(halfspaces - halfspace) < epsilon):
        return True

    m, _ = halfspaces.shape

    b = np.zeros(m)
    solution = linprog(
        halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1), method="revised simplex"
    )
    logging.debug(f"LP Solution={solution}")
    if solution["status"] != 0:
        logging.info("Revised simplex method failed. Trying interior point method.")
        solution = linprog(halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1))

    if solution["status"] != 0:
        # Not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue.
        raise Exception("LP NOT SOLVABLE")
    elif solution["fun"] < -epsilon:
        # If less than zero then constraint is needed to keep c^T w >=0
        return False
    else:
        # redundant since without constraint c^T w >=0
        logging.debug("Redundant")
        return True


def main(
    reward_path: Path,
    outdir: Path,
    policy_path: Optional[Path] = None,
    n_states: int = 10_000,
    n_trajs: int = 10_000,
    n_envs: int = 100,
    seed: int = 0,
):
    reward = np.load(reward_path)
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    setup_logging(level="INFO", outdir=outdir, name="aligned_reward_set.log")

    torch.manual_seed(seed)

    env = ExtractDictObWrapper(Miner(reward_weights=reward, num=n_envs), "rgb")
    policy = get_policy(policy_path, actype=env.ac_space, num=n_envs)

    diffs = make_aligned_reward_set(
        reward,
        n_states=n_states,
        n_trajs=n_trajs,
        env=env,
        policy=policy,
        tqdm=True,
        outdir=outdir,
    )
    np.save(outdir / "aligned_reward_set.npy", diffs)


if __name__ == "__main__":
    fire.Fire(main)

import gc
import logging
import time
from itertools import product
from math import ceil
from pathlib import Path
from typing import Final, List, Literal, Optional

import fire  # type: ignore
import numpy as np
import torch
from mrl.dataset.preferences import get_policy
from mrl.envs.util import FEATURE_ENV_NAMES, get_root_env, make_env
from mrl.memprof import get_memory
from mrl.util import (
    is_redundant,
    max_traj_batch_size,
    procgen_rollout_dataset,
    procgen_rollout_features,
    setup_logging,
)
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen.env import ProcgenGym3Env

FEATURES_FILENAME: Final = "ars_features.npy"
ARS_FILENAME: Final = "aligned_reward_set.npy"


def get_features(
    n_states: int,
    n_trajs: int,
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    outdir: Optional[Path],
    overwrite: bool,
    batch_timesteps: int,
    tqdm: bool = False,
) -> np.ndarray:
    root_env = get_root_env(env)

    if (
        not overwrite
        and outdir is not None
        and (feature_file := outdir / FEATURES_FILENAME).is_file()
    ):
        logging.info("Loading features from file")
        features = np.load(feature_file)
        if len(features) >= n_states + n_trajs:
            logging.info(f"n_states={n_states} n_trajs={n_trajs}")
            return np.concatenate((features[:n_states], features[-n_trajs:]))

    logging.info("Generating states")
    state_features = procgen_rollout_features(
        env=env,
        policy=policy,
        timesteps=ceil(n_states / env.num),
        tqdm=tqdm,
    ).reshape(-1, root_env._reward_weights.shape[0])
    assert (
        state_features.shape[0] >= n_states
    ), f"state features shape={state_features.shape} when {n_states} states requested"

    if batch_timesteps == -1:
        one_step = procgen_rollout_dataset(
            env=env,
            policy=get_policy(None, env=env),
            timesteps=1,
            n_trajs=1,
            flags=["feature", "first"],
        )
        assert one_step.features is not None
        nbytes = one_step.features.element_size() * one_step.features.nelement()
        logging.debug(f"one_step nbytes={nbytes}")
        batch_timesteps = max_traj_batch_size(n_trajs, env.num, nbytes)

    logging.info("Generating trajs")
    traj_feature_batches: List[np.ndarray] = []
    current_trajs = 0
    while current_trajs < n_trajs:
        logging.info(f"{current_trajs}/{n_trajs}")
        gc.collect()
        logging.debug(
            f"Before rollout_dataset vm={get_memory()['VmSize']}, peak={get_memory()['VmPeak']}"
        )
        trajs = list(
            procgen_rollout_dataset(
                env=env,
                policy=policy,
                timesteps=batch_timesteps,
                n_trajs=n_trajs - current_trajs,
                flags=["feature", "first"],
                tqdm=tqdm,
            ).trajs()
        )

        traj_features = np.stack([np.sum(traj.features.numpy(), axis=0) for traj in trajs])  # type: ignore
        current_trajs += traj_features.shape[0]
        traj_feature_batches.append(traj_features)
        assert (
            len(traj_features.shape) == 2
        ), f"traj feature has wrong dimension {traj_features.shape}"
        assert (
            traj_features.shape[1] == state_features.shape[1]
        ), f"traj and state feature dims don't match {traj_features.shape}, {state_features.shape}"
        del trajs

    traj_features = np.concatenate(traj_feature_batches)

    assert (
        traj_features.shape[0] >= n_trajs
    ), f"traj features {traj_features.shape} not expected length {n_trajs}"

    features = np.concatenate(
        (state_features[:n_states], traj_features[:n_trajs]), axis=0
    )
    if outdir is not None:
        np.save(outdir / FEATURES_FILENAME, features)

    return features


def make_aligned_reward_set(
    reward: np.ndarray,
    n_states: int,
    n_trajs: int,
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    tqdm: bool = False,
    outdir: Optional[Path] = None,
    overwrite: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    features = get_features(
        n_states=n_states,
        n_trajs=n_trajs,
        env=env,
        policy=policy,
        outdir=outdir,
        overwrite=overwrite,
        tqdm=tqdm,
        batch_timesteps=timesteps,
    )
    assert np.all(np.isfinite(features))
    logging.info(f"Features shape={features.shape}")

    logging.info("Finding non-redundant constraint set")
    start = time.time()

    total = features.shape[0] ** 2 - features.shape[0]
    logging.info(f"{total} total comparisons")

    if (
        not overwrite
        and outdir is not None
        and (diffs_path := outdir / ARS_FILENAME).is_file()
    ):
        logging.info(f"Loading diffs from {diffs_path}")
        diffs = np.load(diffs_path)
    else:
        diffs = np.empty((0, features.shape[1]))

    rng.shuffle(features)
    assert np.all(np.isfinite(features))

    order1 = np.arange(features.shape[0])
    rng.shuffle(order1)
    order2 = np.arange(features.shape[0])
    rng.shuffle(order2)

    last_new = 0
    iterations = 0
    for i, j in product(order1, order2):
        if i == j:
            continue
        iterations += 1

        diff = features[i] - features[j]
        assert np.all(np.isfinite(diff))
        return_diff = reward @ diff
        opinion = np.sign(return_diff)
        if opinion == 0 or np.abs(return_diff) < 1e-8:
            continue
        diff *= opinion
        assert reward @ diff > 1e-8

        try:
            if diffs.shape[0] < 2 or not is_redundant(np.copy(diff), diffs):
                assert diff @ reward > 1e-8
                diffs = np.append(diffs, [diff], axis=0)
                last_new = iterations
                if outdir is not None:
                    np.save(outdir / ARS_FILENAME, diffs)
                logging.info(f"{len(diffs)} total diffs")
        except Exception as e:
            logging.error(f"{e}")
            logging.warning("Unable to solve LP, adding item to set anyway")
            diffs = np.append(diffs, [diff], axis=0)
            last_new = iterations
            if outdir is not None:
                np.save(outdir / ARS_FILENAME, diffs)
            logging.info(f"{len(diffs)} total diffs")
        if total >= 1000:
            if iterations == 1000:
                stop = time.time()
                duration = stop - start
                logging.info(
                    f"First 1000 iterations took {duration:0.1f} seconds. {total} total iters expected to take {duration * total / 1000: 0.1f} seconds."
                )
            if iterations % (total // 1000) == 0:
                stop = time.time()
                duration = stop - start
                logging.info(
                    f"{iterations}/{total} pairs considered, {iterations / total * 100 : 0.2f}%, {duration * total / iterations: 0.1f} seconds remaining, {iterations / duration : 0.1f} diffs per second"
                )

            if iterations - last_new > 1e7:
                logging.info(f"1e7 iterations since last new diff, stopping.")
                break

    return diffs


def main(
    reward_path: Path,
    env_name: FEATURE_ENV_NAMES,
    outdir: Path,
    use_miner_done_feature: bool = False,
    policy_path: Optional[Path] = None,
    n_states: int = 10_000,
    n_trajs: int = 10_000,
    n_envs: int = 100,
    seed: int = 0,
    timesteps: int = -1,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    setup_logging(level=verbosity, outdir=outdir, name="aligned_reward_set.log")

    reward = np.load(reward_path)
    if env_name == "miner" and not use_miner_done_feature:
        reward = np.delete(reward, 1)

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)

    env = make_env(kind=env_name, reward=reward, num=n_envs, rand_seed=seed)
    policy = get_policy(policy_path, env=env)

    diffs = make_aligned_reward_set(
        reward=reward,
        n_states=n_states,
        n_trajs=n_trajs,
        env=env,
        policy=policy,
        tqdm=True,
        outdir=outdir,
        overwrite=overwrite,
        timesteps=timesteps,
        rng=rng,
    )
    np.save(outdir / ARS_FILENAME, diffs)


if __name__ == "__main__":
    fire.Fire(main)

from __future__ import annotations

import gc
import logging
import random
from math import sqrt
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mrl.envs.feature_envs import FeatureEnv  # type: ignore
from mrl.envs.util import FEATURE_ENV_NAMES, get_root_env, make_env
from mrl.folders import HyperFolders
from mrl.util import (
    get_policy,
    max_shard_index,
    max_state_batch_size,
    max_traj_batch_size,
    normalize_diffs,
    np_gather,
    np_remove,
    procgen_rollout_dataset,
    procgen_rollout_features,
    setup_logging,
)
from phasic_policy_gradient.ppg import PhasicValueModel


def gen_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    outname: str,
    n_prefs: int = 1000,
    n_calibration_prefs: int = 100,
    n_envs: int = 100,
    flip_prob: float = 0.2,
    init_state_temp: float = 1.0,
    init_traj_temp: float = 10.0,
    sensitivity: float = 0.01,
    normalize_step: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    overwrite: bool = False,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[Path, Path]:
    rootdir = Path(rootdir)
    outdir = rootdir / "prefs"
    setup_logging(level=verbosity, outdir=outdir)

    reward = np.load(rootdir / "reward.npy")
    rng = np.random.default_rng(seed)

    generator = Generator(
        env_name=env,
        n_envs=n_envs,
        normalize_features=normalize_step,
        policy_paths=[None],
        rng=rng,
    )

    logging.info("Searching for state temperature")
    state_temp = calibrate_states(
        reward=reward,
        generator=generator,
        target_flip_prob=flip_prob,
        n_states=n_calibration_prefs,
        normalize_differences=normalize_differences,
        init_temperature=init_state_temp,
        sensitivity=sensitivity,
    )
    logging.info(f"Using temp={state_temp} for states")

    logging.info("Searching for trajectory temperature")
    traj_temp = calibrate_trajs(
        reward=reward,
        generator=generator,
        target_flip_prob=flip_prob,
        n_trajs=n_calibration_prefs,
        normalize_differences=normalize_differences,
        init_temperature=init_traj_temp,
        sensitivity=sensitivity,
    )
    logging.info(f"Using temp={traj_temp} for trajs")

    logging.info("Generating state preferences")
    state_path = gen_state_preferences(
        rootdir=rootdir,
        env=env,
        n_states=n_prefs,
        n_parallel_envs=n_envs,
        outname=outname,
        temperature=state_temp,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        overwrite=overwrite,
        verbosity=verbosity,
    )

    logging.info("Generating trajectory preferenes")
    traj_path = gen_traj_preferences(
        rootdir=rootdir,
        env=env,
        n_trajs=n_prefs,
        n_parallel_envs=n_envs,
        outname=outname,
        temperature=traj_temp,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        overwrite=overwrite,
        verbosity=verbosity,
    )
    return state_path, traj_path


def gen_state_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    n_states: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_step_features: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Path:
    rootdir = Path(rootdir)
    folders = HyperFolders(rootdir / "prefs", schema=["modality", "temp"])
    outdir = folders.add_experiment({"modality": "state", "temp": temperature})

    setup_logging(level=verbosity, outdir=outdir, name=f"{outname}.log")

    outname = str(outname)
    reward = np.load(rootdir / "reward.npy")

    rng = np.random.default_rng()

    generator = Generator(
        env_name=env,
        n_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=list(set([policy_path, None])),
        rng=rng,
        tqdm=True,
    )

    batch_timesteps = max_state_batch_size(
        n_states=n_states,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    n_prefs = 0

    outpath = outdir / (outname + ".features.npy")
    if overwrite:
        np_remove(outdir, outname)

    if outpath.exists():
        current_prefs = np.load(outpath)
        n_prefs += len(current_prefs)
        logging.info(f"Starting with {n_prefs} existing preferences")
    else:
        current_prefs = None
        if not overwrite:
            logging.info(f"Did not find existing preferences at {outpath}")

    if n_prefs >= n_states:
        logging.warning(f"No new states needed for {outdir}")
        return outdir / outname

    feature_batches: List[np.ndarray] = []
    flip_prob_batches: List[np.ndarray] = []

    while n_prefs < n_states:
        oriented_features, probs = state_collect_step(
            generator, batch_timesteps, reward, temperature, normalize_differences
        )
        if oriented_features is None or probs is None:
            continue
        feature_batches.append(oriented_features)
        flip_prob_batches.append(probs)
        n_prefs += oriented_features.shape[0]

        logging.info(f"Collected {n_prefs} of {n_states} preferences")

    features = np.concatenate(feature_batches)
    assert len(features.shape) == 3
    assert features.shape[1] == 2
    np.save(outdir / (outname + ".features.npy"), features)

    flip_probs = np.concatenate(flip_prob_batches)
    np.save(outdir / (outname + ".flip-probs.npy"), flip_probs)

    plt.hist(flip_probs, bins=100)
    plt.title(f"Histogram of noise flip probabilities (temp={temperature:0.5f})")
    plt.xlabel("Flip Probability")
    plt.savefig(outdir / (outname + ".flip-probs.png"))
    plt.close()

    return outdir / outname


def calibrate_states(
    reward: np.ndarray,
    generator: Generator,
    target_flip_prob: float,
    n_states: int,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ],
    init_temperature: float = 1.0,
    sensitivity: float = 0.05,
) -> float:
    batch_timesteps = max_state_batch_size(
        n_states=n_states,
        n_parallel_envs=generator.env.num,
        step_nbytes=generator.step_nbytes,
    )

    temperature = init_temperature
    max_temp = temperature
    min_temp = temperature
    first = True
    while first or abs(average_flip_prob - target_flip_prob) > sensitivity:
        if not first:
            # Log_2 binary search over temperature values
            if target_flip_prob > average_flip_prob:
                min_temp = temperature
                if max_temp == temperature:
                    temperature *= 2.0
                    max_temp = temperature
                else:
                    temperature = sqrt(min_temp * max_temp)
            else:
                max_temp = temperature
                if min_temp == temperature:
                    temperature /= 2.0
                    min_temp = temperature
                else:
                    temperature = sqrt(min_temp * max_temp)
        first = False

        logging.debug(f"temp={temperature}")

        flip_prob_batches: List[np.ndarray] = []
        n_diffs = 0
        while n_diffs < n_states:
            logging.debug(f"{n_diffs}/{n_states}")
            _, probs = state_collect_step(
                generator, batch_timesteps, reward, temperature, normalize_differences
            )
            if probs is None:
                continue
            flip_prob_batches.append(probs)
            n_diffs += probs.shape[0]
        flip_probs = np.concatenate(flip_prob_batches)
        average_flip_prob: float = 0.5 - np.mean(np.abs(flip_probs - 0.5))
        logging.debug(f"average_flip_prob={average_flip_prob}")

    return temperature


def state_collect_step(
    generator: Generator,
    timesteps: int,
    reward: np.ndarray,
    temperature: float,
    normalize_differences,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    feature_a, feature_b = generator.gen_state_pairs(timesteps=timesteps)
    features = np.stack((feature_a, feature_b), axis=1)
    feature_diffs = normalize_diffs(features, mode=normalize_differences)

    nonempty_rows = np.any(feature_diffs != 0, axis=1)
    features = features[nonempty_rows]
    feature_diffs = feature_diffs[nonempty_rows]

    if feature_diffs.shape[0] == 0:
        return (None, None)

    oriented_features, probs = orient_features(
        features=features,
        diffs=feature_diffs,
        temperature=temperature,
        reward=reward,
        rng=generator.rng,
    )
    if oriented_features.shape[0] == 0:
        return (None, None)
    return oriented_features, probs


def gen_traj_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    n_trajs: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_step_features: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Path:
    rootdir = Path(rootdir)
    folders = HyperFolders(rootdir / "prefs", schema=["modality", "temp"])
    outdir = folders.add_experiment({"modality": "traj", "temp": temperature})

    setup_logging(level=verbosity, outdir=outdir, name=f"{outname}.log")

    outname = str(outname)
    reward = np.load(rootdir / "reward.npy")

    rng = np.random.default_rng()

    generator = Generator(
        env_name=env,
        n_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=[Path(policy_path), None] if policy_path is not None else [None],
        rng=rng,
        tqdm=True,
    )

    batch_timesteps = max_traj_batch_size(
        n_trajs=n_trajs,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    current_trajs = 0
    if overwrite:
        np_remove(outdir, outname)
        collection_batch = -1
    else:
        collection_batch = max_shard_index(outdir, outname + ".features")

        if collection_batch >= 0:
            data = np_gather(outdir, outname + ".features")

            current_trajs += data.shape[0]
            del data
            gc.collect()
            logging.info(f"Starting with {current_trajs} preferences")
        else:
            logging.info(f"No existing data found at {outdir} with name {outname}")
    collection_batch += 1

    if current_trajs >= n_trajs:
        logging.warning(f"No new trajectories needed for {outdir}")
        return outdir / outname

    while current_trajs < n_trajs:
        logging.info(
            f"Asking for {n_trajs - current_trajs} trajs or {batch_timesteps} timesteps"
        )

        features, probs = traj_collect_step(
            generator,
            batch_timesteps,
            n_trajs - current_trajs,
            reward,
            temperature,
            normalize_differences,
        )

        diffs = features[:, 0] - features[:, 1]
        assert not np.any(np.all(diffs == 0, axis=1))
        assert features.shape[1] == 2
        np.save(outdir / f"{outname}.features.{collection_batch}.npy", features)

        probs_file = outdir / f"{outname}.flip-probs.{collection_batch}.npy"
        np.save(probs_file, probs)

        plt.hist(probs, bins=100)
        plt.title(f"Histogram of noise flip probabilities (temp={temperature:0.5f})")
        plt.xlabel("Flip Probability")
        plt.savefig(outdir / (outname + ".flip-probs.png"))
        plt.close()

        collection_batch += 1
        current_trajs += features.shape[0]
        del features
        del probs
        gc.collect()

    return outdir / outname


def calibrate_trajs(
    reward: np.ndarray,
    generator: Generator,
    target_flip_prob: float,
    n_trajs: int,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ],
    init_temperature: float = 1.0,
    sensitivity: float = 0.05,
) -> float:
    batch_timesteps = max_traj_batch_size(
        n_trajs=n_trajs,
        n_parallel_envs=generator.env.num,
        step_nbytes=generator.step_nbytes,
    )

    temperature = init_temperature
    max_temp = temperature
    min_temp = temperature
    first = True
    while first or abs(average_flip_prob - target_flip_prob) > sensitivity:
        if not first:
            # Log_2 binary search over temperature values
            if target_flip_prob > average_flip_prob:
                min_temp = temperature
                if max_temp == temperature:
                    temperature *= 2.0
                    max_temp = temperature
                else:
                    temperature = sqrt(min_temp * max_temp)
            else:
                max_temp = temperature
                if min_temp == temperature:
                    temperature /= 2.0
                    min_temp = temperature
                else:
                    temperature = sqrt(min_temp * max_temp)
        first = False
        logging.debug(f"temp={temperature}")

        probs = np.empty((0,))
        n_diffs = 0
        while n_diffs < n_trajs:
            logging.debug(f"{n_diffs}/{n_trajs}")
            _, probs_batch = traj_collect_step(
                generator,
                batch_timesteps,
                n_trajs - n_diffs,
                reward,
                temperature,
                normalize_differences,
            )
            assert (
                len(probs_batch.shape) == 1
            ), f"probs batch shape {probs_batch.shape} not expected"
            n_diffs += probs_batch.shape[0]
            probs = np.concatenate((probs, probs_batch))
        average_flip_prob: float = 0.5 - np.mean(np.abs(probs - 0.5))
        logging.debug(f"average_flip_prob={average_flip_prob}")
    return temperature


def traj_collect_step(
    generator: Generator,
    timesteps: int,
    n_trajs: int,
    reward: np.ndarray,
    temperature: float,
    normalize_differences,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_batch: List[np.ndarray] = []
    probs_batch: List[np.ndarray] = []
    for traj_a, traj_b in zip(
        *generator.gen_traj_pairs(timesteps=timesteps, n_trajs=n_trajs)
    ):
        assert traj_a.features is not None and traj_b.features is not None
        feature_a: np.ndarray = traj_a.features.numpy().sum(axis=0, keepdims=True)
        feature_b: np.ndarray = traj_b.features.numpy().sum(axis=0, keepdims=True)
        features = np.stack((feature_a, feature_b), axis=1)
        assert features.shape[:2] == (1, 2)

        feature_diff = normalize_diffs(features, normalize_differences)
        assert len(feature_diff.shape) == 2

        if not np.any(feature_diff > 1e-8):
            continue

        oriented_features, probs = orient_features(
            features=features,
            diffs=feature_diff,
            temperature=temperature,
            reward=reward,
            rng=generator.rng,
        )
        if oriented_features.shape[0] == 0 or oriented_features.shape[1] == 0:
            continue

        logging.debug(
            f"Raw={feature_a}, {feature_b}, raw_diff={feature_diff}, oriented={oriented_features}, oriented_diff={oriented_features[:,0] - oriented_features[:,1]}"
        )
        assert not np.allclose(
            oriented_features[:, 0], oriented_features[:, 1]
        ), f"equal features not filtered"

        feature_batch.append(oriented_features)
        probs_batch.append(probs)
    return np.concatenate(feature_batch, axis=0), np.concatenate(probs_batch)


def orient_features(
    features: np.ndarray,
    diffs: np.ndarray,
    temperature: float,
    reward: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(features.shape) == 3, f"features shape={features.shape} not expected"
    assert len(diffs.shape) == 2, f"diff shape={diffs.shape} not expected"
    assert diffs.shape[0] == features.shape[0]
    assert diffs.shape[1] == features.shape[2]

    strengths = diffs @ reward
    logging.debug(f"strength={strengths}")
    nonzero_rows = np.abs(strengths) > 1e-8
    features = features[nonzero_rows]
    diffs = diffs[nonzero_rows]
    strengths = strengths[nonzero_rows]

    if temperature == 0.0:
        opinions = np.sign(strengths)
        for i in range(len(features)):
            if opinions[i] == -1:
                features[i, [0, 1]] = features[i, [1, 0]]

        prob_first_better = np.maximum(opinions, 0)
    else:
        prob_first_better = 1.0 / (1.0 + np.exp(-strengths / temperature))
        second_better = rng.random(size=diffs.shape[0]) > prob_first_better
        for i in range(len(features)):
            if second_better[i]:
                features[i, [0, 1]] = features[i, [1, 0]]

    return features, prob_first_better


class Generator:
    def __init__(
        self,
        env_name: FEATURE_ENV_NAMES,
        n_envs: int,
        normalize_features: bool,
        policy_paths: List[Optional[Path]],
        rng: np.random.Generator,
        tqdm: bool = False,
    ) -> None:
        self.env = make_env(
            name=env_name,
            reward=0,
            num=n_envs,
            normalize_features=normalize_features,
        )
        self.root_env = cast(FeatureEnv, get_root_env(self.env))
        assert isinstance(self.root_env, FeatureEnv)

        self.policies = [get_policy(path, env=self.env) for path in policy_paths]

        feature = procgen_rollout_features(
            env=self.env,
            policy=self.policies[0],
            timesteps=1,
        )
        self.step_nbytes = feature.nbytes
        logging.info(f"One timestep size={self.step_nbytes}")

        self.rng = rng
        self.tqdm = tqdm

    def select_policy_pair(
        self, policy_indices: Optional[Tuple[int, int]] = None
    ) -> Tuple[PhasicValueModel, PhasicValueModel]:
        if policy_indices is None:
            policy_a = random.choice(self.policies)
            policy_b = random.choice(self.policies)
        else:
            policy_a, policy_b = (
                self.policies[policy_indices[0]],
                self.policies[policy_indices[1]],
            )

        return policy_a, policy_b

    def gen_state_pairs(
        self, timesteps: int, policy_indices: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        policy_a, policy_b = self.select_policy_pair(policy_indices)
        features_a = procgen_rollout_features(
            env=self.env,
            policy=policy_a,
            timesteps=timesteps,
            tqdm=self.tqdm,
        ).reshape(-1, self.root_env._reward_weights.shape[0])

        features_b = procgen_rollout_features(
            env=self.env,
            policy=policy_b,
            timesteps=timesteps,
            tqdm=self.tqdm,
        ).reshape(-1, self.root_env._reward_weights.shape[0])

        return features_a, features_b

    def gen_traj_pairs(
        self,
        timesteps: int,
        n_trajs: int,
        policy_indices: Optional[Tuple[int, int]] = None,
    ):
        policy_a, policy_b = self.select_policy_pair(policy_indices)
        data_a = procgen_rollout_dataset(
            env=self.env,
            policy=policy_a,
            timesteps=timesteps,
            n_trajs=n_trajs,
            flags=["feature", "first"],
            tqdm=self.tqdm,
        )
        data_b = procgen_rollout_dataset(
            env=self.env,
            policy=policy_b,
            timesteps=timesteps,
            n_trajs=n_trajs,
            flags=["feature", "first"],
            tqdm=self.tqdm,
        )

        return data_a.trajs(), data_b.trajs()


if __name__ == "__main__":
    fire.Fire(
        {
            "state": gen_state_preferences,
            "traj": gen_traj_preferences,
            "both": gen_preferences,
        },
    )

from __future__ import annotations

import logging
import random
from math import sqrt
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, cast

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mrl.dataset.roller import procgen_rollout_dataset, procgen_rollout_features
from mrl.envs.feature_envs import FeatureEnv  # type: ignore
from mrl.envs.util import FEATURE_ENV_NAMES, get_root_env, make_env
from mrl.folders import HyperFolders
from mrl.util import (
    get_angle,
    get_policy,
    max_state_batch_size,
    max_traj_batch_size,
    normalize_diffs,
    np_remove,
    setup_logging,
)
from phasic_policy_gradient.ppg import PhasicValueModel


def gen_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    outname: str,
    prefs_per_trial: int = 1000,
    n_trials: int = 1,
    n_calibration_prefs: int = 100,
    n_envs: int = 100,
    flip_prob: float = 0.2,
    init_state_temp: float = 1.0,
    init_traj_temp: float = 10.0,
    sensitivity: float = 0.01,
    deduplicate: bool = False,
    normalize_step: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    append: bool = False,
    overwrite: bool = False,
    seed: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[Tuple[Path, int], Tuple[Path, int]]:
    rootdir = Path(rootdir)
    outdir = rootdir / "prefs"
    setup_logging(level=verbosity, outdir=outdir, multiple_files=False)

    if append and overwrite:
        raise ValueError("Cannot specify both append and overwrite.")

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
        prefs_per_trial=prefs_per_trial,
        n_trials=n_trials,
        n_parallel_envs=n_envs,
        outname=outname,
        temperature=state_temp,
        deduplicate=deduplicate,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        append=append,
        overwrite=overwrite,
        verbosity=verbosity,
    )

    logging.info("Generating trajectory preferenes")
    traj_path = gen_traj_preferences(
        rootdir=rootdir,
        env=env,
        prefs_per_trial=prefs_per_trial,
        n_trials=n_trials,
        n_parallel_envs=n_envs,
        outname=outname,
        temperature=traj_temp,
        deduplicate=deduplicate,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        append=append,
        overwrite=overwrite,
        verbosity=verbosity,
    )
    return state_path, traj_path


# TODO: Factor out common code between this and traj preferences
def gen_state_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    prefs_per_trial: int,
    n_trials: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    deduplicate: bool = False,
    normalize_step_features: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    append: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[Path, int]:
    rootdir = Path(rootdir)

    if append and overwrite:
        raise ValueError("Cannot append and overwrite")

    folders = HyperFolders(
        rootdir / "prefs", schema=["modality", "temp", "dedup", "norm"]
    )
    outdir = folders.add_experiment(
        {
            "modality": "state",
            "temp": temperature,
            "dedup": "dedup" if deduplicate else "no-dedup",
            "norm": f"norm-{normalize_differences}",
        }
    )
    outname = str(outname)

    setup_logging(level=verbosity, outdir=outdir, name=f"{outname}.log", append=append)

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

    states_per_batch = prefs_per_trial
    batch_timesteps = max_state_batch_size(
        n_states=states_per_batch,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    step = generator.gen_state_pairs(1)
    feature_dim = step[0].shape[1]

    if overwrite:
        logging.info(f"Overwriting data at {outdir}")
        np_remove(outdir, name="*")

    # (trial, pref index, first/second, features)
    features = np.empty(
        (n_trials, prefs_per_trial, 2, feature_dim), dtype=step[0].dtype
    )
    flip_probs = np.empty((n_trials, prefs_per_trial), dtype=np.float32)

    current_prefs, current_flip_probs = load_data(outdir, outname)

    if current_prefs is not None and current_flip_probs is not None:
        if append and current_prefs.shape[1] != prefs_per_trial:
            raise ValueError(
                f"Existing data uses different amount of prefs per trial, unsupported."
            )

        if len(current_flip_probs.shape) == 1:
            # Old flip_probs used to be one dimensional, convert to new format
            current_flip_probs = current_flip_probs.reshape(current_prefs.shape[:2])

        if not append:
            if (
                current_prefs.shape[0] >= n_trials
                and current_prefs.shape[1] >= prefs_per_trial
            ):
                logging.warning(f"No new states needed for {outdir}")
                return outdir / outname, 0
            else:
                new_trials = max(0, n_trials - current_prefs.shape[0])
                new_prefs_per_trial = max(0, prefs_per_trial - current_prefs.shape[1])
                features = np.pad(
                    current_prefs,
                    [(0, new_trials), (0, new_prefs_per_trial), (0, 0), (0, 0)],
                    "constant",
                    constant_values=0,
                )

                if len(current_flip_probs.shape) == 1:
                    # Old flip_probs used to be one dimensional, convert to new format
                    current_flip_probs = current_flip_probs.reshape(
                        current_prefs.shape[:2]
                    )
                flip_probs = np.pad(
                    current_flip_probs,
                    [(0, new_trials), (0, new_prefs_per_trial)],
                    "constant",
                    constant_values=0,
                )

    for trial in range(n_trials):
        logging.info(
            f"Collecting state comparisons for trial {trial + 1} of {n_trials}"
        )

        trial_prefs = 0
        if not append and current_prefs is not None and trial < current_prefs.shape[0]:
            # If there is already some data in this row, start at the end of that existing data.
            trial_prefs = current_prefs.shape[1]

        while trial_prefs < prefs_per_trial:
            new_features, new_probs = state_collect_step(
                generator, batch_timesteps, reward, temperature, normalize_differences
            )
            if new_features is None or new_probs is None:
                continue

            indices = (
                find_soft_unique_indices(
                    new=new_features[:, 0] - new_features[:, 1],
                    core=features[trial, :trial_prefs, 0]
                    - features[trial, :trial_prefs, 1],
                )
                if deduplicate
                else np.ones_like(new_probs, dtype=bool)
            )

            if indices.shape[0] == 0:
                states_per_batch *= 10
                batch_timesteps = max_state_batch_size(
                    n_states=states_per_batch,
                    n_parallel_envs=n_parallel_envs,
                    step_nbytes=generator.step_nbytes,
                )

            logging.debug(f"{indices.shape[0]=}")

            end_index = min(trial_prefs + indices.shape[0], prefs_per_trial)
            overflow_index = end_index - trial_prefs

            logging.debug(
                f"{features.shape=}, {trial_prefs=}, {end_index=}, {indices.shape=}, {overflow_index=}"
            )

            features[trial, trial_prefs:end_index] = new_features[indices][
                :overflow_index
            ]
            flip_probs[trial, trial_prefs:end_index] = new_probs[indices][
                :overflow_index
            ]

            trial_prefs = end_index

            logging.info(f"Collected {trial_prefs} of {prefs_per_trial} preferences")

    start_trial = 0
    if append and current_prefs is not None and current_flip_probs is not None:
        features = np.concatenate((current_prefs, features), axis=0)
        flip_probs = np.concatenate((current_flip_probs, flip_probs), axis=0)
        start_trial = current_prefs.shape[0]

    np.save(outdir / (outname + ".features.npy"), features)
    np.save(outdir / (outname + ".flip-probs.npy"), flip_probs)

    plt.hist(flip_probs.flatten(), bins=100)
    plt.title(f"Histogram of noise flip probabilities (temp={temperature:0.5f})")
    plt.xlabel("Flip Probability")
    plt.xlim((0, 1))
    plt.savefig(outdir / (outname + ".flip-probs.png"))
    plt.close()

    return outdir / outname, start_trial


def load_data(
    outdir: Path, outname: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    logging.info(f"Loading data from {outdir}/{outname}")
    features_outpath = outdir / (outname + ".features.npy")
    flip_probs_outpath = outdir / (outname + ".flip-probs.npy")

    current_prefs = None
    current_flip_probs = None
    if features_outpath.exists() and flip_probs_outpath.exists():
        current_prefs = np.load(features_outpath)
        current_flip_probs = np.load(flip_probs_outpath)

        if len(current_flip_probs.shape) == 1:
            # Old flip_probs used to be one dimensional, convert to new format
            current_flip_probs = current_flip_probs.reshape(current_prefs.shape[:2])

        if current_prefs.shape[:2] != current_flip_probs.shape[:2]:
            logging.warning(
                "Different number of saved prefs and flip probs, deleting both."
            )
            np_remove(outdir, outname)
        else:
            logging.info(
                f"Found {current_prefs.shape[0]} trials and {current_prefs.shape[1]} prefs per trial."
            )
    elif features_outpath.exists() or flip_probs_outpath.exists():
        logging.warning("Found features or flip_probs but not both, removing.")
        np_remove(outdir, outname)
    else:
        logging.info(
            f"Did not find existing preferences at {outdir} with name {outname}"
        )

    return current_prefs, current_flip_probs


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
    prefs_per_trial: int,
    n_trials: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    deduplicate: bool = False,
    normalize_step_features: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    seed: Optional[int] = None,
    append: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Tuple[Path, int]:
    rootdir = Path(rootdir)

    if append and overwrite:
        raise ValueError("Cannot append and overwrite")

    folders = HyperFolders(
        rootdir / "prefs", schema=["modality", "temp", "dedup", "norm"]
    )
    outdir = folders.add_experiment(
        {
            "modality": "traj",
            "temp": temperature,
            "dedup": "dedup" if deduplicate else "no-dedup",
            "norm": f"norm-{normalize_differences}",
        }
    )
    outname = str(outname)

    setup_logging(level=verbosity, outdir=outdir, name=f"{outname}.log", append=append)

    reward = np.load(rootdir / "reward.npy")

    rng = np.random.default_rng(seed=seed)

    generator = Generator(
        env_name=env,
        n_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=[Path(policy_path), None] if policy_path is not None else [None],
        rng=rng,
        tqdm=True,
    )

    trajs_per_batch = prefs_per_trial
    batch_timesteps = max_traj_batch_size(
        n_trajs=prefs_per_trial,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    step = generator.gen_state_pairs(1)
    feature_dim = step[0].shape[1]

    if overwrite:
        np_remove(outdir, name="*")

    # (trial, pref index, first/second, features)
    features = np.empty(
        (n_trials, prefs_per_trial, 2, feature_dim), dtype=step[0].dtype
    )
    flip_probs = np.empty((n_trials, prefs_per_trial), dtype=np.float32)

    current_prefs, current_flip_probs = load_data(outdir, outname)

    if current_prefs is not None and current_flip_probs is not None:
        if append and current_prefs.shape[1] != prefs_per_trial:
            raise ValueError(
                f"Existing data uses different amount of prefs per trial, unsupported."
            )

        if len(current_flip_probs.shape) == 1:
            # Old flip_probs used to be one dimensional, convert to new format
            current_flip_probs = current_flip_probs.reshape(current_prefs.shape[:2])

        if not append:
            if (
                current_prefs.shape[0] >= n_trials
                and current_prefs.shape[1] >= prefs_per_trial
            ):
                logging.warning(f"No new states needed for {outdir}")
                return outdir / outname, 0
            else:
                new_trials = max(0, n_trials - current_prefs.shape[0])
                new_prefs_per_trial = max(0, prefs_per_trial - current_prefs.shape[1])
                features = np.pad(
                    current_prefs,
                    [(0, new_trials), (0, new_prefs_per_trial), (0, 0), (0, 0)],
                    "constant",
                    constant_values=0,
                )

                flip_probs = np.pad(
                    current_flip_probs,
                    [(0, new_trials), (0, new_prefs_per_trial)],
                    "constant",
                    constant_values=0,
                )

    for trial in range(n_trials):
        logging.info(f"Collecting traj comparisons for trial {trial + 1} of {n_trials}")
        trial_prefs = 0
        if not append and current_prefs is not None and trial < current_prefs.shape[0]:
            # If there is already some data in this row, start at the end of that existing data.
            trial_prefs = current_prefs.shape[1]

        while trial_prefs < prefs_per_trial:
            prefs_remaining = prefs_per_trial - trial_prefs
            logging.info(
                f"Asking for {prefs_remaining} trajs or {batch_timesteps} timesteps"
            )

            new_features, new_probs = traj_collect_step(
                generator=generator,
                timesteps=batch_timesteps,
                n_trajs=prefs_remaining,
                reward=reward,
                temperature=temperature,
                normalize_differences=normalize_differences,
            )

            new_diffs = new_features[:, 0] - new_features[:, 1]
            old_diffs = (
                features[trial, :trial_prefs, 0] - features[trial, :trial_prefs, 1]
            )
            assert not np.any(np.all(new_diffs == 0, axis=1))
            assert new_features.shape[1] == 2

            logging.debug(f"{features.shape=}")
            indices = (
                find_soft_unique_indices(new=new_diffs, core=old_diffs)
                if deduplicate
                else np.ones_like(new_probs, dtype=bool)
            )

            if indices.shape[0] == 0:
                # If no non-duplicate preferences were found, look for 10 times more next time
                trajs_per_batch *= 10
                batch_timesteps = max_traj_batch_size(
                    n_trajs=trajs_per_batch,
                    n_parallel_envs=n_parallel_envs,
                    step_nbytes=generator.step_nbytes,
                )

            end_index = min(trial_prefs + indices.shape[0], prefs_per_trial)
            overflow_index = end_index - trial_prefs

            features[trial, trial_prefs:end_index] = new_features[indices][
                :overflow_index
            ]
            flip_probs[trial, trial_prefs:end_index] = new_probs[indices][
                :overflow_index
            ]

            trial_prefs = end_index
            logging.info(f"Collected {trial_prefs} of {prefs_per_trial} preferences")

    start_trial = 0
    if append and current_prefs is not None and current_flip_probs is not None:
        features = np.concatenate((current_prefs, features), axis=0)
        flip_probs = np.concatenate((current_flip_probs, flip_probs), axis=0)
        start_trial = current_prefs.shape[0]

    np.save(outdir / (outname + ".features.npy"), features)
    np.save(outdir / (outname + ".flip-probs.npy"), flip_probs)

    plt.hist(flip_probs, bins=100)
    plt.title(f"Histogram of noise flip probabilities (temp={temperature:0.5f})")
    plt.xlabel("Flip Probability")
    plt.xlim((0, 1))
    plt.savefig(outdir / (outname + ".flip-probs.png"))
    plt.close()

    return outdir / outname, start_trial


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
        feature_a: np.ndarray = traj_a.features.sum(axis=0, keepdims=True)
        feature_b: np.ndarray = traj_b.features.sum(axis=0, keepdims=True)
        features = np.stack((feature_a, feature_b), axis=1)
        assert features.shape[:2] == (1, 2)

        feature_diff = normalize_diffs(features, normalize_differences)
        assert len(feature_diff.shape) == 2

        if not np.any(np.abs(feature_diff) > 1e-8):
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
        assert np.array_equal(oriented_features, features) or np.array_equal(
            oriented_features[:, [1, 0]], features
        )

        feature_batch.append(oriented_features)
        probs_batch.append(probs)
    return np.concatenate(feature_batch, axis=0), np.concatenate(probs_batch)


def find_soft_unique_indices(
    new: np.ndarray,
    core: Optional[np.ndarray] = None,
    dist: Callable[[np.ndarray, np.ndarray], float] = get_angle,
    epsilon: float = 1e-3,
) -> np.ndarray:
    if core is None:
        core = np.empty((0, *new.shape[1:]), dtype=new.dtype)
    assert (
        new.shape[1:] == core.shape[1:]
    ), f"New {new.shape} and core {core.shape} shapes don't match"
    indices: List[int] = []
    unique_rows = core
    has_zero = bool(np.any(np.all(unique_rows == 0, axis=1)))
    for i, row in enumerate(new):
        if np.all(row == 0):
            if not has_zero:
                indices.append(i)
                has_zero = True
        elif all(
            (d := dist(row, other) >= epsilon) and np.isfinite(d)
            for other in unique_rows
        ):
            indices.append(i)
            unique_rows = np.concatenate((unique_rows, [row]))
    return np.array(indices, dtype=int)


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
        assert data_a.data["features"] is not None
        assert data_b.data["features"] is not None

        return data_a.trajs(), data_b.trajs()


if __name__ == "__main__":
    fire.Fire(
        {
            "state": gen_state_preferences,
            "traj": gen_traj_preferences,
            "both": gen_preferences,
        },
    )

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from mrl.dataset.roller import procgen_rollout_dataset, procgen_rollout_features
from mrl.envs.feature_envs import FeatureEnv  # type: ignore
from mrl.envs.util import FEATURE_ENV_NAMES, get_root_env, make_env
from mrl.folders import HyperFolders
from mrl.util import (
    NORM_DIFF_MODES,
    dump,
    get_angle,
    get_policy,
    load,
    max_state_batch_size,
    max_traj_batch_size,
    normalize_diffs,
    np_remove,
    setup_logging,
)
from phasic_policy_gradient.ppg import PhasicValueModel


@dataclass(frozen=True)
class TemperatureSearchKey:
    flip_prob: float
    modality: Literal["state", "traj"]
    deduplicate: bool
    normalize_differences: NORM_DIFF_MODES


def gen_preferences_flip_prob(
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
    normalize_differences: NORM_DIFF_MODES = None,
    max_length: Optional[int] = None,
    append: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[Tuple[Path, int], Tuple[Path, int]]:
    rootdir = Path(rootdir)
    outdir = rootdir / "prefs"
    setup_logging(level=verbosity, outdir=outdir, multiple_files=False)

    if append and overwrite:
        raise ValueError("Cannot specify both append and overwrite.")

    reward = np.load(rootdir / "reward.npy")

    generator = Generator(
        env_name=env,
        n_envs=n_envs,
        normalize_features=normalize_step,
        policy_paths=[None],
        rng=rng,
    )

    temp_search_results_path = rootdir / "search_results.pkl"
    temp_search_results: Dict[TemperatureSearchKey, float] = {}
    if temp_search_results_path.exists():
        temp_search_results = load(temp_search_results_path)

    state_key = TemperatureSearchKey(
        flip_prob=flip_prob,
        modality="state",
        deduplicate=deduplicate,
        normalize_differences=normalize_differences,
    )
    if overwrite:
        del temp_search_results[state_key]
    state_temp = temp_search_results.get(state_key, None)
    if state_temp is None:
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
        temp_search_results[state_key] = state_temp
    logging.info(f"Using temp={state_temp} for states")

    traj_key = TemperatureSearchKey(
        flip_prob=flip_prob,
        modality="traj",
        deduplicate=deduplicate,
        normalize_differences=normalize_differences,
    )
    if overwrite:
        del temp_search_results[traj_key]
    traj_temp = temp_search_results.get(traj_key, None)
    if traj_temp is None:
        logging.info("Searching for trajectory temperature")
        traj_temp = calibrate_trajs(
            reward=reward,
            generator=generator,
            target_flip_prob=flip_prob,
            n_trajs=n_calibration_prefs,
            normalize_differences=normalize_differences,
            max_length=max_length,
            init_temperature=init_traj_temp,
            sensitivity=sensitivity,
            rng=rng,
        )
        temp_search_results[traj_key] = traj_temp
    logging.info(f"Using temp={traj_temp} for trajs")

    dump(temp_search_results, temp_search_results_path)

    logging.info("Generating state preferences")
    state_path = gen_preferences(
        rootdir=rootdir,
        env=env,
        modality="state",
        prefs_per_trial=prefs_per_trial,
        n_trials=n_trials,
        n_parallel_envs=n_envs,
        outname=outname,
        max_length=max_length,
        temperature=state_temp,
        deduplicate=deduplicate,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        append=append,
        overwrite=overwrite,
        verbosity=verbosity,
    )

    logging.info("Generating trajectory preferences")
    traj_path = gen_preferences(
        rootdir=rootdir,
        env=env,
        modality="traj",
        prefs_per_trial=prefs_per_trial,
        n_trials=n_trials,
        n_parallel_envs=n_envs,
        outname=outname,
        max_length=max_length,
        temperature=traj_temp,
        deduplicate=deduplicate,
        normalize_step_features=normalize_step,
        normalize_differences=normalize_differences,
        append=append,
        overwrite=overwrite,
        verbosity=verbosity,
    )
    return state_path, traj_path


def gen_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    modality: Literal["state", "traj"],
    prefs_per_trial: int,
    n_trials: int,
    n_parallel_envs: int,
    outname: str,
    max_length: Optional[int] = None,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    deduplicate: bool = False,
    normalize_step_features: bool = False,
    normalize_differences: NORM_DIFF_MODES = None,
    append: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
    rng: np.random.Generator = np.random.default_rng(),
):
    rootdir = Path(rootdir)

    if append and overwrite:
        raise ValueError("Cannot append and overwrite")

    outdir = setup_outdir(
        rootdir, modality, max_length, temperature, deduplicate, normalize_differences
    )
    features_outpath = outdir / (outname + ".features.npy")

    setup_logging(level=verbosity, outdir=outdir, name=f"{outname}.log", append=append)

    reward = np.load(rootdir / "reward.npy")

    generator = Generator(
        env_name=env,
        n_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=list(set([policy_path, None])),
        rng=rng,
        tqdm=True,
    )

    prefs_per_batch = prefs_per_trial
    batch_timesteps = max_batch_size(
        modality, prefs_per_trial, n_parallel_envs, generator, prefs_per_batch
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
                logging.warning(f"No new prefs needed for {outdir}")
                return features_outpath, 0
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
            f"Collecting {modality} comparisons for trial {trial + 1} of {n_trials}"
        )

        trial_prefs = 0
        if not append and current_prefs is not None and trial < current_prefs.shape[0]:
            # If there is already some data in this row, start at the end of that existing data.
            trial_prefs = current_prefs.shape[1]

        while trial_prefs < prefs_per_trial:
            prefs_remaining = prefs_per_trial - trial_prefs
            if modality == "state":
                new_features, new_probs = state_collect_step(
                    generator,
                    batch_timesteps,
                    reward,
                    temperature,
                    normalize_differences,
                )
            elif modality == "traj":
                new_features, new_probs = traj_collect_step(
                    generator=generator,
                    timesteps=batch_timesteps,
                    n_trajs=prefs_remaining,
                    reward=reward,
                    temperature=temperature,
                    normalize_differences=normalize_differences,
                    max_length=max_length,
                    rng=rng,
                )
            else:
                raise ValueError(f"Modality {modality} must be 'state' or 'traj'")
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
                prefs_per_batch *= 10
                if modality is "state":
                    batch_timesteps = max_state_batch_size(
                        n_states=prefs_per_batch,
                        n_parallel_envs=n_parallel_envs,
                        step_nbytes=generator.step_nbytes,
                    )
                elif modality is "traj":
                    batch_timesteps = max_traj_batch_size(
                        n_trajs=prefs_per_batch,
                        n_parallel_envs=n_parallel_envs,
                        step_nbytes=generator.step_nbytes,
                    )
                else:
                    raise ValueError(f"Modality {modality} must be 'state' or 'traj'")
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

    np.save(features_outpath, features)
    np.save(outdir / (outname + ".flip-probs.npy"), flip_probs)

    plot_flip_probs(flip_probs, outdir, outname, temperature)

    return features_outpath, start_trial


def setup_outdir(
    rootdir: Path,
    modality: Literal["state", "traj"],
    max_length: Optional[int],
    temperature: float,
    deduplicate: bool,
    normalize_differences: NORM_DIFF_MODES,
):
    schema = ["modality", "temp", "dedup", "norm"]
    if modality == "traj":
        schema.append("length")

    folders = HyperFolders(rootdir / "prefs", schema=schema)
    hyper_values: Dict[str, Any] = {
        "modality": modality,
        "temp": temperature,
        "dedup": "dedup" if deduplicate else "no-dedup",
        "norm": f"norm-{normalize_differences}",
    }
    if modality is "traj":
        hyper_values["length"] = (
            f"length-{max_length}" if max_length is not None else "no-max",
        )

    outdir = folders.add_experiment(hyper_values)
    return outdir


def max_batch_size(
    modality: Literal["state", "traj"],
    prefs_per_trial: int,
    n_parallel_envs: int,
    generator: Generator,
    prefs_per_batch: int,
) -> int:
    if modality is "state":
        return max_state_batch_size(
            n_states=prefs_per_batch,
            n_parallel_envs=n_parallel_envs,
            step_nbytes=generator.step_nbytes,
        )
    elif modality is "traj":
        return max_traj_batch_size(
            n_trajs=prefs_per_trial,
            n_parallel_envs=n_parallel_envs,
            step_nbytes=generator.step_nbytes,
        )
    else:
        raise ValueError(f"Modality {modality} must be 'state' or 'traj'")


def plot_flip_probs(
    flip_probs: np.ndarray, outdir: Path, outname: str, temperature: float
) -> None:
    plt.hist(flip_probs.flatten(), bins=100)
    plt.title(f"Histogram of noise flip probabilities (temp={temperature:0.5f})")
    plt.xlabel("Flip Probability")
    plt.xlim((0, 1))
    plt.savefig(outdir / (outname + ".flip-probs.png"))
    plt.close()


def load_data(
    outdir: Path, outname: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    logging.info(f"Loading data from {outdir}/{outname}")
    features_outpath = outdir / (outname + ".features.npy")
    flip_probs_outpath = outdir / (outname + ".flip-probs.npy")

    current_prefs: Optional[np.ndarray] = None
    current_flip_probs: Optional[np.ndarray] = None
    if features_outpath.exists() and flip_probs_outpath.exists():
        current_prefs = np.load(features_outpath)
        current_flip_probs = np.load(flip_probs_outpath)
        assert current_prefs is not None and current_flip_probs is not None

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
    normalize_differences: NORM_DIFF_MODES,
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
    normalize_differences: NORM_DIFF_MODES,
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


def calibrate_trajs(
    reward: np.ndarray,
    generator: Generator,
    target_flip_prob: float,
    n_trajs: int,
    rng: np.random.Generator,
    normalize_differences: NORM_DIFF_MODES,
    max_length: Optional[int],
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
                generator=generator,
                timesteps=batch_timesteps,
                n_trajs=n_trajs - n_diffs,
                reward=reward,
                temperature=temperature,
                normalize_differences=normalize_differences,
                max_length=max_length,
                rng=rng,
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
    normalize_differences: NORM_DIFF_MODES,
    max_length: Optional[int] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Tuple[np.ndarray, np.ndarray]:
    feature_batch: List[np.ndarray] = []
    probs_batch: List[np.ndarray] = []
    for traj_a, traj_b in zip(
        *generator.gen_traj_pairs(timesteps=timesteps, n_trajs=n_trajs)
    ):
        features_a = traj_a.features
        features_b = traj_b.features
        assert features_a is not None and features_b is not None

        if max_length is not None:
            start_a = rng.integers(low=0, high=max(1, features_a.shape[0] - max_length))
            start_b = rng.integers(low=0, high=max(1, features_b.shape[0] - max_length))
            features_a = features_a[start_a : start_a + max_length]
            features_b = features_b[start_b : start_b + max_length]

        feature_a: np.ndarray = features_a.sum(axis=0, keepdims=True)
        feature_b: np.ndarray = features_b.sum(axis=0, keepdims=True)
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

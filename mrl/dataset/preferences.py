from __future__ import annotations

import gc
import logging
import random
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import psutil  # type: ignore
import torch
from mrl.envs.feature_envs import FeatureEnv  # type: ignore
from mrl.envs.util import FEATURE_ENV_NAMES, get_root_env, make_env
from mrl.util import (
    get_policy,
    max_traj_batch_size,
    np_gather,
    np_remove,
    procgen_rollout_dataset,
    procgen_rollout_features,
    setup_logging,
)
from phasic_policy_gradient.ppg import PhasicValueModel


def gen_state_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    n_states: int,
    n_parallel_envs: int,
    outname: str,
    use_exit: bool = False,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_step_features: bool = False,
    normalize_differences: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    rootdir = Path(rootdir)
    outdir = rootdir / f"prefs/state/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=verbosity, outdir=outdir)

    outname = str(outname)
    reward = np.load(rootdir / "reward.npy")
    if env == "miner" and not use_exit:
        reward = np.delete(reward, 1)

    rng = np.random.default_rng()

    generator = Generator(
        kind=env,
        n_parallel_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=list(set([policy_path, None])),
        rng=rng,
    )

    batch_timesteps = max_state_batch_size(
        n_states=n_states,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    diff_batches: List[np.ndarray] = []
    flip_prob_batches: List[np.ndarray] = []
    n_diffs = 0

    outpath = outdir / (outname + ".npy")
    if overwrite:
        np_remove(outdir, outname)

    if outpath.exists():
        current_diffs = np.load(outpath)
        n_diffs += len(current_diffs)
        logging.info(f"Starting with {n_diffs} existing preferences.")
    else:
        current_diffs = None
        if not overwrite:
            logging.info(f"Did not find existing preferences at {outpath}")

    if n_diffs >= n_states:
        logging.warning(f"No new states needed for {outdir}")
        return

    while n_diffs < n_states:
        feature_a, feature_b = generator.gen_state_pairs(timesteps=batch_timesteps)
        feature_diffs = feature_a - feature_b
        if env == "miner" and not use_exit:
            feature_diffs = np.delete(feature_diffs, 1, axis=1)
        feature_diffs = feature_diffs[np.any(feature_diffs != 0, axis=1)]
        oriented_diffs, probs = orient_diffs(
            feature_diffs,
            temperature,
            reward,
            normalize_differences,
            generator.rng,
        )
        diff_batches.append(oriented_diffs)
        flip_prob_batches.append(probs)
        n_diffs += oriented_diffs.shape[0]

        logging.info(f"Collected {n_diffs} of {n_states} preferences.")

    out_diffs: np.ndarray = np.concatenate(diff_batches)
    if current_diffs is not None:
        out_diffs = np.concatenate((current_diffs, out_diffs))

    assert len(out_diffs.shape) == 2
    np.save(outdir / outname, out_diffs)

    flip_probs = np.concatenate(flip_prob_batches)
    np.save(outdir / (outname + ".flip-probs.npy"), flip_probs)

    plt.hist(flip_probs)
    plt.title(f"Histogram of noise flip probabilities (temp={temperature})")
    plt.xlabel("Flip Probability")
    plt.savefig(outdir / (outname + ".flip-probs.png"))
    plt.close()


def gen_traj_preferences(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    n_trajs: int,
    n_parallel_envs: int,
    outname: str,
    use_exit: bool = False,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_step_features: bool = False,
    normalize_differences: bool = False,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    rootdir = Path(rootdir)
    outdir = rootdir / f"prefs/traj/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    setup_logging(level=verbosity, outdir=outdir)

    outname = str(outname)
    reward = np.load(rootdir / "reward.npy")
    if env == "miner" and not use_exit:
        reward = np.delete(reward, 1)

    rng = np.random.default_rng()

    generator = Generator(
        kind=env,
        n_parallel_envs=n_parallel_envs,
        normalize_features=normalize_step_features,
        policy_paths=[Path(policy_path), None] if policy_path is not None else [None],
        rng=rng,
    )

    batch_timesteps = max_traj_batch_size(
        n_trajs=n_trajs,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    current_trajs = 0
    if overwrite:
        np_remove(outdir, outname)
        collection_batch = 0
    else:
        collection_batch = max_shard_index(outdir, outname)

        if collection_batch > 0:
            data = np_gather(outdir, outname)

            current_trajs += data.shape[0]
            del data
            gc.collect()
            logging.info(f"Starting with {current_trajs} preferences.")
        else:
            logging.info(f"No existing data found at {outdir}.")

    if current_trajs >= n_trajs:
        logging.warning(f"No new trajectories needed for {outdir}.")
        exit()

    while current_trajs < n_trajs:
        logging.info(
            f"Asking for {n_trajs - current_trajs} trajs or {batch_timesteps} timesteps"
        )

        diff_batch: List[np.ndarray] = []
        probs_batch: List[np.ndarray] = []
        for traj_a, traj_b in zip(
            *generator.gen_traj_pairs(
                timesteps=batch_timesteps, n_trajs=n_trajs - current_trajs
            )
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = (
                torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
            ).numpy()
            assert (
                env != "miner"
                or feature_diff.shape == (4,)
                or feature_diff.shape == (5,)
            )
            if env == "miner" and not use_exit:
                feature_diff = np.delete(feature_diff, 1)
            if np.linalg.norm(feature_diff) == 0:
                continue

            oriented_diff, probs = orient_diffs(
                feature_diff,
                temperature,
                reward,
                normalize_differences,
                generator.rng,
            )
            if oriented_diff.shape[0] == 0:
                continue
            assert (
                env != "miner"
                or oriented_diff.shape == (4,)
                or oriented_diff.shape == (5,)
            )
            diff_batch.append(oriented_diff)
            probs_batch.append(probs)

        diffs_file = outdir / f"{outname}.{collection_batch}.npy"
        logging.info(f"Writing current batch to {diffs_file}.")
        out = np.stack(diff_batch)
        assert len(out.shape) == 2, f"out shape={out.shape} not expected (-1, 4)"
        np.save(diffs_file, out)

        flip_probs = np.stack(probs_batch)
        probs_file = outdir / f"{outname}.{collection_batch}.flip-probs.npy"
        np.save(probs_file, flip_probs)

        plt.hist(flip_probs)
        plt.title(f"Histogram of noise flip probabilities (temp={temperature})")
        plt.xlabel("Flip Probability")
        plt.savefig(outdir / (outname + ".flip-probs.png"))
        plt.close()

        collection_batch += 1
        current_trajs += out.shape[0]
        del diff_batch
        gc.collect()


def relabel_preferences(
    rootdir: Path,
    in_name: str,
    temperature: float,
    in_modalities: Optional[List[str]] = None,
    out_modalities: Optional[List[str]] = None,
    replications: Optional[int] = None,
    seed: int = 0,
) -> None:
    rootdir = Path(rootdir)
    if in_modalities is None:
        in_modalities = ["state", "traj"]
    if out_modalities is None:
        out_modalities = ["state", "traj"]

    rng = np.random.default_rng(seed)

    if replications is None:
        reward = np.load(rootdir / "reward.npy")
        for in_modality, out_modality in zip(in_modalities, out_modalities):
            in_dir = rootdir / "prefs" / in_modality
            in_dir /= str(temperature)

            diffs = np_gather(in_dir, in_name)
            diffs, _ = orient_diffs(
                diffs, temperature, reward, normalize_differences=False, rng=rng
            )

            out_dir = rootdir / "prefs" / out_modality / str(temperature)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / (in_name + ".npy"), diffs)
    else:
        pass


def max_shard_index(outdir: Path, outname: str) -> int:
    max_index = 0
    for file in outdir.glob(f"{outname}.[0-9]*.npy"):
        match = re.search("([0-9]*).npy", str(file))
        if match is None:
            continue
        current_index = int(match[1])
        max_index = max(max_index, current_index)
    return max_index


def max_state_batch_size(n_states: int, n_parallel_envs: int, step_nbytes: int) -> int:
    gc.collect()
    free_memory = psutil.virtual_memory().available
    logging.info(f"Free memory={free_memory}")

    batch_timesteps = min(
        n_states // n_parallel_envs, int(free_memory / step_nbytes * 0.8)
    )
    logging.info(f"batch_timesteps={batch_timesteps}")

    return batch_timesteps


def orient_diffs(
    diffs: np.ndarray,
    temperature: float,
    reward: np.ndarray,
    normalize_differences: bool,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    single_diff = len(diffs.shape) == 1
    if normalize_differences:
        diffs = diffs / np.linalg.norm(diffs, axis=1, keepdims=True)

    strengths = diffs @ reward
    good_diffs = np.abs(strengths) > 1e-8
    diffs = diffs[good_diffs]
    strengths = strengths[good_diffs]

    if temperature == 0.0:
        opinions = np.sign(strengths)
        diffs = (diffs.T * opinions).T
        prob_first_better = np.maximum(opinions, 0)
    else:
        prob_first_better = 1.0 / (1.0 + np.exp(-strengths / temperature))
        second_better = rng.random(size=diffs.shape[0]) > prob_first_better
        diffs[second_better] *= -1

    if single_diff:
        diffs = diffs.reshape(-1)
    return diffs, prob_first_better


class Generator:
    def __init__(
        self,
        kind: FEATURE_ENV_NAMES,
        n_parallel_envs: int,
        normalize_features: bool,
        policy_paths: List[Optional[Path]],
        rng: np.random.Generator,
    ) -> None:
        self.env = make_env(
            kind=kind,
            reward=0,
            num=n_parallel_envs,
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
            tqdm=True,
        ).reshape(-1, self.root_env._reward_weights.shape[0])

        features_b = procgen_rollout_features(
            env=self.env,
            policy=policy_b,
            timesteps=timesteps,
            tqdm=True,
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
            tqdm=True,
        )
        data_b = procgen_rollout_dataset(
            env=self.env,
            policy=policy_b,
            timesteps=timesteps,
            n_trajs=n_trajs,
            flags=["feature", "first"],
            tqdm=True,
        )

        return data_a.trajs(), data_b.trajs()


if __name__ == "__main__":
    fire.Fire(
        {
            "state": gen_state_preferences,
            "traj": gen_traj_preferences,
            "relabel": relabel_preferences,
        },
    )

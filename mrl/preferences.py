from __future__ import annotations

import gc
import logging
import random
import re
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, cast

import fire  # type: ignore
import numpy as np
import psutil  # type: ignore
import torch
from gym3.extract_dict_ob import ExtractDictObWrapper  # type: ignore
from gym3.types import ValType  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from torch.distributions import Categorical

from mrl.envs import Miner
from mrl.util import (
    np_gather,
    procgen_rollout_dataset,
    procgen_rollout_features,
    setup_logging,
)


def gen_state_preferences(
    rootdir: Path,
    n_states: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    outdirs, rewards = make_replication_paths(
        Path(rootdir),
        modality="state",
        temperature=temperature,
        replications=replications,
    )

    rng = np.random.default_rng()

    generator = Generator(
        n_parallel_envs,
        normalize_features,
        policy_paths=list(set([policy_path, None])),
        rng=rng,
    )

    batch_timesteps = max_state_batch_size(
        n_states=n_states,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    for outdir, reward in zip(outdirs, rewards):
        diffs: List[np.ndarray] = []
        n_diffs = 0

        outpath = outdir / (outname + ".npy")
        if outpath.exists():
            current_diffs = np.load(outpath)
            n_diffs += len(current_diffs)
            logging.info(f"Starting with {n_diffs} existing preferences.")
        else:
            current_diffs = None
            logging.info(f"Did not find existing preferences at {outpath}")

        while n_diffs < n_states:
            feature_a, feature_b = generator.gen_state_pairs(timesteps=batch_timesteps)
            feature_diffs = feature_a - feature_b
            for feature_diff in feature_diffs:
                if np.all(feature_diff == 0):
                    continue
                feature_diff, opinion = orient_diff(
                    feature_diff, temperature, reward, generator.rng
                )
                if opinion != 0:
                    diffs.append(feature_diff.copy())
                    n_diffs += 1
            logging.info(f"Collected {n_diffs} of {n_states} preferences.")

        out_diffs = np.stack(diffs)
        if current_diffs is not None:
            out_diffs = np.concatenate((current_diffs, out_diffs))

        assert len(out_diffs.shape) == 2
        np.save(outdir / outname, out_diffs)


def gen_traj_preferences(
    rootdir: Path,
    n_trajs: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Optional[Path] = None,
    temperature: float = 0.0,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    setup_logging(level=verbosity)
    outdirs, rewards = make_replication_paths(
        Path(rootdir),
        modality="traj",
        temperature=temperature,
        replications=replications,
    )

    rng = np.random.default_rng()

    generator = Generator(
        n_parallel_envs,
        normalize_features,
        policy_paths=[Path(policy_path), None] if policy_path is not None else [None],
        rng=rng,
    )

    batch_timesteps = max_traj_batch_size(
        n_trajs=n_trajs,
        n_parallel_envs=n_parallel_envs,
        step_nbytes=generator.step_nbytes,
    )

    for outdir, reward in zip(outdirs, rewards):
        collection_batch = max_shard_index(outdir, outname)

        current_trajs = 0
        if collection_batch > 0:
            data = np_gather(outdir, outname)
            current_trajs += data.shape[0]
            del data
            gc.collect()
            logging.info(f"Starting with {current_trajs} preferences.")
        else:
            logging.info(f"No existing data found at {outdir}.")

        while current_trajs < n_trajs:
            logging.info(
                f"Asking for {n_trajs - current_trajs} trajs or {batch_timesteps} timesteps"
            )

            diffs: List[np.ndarray] = []
            for traj_a, traj_b in zip(
                *generator.gen_traj_pairs(
                    timesteps=batch_timesteps, n_trajs=n_trajs - current_trajs
                )
            ):
                assert traj_a.features is not None and traj_b.features is not None
                feature_diff = (
                    torch.sum(traj_a.features, dim=0)
                    - torch.sum(traj_b.features, dim=0)
                ).numpy()
                if np.linalg.norm(feature_diff) == 0:
                    continue
                feature_diff, opinion = orient_diff(
                    feature_diff, temperature, reward, generator.rng
                )
                if opinion != 0:
                    diffs.append(feature_diff.copy())

            diffs_file = outdir / f"{outname}.{collection_batch}.npy"
            logging.info(f"Writing current batch to {diffs_file}.")
            out = np.stack(diffs)
            assert len(out.shape) == 2
            np.save(diffs_file, out)

            collection_batch += 1
            current_trajs += out.shape[0]
            del diffs
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
            if temperature is not None:
                in_dir /= str(temperature)

            diffs = np_gather(in_dir, in_name)
            if temperature is None or temperature <= 0:
                opinions: np.ndarray = np.sign(diffs @ reward)
                diffs = (diffs.T * opinions).T
                diffs = diffs[opinions != 0]
                assert np.all(diffs @ reward > 0)
            else:
                out_diffs = []
                for diff in diffs:
                    diff, opinion = orient_diff(diff, temperature, reward, rng)
                    if opinion != 0:
                        out_diffs.append(diff)
                diffs = np.stack(out_diffs)

            out_dir = rootdir / "prefs" / out_modality
            if temperature is not None:
                out_dir /= str(temperature)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / (in_name + ".npy"), diffs)
    else:
        pass


def make_replication_paths(
    rootdir: Path, modality: str, temperature: float, replications: Optional[int]
) -> Tuple[List[Path], List[np.ndarray]]:
    if replications is not None:
        offset = 0 if (rootdir / "0").exists() else 1
        rootdirs = [
            rootdir / str(replication)
            for replication in range(offset, replications + offset)
        ]
    else:
        rootdirs = [rootdir]

    outdirs = [rootdir / f"prefs/{modality}/{temperature}" for rootdir in rootdirs]
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)

    rewards = [np.load(rootdir / "reward.npy") for rootdir in rootdirs]
    return outdirs, rewards


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


def max_traj_batch_size(n_trajs: int, n_parallel_envs: int, step_nbytes: int) -> int:
    gc.collect()
    free_memory = psutil.virtual_memory().available
    logging.info(f"Free memory: {free_memory}")

    # How many timesteps can we fit into the available memory?
    batch_timesteps = min(
        (n_trajs * 1000 * 2) // n_parallel_envs, int(free_memory / step_nbytes * 0.8)
    )
    logging.info(f"batch_timesteps={batch_timesteps}")
    return batch_timesteps


def orient_diff(
    diff: np.ndarray,
    temperature: float,
    reward: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int]:
    if temperature == 0.0:
        opinion = np.sign(reward @ diff)
        diff *= opinion
    else:
        first_better, diff = noisy_pref(
            diff,
            reward,
            temperature,
            rng,
        )
        opinion = 1 if first_better else -1
    return diff, opinion


def noisy_pref(
    diff: np.ndarray,
    reward: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> Tuple[bool, np.ndarray]:
    """Generates a noisy preference between feature_a and feature_b

    Args:
        feature_a (np.ndarray): Reward features
        feature_b (np.ndarray): Reward features
        reward (np.ndarray): Reward weights
        temperature (float): How noisy the preference is. Low temp means preference is more often the non-noisy preference, high temperature is closer to random.
        rng (np.random.Generator): Random number Generator

    Returns:
        Tuple[bool, torch.Tensor]: If the first feature vector is better than the second, and their signed difference
    """
    strength = reward @ diff
    p_correct = 1.0 / (1.0 + np.exp(-strength / temperature))
    a_better = rng.random() < p_correct
    if not a_better:
        diff *= -1
    return a_better, diff


def get_policy(
    path: Optional[Path], actype: ValType, num: Optional[int] = None
) -> PhasicValueModel:
    if path is not None:
        policy = cast(PhasicValueModel, torch.load(path))
        policy.to(device=policy.device)
        return policy
    elif num is not None:
        return RandomPolicy(actype, num=num)
    else:
        raise ValueError("Either path or num must be specified")


class Generator:
    def __init__(
        self,
        n_parallel_envs: int,
        normalize_features: bool,
        policy_paths: List[Optional[Path]],
        rng: np.random.Generator,
    ) -> None:
        env = Miner(
            reward_weights=np.zeros(5),
            num=n_parallel_envs,
            normalize_features=normalize_features,
        )
        self.env = ExtractDictObWrapper(env, "rgb")

        self.policies = [
            get_policy(path, actype=env.ac_space, num=n_parallel_envs)
            for path in policy_paths
        ]

        feature = procgen_rollout_features(
            env=env,
            policy=self.policies[0],
            timesteps=1,
        )
        self.step_nbytes = feature.nbytes
        logging.info(f"one timestep size={self.step_nbytes}")

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
        ).reshape(-1, 5)

        features_b = procgen_rollout_features(
            env=self.env,
            policy=policy_b,
            timesteps=timesteps,
            tqdm=True,
        ).reshape(-1, 5)

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


# TODO: Move to own file
class RandomPolicy(PhasicValueModel):
    def __init__(self, actype: ValType, num: int):
        self.actype = actype
        self.act_dist = Categorical(
            probs=torch.ones(actype.eltype.n) / np.prod(actype.eltype.n)
        )
        self.device = torch.device("cpu")
        self.num = num

    def act(self, ob, first, state_in):
        return self.act_dist.sample((self.num,)), None, None

    def initial_state(self, batchsize):
        return None


if __name__ == "__main__":
    fire.Fire(
        {
            "state": gen_state_preferences,
            "traj": gen_traj_preferences,
            "relabel": relabel_preferences,
        },
    )

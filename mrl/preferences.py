from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

import fire  # type: ignore
import numpy as np
import psutil  # type: ignore
import torch
from gym3.extract_dict_ob import ExtractDictObWrapper  # type: ignore
from gym3.types import ValType  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from torch.distributions import Categorical

from mrl.envs import Miner
from mrl.util import procgen_rollout_dataset, procgen_rollout_features, setup_logging


def gen_mixed_state_preferences(
    rootdir: Path,
    n_random_states: int,
    n_policy_states: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Path,
    temperature: float = 0.0,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    outdirs, rewards = make_replication_paths(
        Path(rootdir), modality="state", temperature=temperature, replications=replications
    )

    rng = np.random.default_rng()

    gen_policy = Generator(
        n_parallel_envs,
        normalize_features,
        policy_path_a=Path(policy_path),
        policy_path_b=None,
        rng=rng,
    )

    gen_random = Generator(
        n_parallel_envs,
        normalize_features,
        policy_path_a=None,
        policy_path_b=None,
        rng=rng,
    )

    batch_timesteps = max_state_batch_size(
        n_states=max(n_random_states, n_policy_states),
        n_parallel_envs=n_parallel_envs,
        step_nbytes=gen_policy.step_nbytes,
    )

    diffs: List[List[np.ndarray]] = [[] for _ in outdirs]

    while len(diffs) < n_random_states:
        feature_a, feature_b = gen_random.gen_state_pairs(timesteps=batch_timesteps)
        for i in range(len(feature_a)):
            feature_diff = feature_a[i] - feature_b[i]
            if np.linalg.norm(feature_diff) > 0:
                for reward_index, reward in enumerate(rewards):
                    feature_diff = orient_diff(feature_diff, temperature, reward, gen_random.rng)
                    diffs[reward_index].append(feature_diff.copy())

    n_random_states = len(diffs)

    while len(diffs) - n_random_states < n_policy_states:
        feature_a, feature_b = gen_policy.gen_state_pairs(timesteps=10_000)
        for i in range(len(feature_a)):
            if feature_a[i, 1] > 0 or feature_b[i, 1] > 0:
                feature_diff = feature_a[i] - feature_b[i]
                for reward_index, reward in enumerate(rewards):
                    feature_diff = orient_diff(feature_diff, temperature, reward, gen_policy.rng)
                    diffs[reward_index].append(feature_diff.copy())

    for reward_index, outdir in enumerate(outdirs):
        np.save(outdir / outname, np.stack(diffs[reward_index]))


def gen_mixed_traj_preferences(
    rootdir: Path,
    n_random_trajs: int,
    n_policy_trajs: int,
    n_parallel_envs: int,
    outname: str,
    policy_path: Path,
    temperature: float = 0.0,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    setup_logging(level=verbosity)
    outdirs, rewards = make_replication_paths(
        Path(rootdir), modality="traj", temperature=temperature, replications=replications
    )

    rng = np.random.default_rng()

    gen_policy = Generator(
        n_parallel_envs,
        normalize_features,
        policy_path_a=Path(policy_path),
        policy_path_b=None,
        rng=rng,
    )
    gen_random = Generator(
        n_parallel_envs,
        normalize_features,
        policy_path_a=None,
        policy_path_b=None,
        rng=rng,
    )

    batch_timesteps = max_traj_batch_size(
        n_trajs=n_random_trajs, n_parallel_envs=n_parallel_envs, step_nbytes=gen_random.step_nbytes
    )

    current_trajs = 0
    collection_batch = 0
    while current_trajs < n_random_trajs:
        collection_batch += 1

        logging.info(
            f"Asking for {n_random_trajs - current_trajs} trajs or {batch_timesteps} timesteps"
        )

        diffs: List[List[np.ndarray]] = [[] for _ in range(len(rewards))]
        for traj_a, traj_b in zip(
            *gen_random.gen_traj_pairs(
                timesteps=batch_timesteps, n_trajs=n_random_trajs - current_trajs
            )
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = (
                torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
            ).numpy()
            if np.linalg.norm(feature_diff) > 0:
                for reward_index, reward in enumerate(rewards):
                    feature_diff = orient_diff(feature_diff, temperature, reward, gen_random.rng)
                    diffs[reward_index].append(feature_diff.copy())

        for reward_index, outdir in enumerate(outdirs):
            diffs_file = outdir / f"{outname}.{collection_batch}.diffs.npy"
            logging.info(f"Writing current batch to {diffs_file}.")
            np.save(diffs_file, np.stack(diffs[reward_index]))
        current_trajs += len(diffs)
        del diffs
        gc.collect()

    collection_batch += 1
    diffs = []
    while len(diffs) < n_policy_trajs:
        for traj_a, traj_b in zip(
            *gen_policy.gen_traj_pairs(
                timesteps=batch_timesteps, n_trajs=n_policy_trajs - len(diffs)
            )
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = (
                torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
            ).numpy()
            if feature_diff[1] != 0:
                for reward_index, reward in enumerate(rewards):
                    feature_diff = orient_diff(feature_diff, temperature, reward, gen_random.rng)
                    diffs[reward_index].append(feature_diff.copy())

        for reward_index, outdir in enumerate(outdirs):
            diffs_file = outdir / f"{outname}.{collection_batch}.diffs.npy"
            logging.info(f"Writing current batch of complete runs to {diffs_file}.")
            np.save(diffs_file, np.stack(diffs[reward_index]))
        current_trajs += len(diffs)
        del diffs
        gc.collect()


def reuse_replications(
    diffs_path: Path,
    replications_rootdir: Path,
    replications: int,
    temperature: float,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(verbosity)
    diffs_path = Path(diffs_path)

    if "traj" in diffs_path.parts:
        modality = "traj"
    elif "state" in diffs_path.parts:
        modality = "state"
    else:
        raise ValueError(f"Unknown modality in {diffs_path}")

    outdirs, rewards = make_replication_paths(
        Path(replications_rootdir),
        modality=modality,
        temperature=temperature,
        replications=replications,
    )

    logging.debug(f"outdirs={outdirs}")

    # Don't overwrite the diffs from the origin folder
    try:
        diffs_index = outdirs.index(diffs_path.parent)
        del outdirs[diffs_index]
        del rewards[diffs_index]
    except ValueError:
        pass

    rng = np.random.default_rng(seed)

    diffs = np.load(diffs_path)
    for outdir, reward in zip(outdirs, rewards):
        oriented_diffs = np.empty_like(diffs)
        for i, diff in enumerate(diffs):
            oriented_diffs[i] = orient_diff(diff, temperature, reward, rng)
        np.save(outdir / diffs_path.name, oriented_diffs)


def make_replication_paths(
    rootdir: Path, modality: str, temperature: float, replications: Optional[int]
) -> Tuple[List[Path], List[np.ndarray]]:
    if replications is not None:
        offset = 0 if (rootdir / "0").exists() else 1
        rootdirs = [
            rootdir / str(replication) for replication in range(offset, replications + offset)
        ]
    else:
        rootdirs = [rootdir]

    outdirs = [rootdir / f"prefs/{modality}/{temperature}" for rootdir in rootdirs]
    for outdir in outdirs:
        outdir.mkdir(parents=True, exist_ok=True)

    rewards = [np.load(rootdir / "reward.npy") for rootdir in rootdirs]
    return outdirs, rewards


def max_state_batch_size(n_states: int, n_parallel_envs: int, step_nbytes: int) -> int:
    gc.collect()
    free_memory = psutil.virtual_memory().available
    logging.info(f"Free memory={free_memory}")

    batch_timesteps = min(n_states // n_parallel_envs, int(free_memory / step_nbytes * 0.8))
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
) -> np.ndarray:
    if temperature == 0.0:
        opinion = np.sign(reward @ diff)
        diff *= opinion
    else:
        _, diff = noisy_pref(
            diff,
            reward,
            temperature,
            rng,
        )
    return diff


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
        policy_path_a: Optional[Path],
        policy_path_b: Optional[Path],
        rng: np.random.Generator,
    ) -> None:
        env = Miner(
            reward_weights=np.zeros(
                4,
            ),
            num=n_parallel_envs,
            normalize_features=normalize_features,
        )
        self.env = ExtractDictObWrapper(env, "rgb")

        self.policy_a = get_policy(policy_path_a, actype=env.ac_space, num=n_parallel_envs)
        self.policy_b = get_policy(policy_path_b, actype=env.ac_space, num=n_parallel_envs)

        feature = procgen_rollout_features(
            env=env,
            policy=self.policy_a,
            timesteps=1,
        )
        self.step_nbytes = feature.nbytes
        logging.info(f"one timestep size={self.step_nbytes}")

        self.rng = rng

    def gen_state_pairs(self, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
        features_a = procgen_rollout_features(
            env=self.env,
            policy=self.policy_a,
            timesteps=timesteps,
            tqdm=True,
        ).reshape(-1, 5)

        features_b = procgen_rollout_features(
            env=self.env,
            policy=self.policy_b,
            timesteps=timesteps,
            tqdm=True,
        ).reshape(-1, 5)

        return features_a, features_b

    def gen_traj_pairs(self, timesteps: int, n_trajs: int):
        data_a = procgen_rollout_dataset(
            env=self.env,
            policy=self.policy_a,
            timesteps=timesteps,
            n_trajs=n_trajs,
            flags=["feature", "first"],
            tqdm=True,
        )
        data_b = procgen_rollout_dataset(
            env=self.env,
            policy=self.policy_b,
            timesteps=timesteps,
            n_trajs=n_trajs,
            flags=["feature", "first"],
            tqdm=True,
        )

        return data_a.trajs(), data_b.trajs()


class RandomPolicy(PhasicValueModel):
    def __init__(self, actype: ValType, num: int):
        self.actype = actype
        self.act_dist = Categorical(probs=torch.ones(actype.eltype.n) / np.prod(actype.eltype.n))
        self.device = torch.device("cpu")
        self.num = num

    def act(self, ob, first, state_in):
        return self.act_dist.sample((self.num,)), None, None

    def initial_state(self, batchsize):
        return None


if __name__ == "__main__":
    fire.Fire(
        {
            "state-mixed": gen_mixed_state_preferences,
            "traj-mixed": gen_mixed_traj_preferences,
            "reuse": reuse_replications,
        },
    )

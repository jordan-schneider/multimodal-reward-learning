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


def noisy_pref(
    feature_a: np.ndarray,
    feature_b: np.ndarray,
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
    diff = feature_a - feature_b
    strength = reward @ diff
    p_correct = 1.0 / (1.0 + np.exp(-strength / temperature))
    a_better = rng.random() < p_correct
    if not a_better:
        diff *= -1
    return a_better, diff


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
    if replications is not None:
        offset = 0 if (rootdir / "0").exists() else 1
        for batch_iter in range(offset, replications + offset):
            gen_mixed_state_preferences(
                rootdir=rootdir / str(batch_iter),
                n_random_states=n_random_states,
                n_policy_states=n_policy_states,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                policy_path=policy_path,
                temperature=temperature,
                normalize_features=normalize_features,
                verbosity=verbosity,
            )
        exit()

    outdir = rootdir / f"state/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    gen_policy = Generator(
        rootdir,
        n_parallel_envs,
        normalize_features,
        policy_path_a=Path(policy_path),
        policy_path_b=None,
        rng=rng,
    )

    gen_random = Generator(
        Path(rootdir),
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

    diffs: List[np.ndarray] = []

    while len(diffs) < n_random_states:
        feature_a, feature_b = gen_random.gen_state_pairs(timesteps=batch_timesteps)
        for i in range(len(feature_a)):
            feature_diff = orient_diff(
                feature_a[i], feature_b[i], temperature, gen_random.reward, gen_random.rng
            )
            if np.linalg.norm(feature_diff) > 0:
                diffs.append(feature_diff)

    n_random_states = len(diffs)

    while len(diffs) - n_random_states < n_policy_states:
        feature_a, feature_b = gen_policy.gen_state_pairs(timesteps=10_000)
        for i in range(len(feature_a)):
            if feature_a[i, 1] > 0 or feature_b[i, 1] > 0:
                feature_diff = orient_diff(
                    feature_a[i], feature_b[i], temperature, gen_policy.reward, gen_policy.rng
                )
                diffs.append(feature_diff)
                print(len(diffs))

    np.save(outdir / outname, np.stack(diffs))


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
    if replications is not None:
        offset = 0 if (rootdir / "0").exists() else 1
        for batch_iter in range(offset, replications + offset):
            gen_mixed_traj_preferences(
                rootdir=rootdir / str(batch_iter),
                n_random_trajs=n_random_trajs,
                n_policy_trajs=n_policy_trajs,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                policy_path=policy_path,
                temperature=temperature,
                normalize_features=normalize_features,
                verbosity=verbosity,
            )
        exit()

    outdir = rootdir / f"traj/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    gen_policy = Generator(
        Path(rootdir),
        n_parallel_envs,
        normalize_features,
        policy_path_a=Path(policy_path),
        policy_path_b=None,
        rng=rng,
    )
    gen_random = Generator(
        Path(rootdir),
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
    i = 0
    while current_trajs < n_random_trajs:
        i += 1

        logging.info(
            f"Asking for {n_random_trajs - current_trajs} trajs or {batch_timesteps} timesteps"
        )

        diffs: List[np.ndarray] = []
        for traj_a, traj_b in zip(
            *gen_random.gen_traj_pairs(
                timesteps=batch_timesteps, n_trajs=n_random_trajs - current_trajs
            )
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = orient_diff(
                feature_a=torch.sum(traj_a.features, dim=0).numpy(),
                feature_b=torch.sum(traj_b.features, dim=0).numpy(),
                temperature=temperature,
                reward=gen_random.reward,
                rng=gen_random.rng,
            )

            if np.linalg.norm(feature_diff) > 0:
                diffs.append(feature_diff)

        np.save(outdir / f"{outname}.{i}.npy", np.stack(diffs))
        current_trajs += len(diffs)
        del diffs
        gc.collect()

    i += 1
    diffs = []
    while len(diffs) < n_policy_trajs:
        for traj_a, traj_b in zip(
            *gen_random.gen_traj_pairs(timesteps=10_000, n_trajs=n_policy_trajs - len(diffs))
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = orient_diff(
                feature_a=torch.sum(traj_a.features, dim=0).numpy(),
                feature_b=torch.sum(traj_b.features, dim=0).numpy(),
                temperature=temperature,
                reward=gen_policy.reward,
                rng=gen_policy.rng,
            )

            if feature_diff[1] != 0:
                diffs.append(feature_diff)

        np.save(outdir / f"{outname}.{i}.npy", np.stack(diffs))


def gen_state_preferences(
    rootdir: Path,
    n_states: int,
    n_parallel_envs: int,
    outname: str,
    temperature: float = 0.0,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
    use_value: bool = False,
    value_path: Optional[Path] = None,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    rootdir = Path(rootdir)
    if replications is not None:
        offset = 0 if (rootdir / "0").exists() else 1
        for batch_iter in range(offset, replications + offset):
            if policy_path_a is not None or policy_path_b is not None or use_value:
                raise NotImplementedError(
                    "Specifying policies and value networks with replications not supported at this time."
                )
            gen_state_preferences(
                rootdir=rootdir / str(batch_iter),
                n_states=n_states,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                temperature=temperature,
                normalize_features=normalize_features,
                verbosity=verbosity,
            )
        exit()

    outdir = rootdir / f"state/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    gen = Generator(
        Path(rootdir),
        n_parallel_envs,
        normalize_features,
        Path(policy_path_a) if policy_path_a is not None else None,
        Path(policy_path_b) if policy_path_b is not None else None,
        rng=rng,
    )

    if use_value:
        raise NotImplementedError("Value networks not supported yet")
        if value_path is not None and not Path(value_path).exists():
            raise ValueError(
                f"--use-value specified, but --value-path {value_path} does not point to valid file."
            )

    batch_timesteps = max_state_batch_size(n_states, n_parallel_envs, gen.step_nbytes)

    # TODO: Instead of generating states from a policy, we could define a valid latent state space,
    # sample from that, and convert to features. Alternatively, we could define a valid feature
    # space and sample from that directly. Doing direct sampling with produce a very different
    # distribution. I'm not certain how important this is. We want the distribution of states
    # that the human will actually see. There are some environments where direct sampling is
    # possible, but the real world is not one of them, and so we might be doomed to using policy
    # based distributions.

    diff_count = 0
    batch_iter = 0
    while diff_count < n_states:
        features_a, features_b = gen.gen_state_pairs(batch_timesteps)
        diffs: List[np.ndarray] = []
        for i in range(len(features_a)):
            diff = orient_diff(features_a[i], features_b[i], temperature, gen.reward, gen.rng)
            if np.linalg.norm(diff) > 0:
                diffs.append(diff)

        logging.info(f"Saving {len(diffs)} state comparisons in batch {batch_iter}")
        np.save(outdir / f"{outname}.{batch_iter}.npy", np.stack(diffs))

        diff_count += len(diffs)
        batch_iter += 1

        del features_a
        del features_b
        gc.collect()


def gen_traj_preferences(
    rootdir: Path,
    n_trajs: int,
    n_parallel_envs: int,
    outname: str,
    temperature: float = 0.0,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
    keep_raw_states: bool = False,
    normalize_features: bool = False,
    replications: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    rootdir = Path(rootdir)
    if replications is not None:
        if policy_path_a is not None or policy_path_b is not None:
            raise NotImplementedError(
                "Replications flag not compatible with specifying a policy path at this time."
            )

        offset = 0 if (rootdir / "0").exists() else 1
        for i in range(offset, replications + offset):
            gen_traj_preferences(
                rootdir=rootdir / str(i),
                n_trajs=n_trajs,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                temperature=temperature,
                keep_raw_states=keep_raw_states,
                normalize_features=normalize_features,
                verbosity=verbosity,
            )
        exit()

    outdir = rootdir / f"traj/{temperature}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    gen = Generator(
        Path(rootdir),
        n_parallel_envs,
        normalize_features,
        Path(policy_path_a) if policy_path_a is not None else None,
        Path(policy_path_b) if policy_path_b is not None else None,
        rng=rng,
    )

    batch_timesteps = max_traj_batch_size(n_trajs, n_parallel_envs, gen.step_nbytes)

    current_trajs = 0
    i = 0
    while current_trajs < n_trajs:
        i += 1

        diffs: List[np.ndarray] = []
        for traj_a, traj_b in zip(
            *gen.gen_traj_pairs(timesteps=batch_timesteps, n_trajs=n_trajs - current_trajs)
        ):
            assert traj_a.features is not None and traj_b.features is not None
            feature_diff = orient_diff(
                feature_a=torch.sum(traj_a.features, dim=0).numpy(),
                feature_b=torch.sum(traj_b.features, dim=0).numpy(),
                temperature=temperature,
                reward=gen.reward,
                rng=gen.rng,
            )

            if np.linalg.norm(feature_diff) > 0:
                diffs.append(feature_diff)

        np.save(outdir / f"{outname}.{i}.npy", np.stack(diffs))
        current_trajs += len(diffs)
        del diffs
        gc.collect()


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
    feature_a: np.ndarray,
    feature_b: np.ndarray,
    temperature: float,
    reward: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if temperature == 0.0:
        feature_diff = feature_a - feature_b
        opinion = np.sign(reward @ feature_diff)
        feature_diff *= opinion
    else:
        _, feature_diff = noisy_pref(
            feature_a,
            feature_b,
            reward,
            temperature,
            rng,
        )
    return feature_diff


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
        rootdir: Path,
        n_parallel_envs: int,
        normalize_features: bool,
        policy_path_a: Optional[Path],
        policy_path_b: Optional[Path],
        rng: np.random.Generator,
    ) -> None:
        reward_path = rootdir / "reward.npy"
        if not reward_path.exists():
            raise ValueError(f"Reward does not exist at {reward_path}")
        self.reward = np.load(reward_path)
        logging.info(f"reward={self.reward}")

        env = Miner(
            reward_weights=self.reward, num=n_parallel_envs, normalize_features=normalize_features
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
            "traj": gen_traj_preferences,
            "state": gen_state_preferences,
            "state-mixed": gen_mixed_state_preferences,
            "traj-mixed": gen_mixed_traj_preferences,
        },
    )

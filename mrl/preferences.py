import gc
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union, cast

import fire  # type: ignore
import numpy as np
import psutil  # type: ignore
import torch
from gym3.extract_dict_ob import ExtractDictObWrapper  # type: ignore
from gym3.types import ValType  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from torch.distributions import Categorical

from mrl.envs import Miner
from mrl.offline_buffer import RlDataset
from mrl.util import procgen_rollout_dataset, procgen_rollout_features


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


def gen_state_preferences(
    path: Path,
    timesteps: int,
    n_parallel_envs: int,
    outname: str,
    temperature: Optional[float] = None,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
    use_value: bool = False,
    value_path: Optional[Path] = None,
    replications: Optional[int] = None,
) -> None:
    path = Path(path)
    if replications is not None:
        offset = 0 if (path / "0").exists() else 1
        for batch_iter in range(offset, replications + offset):
            if policy_path_a is not None or policy_path_b is not None or use_value:
                raise NotImplementedError(
                    "Specifying policies and value networks with replications not supported at this time."
                )
            gen_state_preferences(
                path=path / str(batch_iter),
                timesteps=timesteps,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                temperature=temperature,
            )
        exit()
    reward_path, outdir = path / "reward.npy", path / "prefs/state"
    if not reward_path.exists():
        raise ValueError(f"Reward does not exist at {reward_path}")
    reward = np.load(reward_path)
    outdir.mkdir(parents=True, exist_ok=True)

    if use_value:
        raise NotImplementedError("Value networks not supported yet")
        if value_path is not None and not Path(value_path).exists():
            raise ValueError(
                f"--use-value specified, but --value-path {value_path} does not point to valid file."
            )

    env = Miner(reward_weights=reward, num=n_parallel_envs)
    env = ExtractDictObWrapper(env, "rgb")

    policy_a = get_policy(policy_path_a, actype=env.ac_space, num=n_parallel_envs)
    policy_b = get_policy(policy_path_b, actype=env.ac_space, num=n_parallel_envs)

    rng = np.random.default_rng()

    feature = procgen_rollout_features(
        env=env,
        policy=policy_a,
        timesteps=1,
    )
    one_step_size = feature.nbytes
    print(f"one timestep size={one_step_size}")
    del feature

    gc.collect()
    free_memory = psutil.virtual_memory().available
    print(f"Free memory={free_memory}")

    batch_timesteps = min(timesteps // n_parallel_envs, int(free_memory / one_step_size * 0.8))
    print(f"batch_timesteps={batch_timesteps}")

    n_batches = max(1, timesteps // (n_parallel_envs * batch_timesteps))

    # TODO: Instead of generating states from a policy, we could define a valid latent state space,
    # sample from that, and convert to features. Alternatively, we could define a valid feature
    # space and sample from that directly. Doing direct sampling with produce a very different
    # distribution. I'm not certain how important this is. We want the distribution of states
    # that the human will actually see. There are some environments where direct sampling is
    # possible, but the real world is not one of them, and so we might be doomed to using policy
    # based distributions.

    for batch_iter in range(n_batches):
        features_a = procgen_rollout_features(
            env=env,
            policy=policy_a,
            timesteps=batch_timesteps,
            tqdm=True,
        ).reshape(-1, 5)

        features_b = procgen_rollout_features(
            env=env,
            policy=policy_b,
            timesteps=batch_timesteps,
            tqdm=True,
        ).reshape(-1, 5)

        diffs: List[np.ndarray] = []
        for i in range(len(features_a)):
            if temperature is None:
                diff = features_a[i] - features_b[i]
                opinion = np.sign(reward @ diff)

                if opinion == 0:
                    continue

                diff *= opinion
                diffs.append(diff)
            else:
                _, diff = noisy_pref(features_a[i], features_b[i], reward, temperature, rng)
                diffs.append(diff)

        np.save(outdir / f"{outname}.{batch_iter}.npy", np.stack(diffs))
        del features_a
        del features_b
        gc.collect()


def gen_traj_preferences(
    path: Path,
    n_trajs: int,
    n_parallel_envs: int,
    outname: str,
    temperature: Optional[float] = None,
    policy_path: Optional[Path] = None,
    keep_raw_states: bool = False,
    replications: Optional[int] = None,
) -> None:
    path = Path(path)
    if replications is not None:
        if policy_path is not None:
            raise NotImplementedError(
                "Replications flag not compatible with specifying a policy path at this time."
            )

        offset = 0 if (path / "0").exists() else 1
        for i in range(offset, replications + offset):
            gen_traj_preferences(
                path=path / str(i),
                n_trajs=n_trajs,
                n_parallel_envs=n_parallel_envs,
                outname=outname,
                temperature=temperature,
                keep_raw_states=keep_raw_states,
            )
        exit()
    reward_path = path / "reward.npy"
    if not reward_path.exists():
        raise ValueError(f"Reward does not exist at {reward_path}")
    reward = np.load(reward_path)
    outdir = path / "prefs/traj/"
    outdir.mkdir(parents=True, exist_ok=True)

    env = Miner(reward_weights=reward, num=n_parallel_envs)
    env = ExtractDictObWrapper(env, "rgb")

    policy = get_policy(policy_path, actype=env.ac_space, num=n_parallel_envs)

    rng = np.random.default_rng()

    datum = procgen_rollout_dataset(
        env=env,
        policy=policy,
        timesteps=1,
    )
    one_step_size = datum.get_bytes()
    print(f"one timestep size={one_step_size}")

    gc.collect()
    free_memory = psutil.virtual_memory().available
    print(f"Free memory: {free_memory}")

    # How many timesteps can we fit into the available memory?
    batch_timesteps = min(int(free_memory / one_step_size * 0.8), n_trajs * 1000 * 2)
    print(f"batch_timesteps={batch_timesteps}")

    current_trajs = 0
    i = 0
    while current_trajs < n_trajs:
        i += 1
        data = procgen_rollout_dataset(
            env=env,
            policy=policy,
            timesteps=batch_timesteps,
            n_trajs=2 * (n_trajs - current_trajs),
            flags=["feature", "first"],
            tqdm=True,
        )

        trajs = data.trajs()
        diffs: List[np.ndarray] = []
        for traj_a, traj_b in zip(trajs, trajs):
            assert traj_a.features is not None and traj_b.features is not None
            if temperature is None:
                feature_diff = (
                    torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
                ).numpy()
                opinion = np.sign(reward @ feature_diff)
                if opinion == 0:
                    continue  # If there is exactly no preference, then skip the entire pair

                feature_diff *= opinion
            else:
                _, feature_diff = noisy_pref(
                    torch.sum(traj_a.features, dim=0).numpy(),
                    torch.sum(traj_b.features, dim=0).numpy(),
                    reward,
                    temperature,
                    rng,
                )

            diffs.append(feature_diff)

        np.save(outdir / f"{outname}.{i}.npy", np.stack(diffs))
        current_trajs += len(diffs)
        del data
        del trajs
        del diffs
        gc.collect()


def get_policy(
    path: Optional[Path], actype: ValType, num: Optional[int] = None
) -> PhasicValueModel:
    if path is not None:
        return cast(PhasicValueModel, torch.load(path))
    elif num is not None:
        return RandomPolicy(actype, num=num)
    else:
        raise ValueError("Either path or num must be specified")


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
        {"traj": gen_traj_preferences, "state": gen_state_preferences},
    )

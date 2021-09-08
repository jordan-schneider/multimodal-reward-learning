import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, NamedTuple, Optional, Tuple, Union, cast

import fire  # type: ignore
import numpy as np
import torch
from gym3.extract_dict_ob import ExtractDictObWrapper  # type: ignore
from gym3.types import ValType  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from torch.distributions import Categorical

from mrl.envs import Miner
from mrl.offline_buffer import RlDataset
from mrl.util import procgen_rollout


@dataclass
class TrajPreferences:
    trajs: List[Tuple[RlDataset.TrajF, RlDataset.TrajF]]
    diffs: torch.Tensor


class SlimTrajF(NamedTuple):
    actions: torch.Tensor
    rewards: torch.Tensor
    features: torch.Tensor


@dataclass
class SlimTrajPreferences:
    trajs: List[Tuple[SlimTrajF, SlimTrajF]]
    diffs: torch.Tensor


@dataclass
class StatePreferences:
    reward_or_value: Literal["reward", "value"]
    states: Tuple[torch.Tensor, torch.Tensor]  # Raw obs
    state_features: Tuple[torch.Tensor, torch.Tensor]  # Features
    diffs: torch.Tensor


def noisy_pref(
    feature_a: torch.Tensor,
    feature_b: torch.Tensor,
    reward: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> Tuple[bool, torch.Tensor]:
    """Generates a noisy preference between feature_a and feature_b

    Args:
        feature_a (torch.Tensor): Reward features
        feature_b (torch.Tensor): Reward features
        reward (np.ndarray): Reward weights
        temperature (float): How noisy the preference is. Low temp means preference is more often the non-noisy preference, high temperature is closer to random.
        rng (np.random.Generator): Random number Generator

    Returns:
        Tuple[bool, torch.Tensor]: If the first feature vector is better than the second, and their signed difference
    """
    diff = feature_a - feature_b
    strength = reward @ diff.numpy()
    p_correct = 1.0 / (1.0 + np.exp(-strength / temperature))
    a_better = rng.random() < p_correct
    if not a_better:
        diff *= -1
    return a_better, diff


def gen_state_preferences(
    reward_path: Path,
    timesteps: int,
    n_parallel_envs: int,
    outdir: Path,
    outname: str,
    temperature: Optional[float] = None,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
    use_value: bool = False,
    value_path: Optional[Path] = None,
) -> None:
    reward_path, outdir = Path(reward_path), Path(outdir)
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

    # TODO: Instead of generating states from a policy, we could define a valid latent state space,
    # sample from that, and convert to features. Alternatively, we could define a valid feature
    # space and sample from that directly. Doing direct sampling with produce a very different
    # distribution. I'm not certain how important this is. We want the distribution of states
    # that the human will actually see. There are some environments where direct sampling is
    # possible, but the real world is not one of them, and so we might be doomed to using policy
    # based distributions.
    data_a = RlDataset.from_gym3(
        *procgen_rollout(
            env=env,
            policy=policy_a,
            timesteps=timesteps // n_parallel_envs,
            tqdm=True,
            return_features=True,
        )
    )
    data_b = RlDataset.from_gym3(
        *procgen_rollout(
            env=env,
            policy=policy_b,
            timesteps=timesteps // n_parallel_envs,
            tqdm=True,
            return_features=True,
        )
    )

    assert data_a.features is not None
    assert data_b.features is not None

    states = (data_a.states, data_b.states)
    state_features = (data_a.features, data_b.features)
    diffs: List[torch.Tensor] = []
    skips = torch.zeros((timesteps,), dtype=torch.bool)
    for i in range(timesteps):
        if temperature is None:
            diff = state_features[0][i] - state_features[1][i]
            opinion = np.sign(reward @ diff.numpy())

            if opinion == 0:
                skips[i] = True
                continue

            if opinion < 0:
                # Swap things so the preferred state is first
                states[0][i], states[1][i] = states[1][i], states[0][i]
                state_features[0][i], state_features[1][i] = (
                    state_features[1][i],
                    state_features[0][i],
                )

            diff *= opinion
            diffs.append(diff)
        else:
            a_better, diff = noisy_pref(
                state_features[0][i], state_features[1][i], reward, temperature, rng
            )
            diffs.append(diff)
            if not a_better:
                states[0][i], states[1][i] = states[1][i], states[0][i]
                state_features[0][i], state_features[1][i] = (
                    state_features[1][i],
                    state_features[0][i],
                )

    states = (states[0][torch.logical_not(skips)], states[1][torch.logical_not(skips)])
    state_features = (
        state_features[0][torch.logical_not(skips)],
        state_features[1][torch.logical_not(skips)],
    )

    out = StatePreferences(
        reward_or_value="value" if use_value else "reward",
        states=states,
        state_features=state_features,
        diffs=torch.stack(diffs),
    )
    pkl.dump(out, (outdir / f"{outname}.pkl").open("wb"))


def strip_states(path: str) -> None:
    prefs = cast(TrajPreferences, pkl.load(open(path, "rb")))
    slim_trajs = [
        (
            SlimTrajF(actions=t[0].actions, rewards=t[0].rewards, features=t[0].features),
            SlimTrajF(actions=t[1].actions, rewards=t[1].rewards, features=t[0].features),
        )
        for t in prefs.trajs
    ]
    slim_prefs = SlimTrajPreferences(trajs=slim_trajs, diffs=prefs.diffs)
    pkl.dump(slim_prefs, open(path + ".slim", "wb"))


def gen_traj_preferences(
    reward_path: Path,
    timesteps: int,
    n_parallel_envs: int,
    outdir: Path,
    outname: str,
    temperature: Optional[float] = None,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
    keep_raw_states: bool = False,
) -> None:
    reward_path, outdir = Path(reward_path), Path(outdir)
    if not reward_path.exists():
        raise ValueError(f"Reward does not exist at {reward_path}")
    reward = np.load(reward_path)
    outdir.mkdir(parents=True, exist_ok=True)

    env = Miner(reward_weights=reward, num=n_parallel_envs)
    env = ExtractDictObWrapper(env, "rgb")

    policy_a = get_policy(policy_path_a, actype=env.ac_space, num=n_parallel_envs)
    policy_b = get_policy(policy_path_b, actype=env.ac_space, num=n_parallel_envs)

    rng = np.random.default_rng()

    data_a = RlDataset.from_gym3(
        *procgen_rollout(
            env=env,
            policy=policy_a,
            timesteps=timesteps // n_parallel_envs,
            tqdm=True,
            return_features=True,
        )
    )
    data_b = RlDataset.from_gym3(
        *procgen_rollout(
            env=env,
            policy=policy_b,
            timesteps=timesteps // n_parallel_envs,
            tqdm=True,
            return_features=True,
        )
    )

    trajs: List[Tuple[Union[RlDataset.TrajF, SlimTrajF], Union[RlDataset.TrajF, SlimTrajF]]] = []
    diffs: List[torch.Tensor] = []
    for traj_a, traj_b in zip(
        data_a.trajs(include_feature=True), data_b.trajs(include_feature=True)
    ):
        if not keep_raw_states:
            traj_a = SlimTrajF(
                actions=traj_a.actions, rewards=traj_a.rewards, features=traj_a.features
            )
            traj_b = SlimTrajF(
                actions=traj_b.actions, rewards=traj_b.rewards, features=traj_b.features
            )
        if temperature is None:
            feature_diff = torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
            opinion = np.sign(reward @ feature_diff.numpy())
            if opinion == 0:
                continue  # If there is exactly no preference, then skip the entire pair

            feature_diff *= opinion
            a_better = opinion > 0
        else:
            a_better, feature_diff = noisy_pref(
                torch.sum(traj_a.features, dim=0),
                torch.sum(traj_b.features, dim=0),
                reward,
                temperature,
                rng,
            )

        diffs.append(feature_diff)
        trajs.append((traj_a, traj_b) if a_better > 0 else (traj_b, traj_a))

    out = TrajPreferences(trajs=trajs, diffs=torch.stack(diffs))
    pkl.dump(out, (outdir / f"{outname}.pkl").open("wb"))


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
        {"traj": gen_traj_preferences, "state": gen_state_preferences, "strip": strip_states},
    )

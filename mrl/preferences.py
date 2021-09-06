import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

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


@dataclass
class StatePreferences:
    reward_or_value: Literal["reward", "value"]
    states: Tuple[torch.Tensor, torch.Tensor]  # Raw obs
    state_features: Tuple[torch.Tensor, torch.Tensor]  # Features
    diffs: torch.Tensor


def gen_state_preferences(
    reward_path: Path,
    timesteps: int,
    n_parallel_envs: int,
    outdir: Path,
    outname: str,
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
        diff = state_features[0][i] - state_features[1][i]
        opinion = np.sign(reward @ diff.numpy())

        if opinion == 0:
            skips[i] = True
            continue

        if opinion < 0:
            # Swap things so the preferred state is first
            states[0][i], states[1][i] = states[1][i], states[0][i]
            state_features[0][i], state_features[1][i] = state_features[1][i], state_features[0][i]

        diff = diff * opinion
        diffs.append(diff)

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


def gen_traj_preferences(
    reward_path: Path,
    timesteps: int,
    n_parallel_envs: int,
    outdir: Path,
    outname: str,
    policy_path_a: Optional[Path] = None,
    policy_path_b: Optional[Path] = None,
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

    trajs: List[Tuple[RlDataset.TrajF, RlDataset.TrajF]] = []
    diffs: List[torch.Tensor] = []
    for traj_a, traj_b in zip(
        data_a.trajs(include_feature=True), data_b.trajs(include_feature=True)
    ):
        feature_diff = torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
        opinion = np.sign(reward @ feature_diff.numpy())
        if opinion == 0:
            continue  # If there is exactly no preference, then skip the entire pair

        feature_diff *= opinion
        diffs.append(feature_diff)
        trajs.append((traj_a, traj_b) if opinion > 0 else (traj_b, traj_a))

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
        {"traj": gen_traj_preferences, "state": gen_state_preferences},
    )

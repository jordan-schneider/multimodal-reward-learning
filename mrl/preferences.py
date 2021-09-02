import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

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


def gen_trajectories(
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
        trajs.append((traj_a, traj_b))
        feature_diff = torch.sum(traj_a.features, dim=0) - torch.sum(traj_b.features, dim=0)
        opinion = np.sign(reward @ feature_diff.numpy())
        if opinion != 0:
            feature_diff *= opinion
            diffs.append(feature_diff)

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
    fire.Fire({"traj": gen_trajectories})

import logging
from pathlib import Path
from typing import Literal, Optional, Type, Union, overload

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from gym3.env import Env  # type: ignore
from gym3.wrapper import Wrapper  # type: ignore
from mrl.envs import Maze, Miner
from mrl.envs.feature_envs import FeatureEnv
from mrl.envs.probe_envs import OneActionNoObsOneTimestepOneReward as Probe1
from mrl.envs.probe_envs import OneActionTwoObsOneTimestepDeterministicReward as Probe2
from procgen.env import ProcgenGym3Env

ENV_NAMES = Literal[
    "maze", "miner", "maze-native", "miner-native", "probe-1", "probe-2"
]
FEATURE_ENV_NAMES = Literal["maze", "miner"]


@overload
def make_env(name: FEATURE_ENV_NAMES, num: int, **kwargs) -> FeatureEnv:
    pass


@overload
def make_env(name: ENV_NAMES, num: int, **kwargs) -> ProcgenGym3Env:
    pass


def make_env(
    name: ENV_NAMES,
    num: int,
    reward: Optional[Union[float, np.ndarray]] = None,
    extract_rgb: bool = True,
    **kwargs
) -> ProcgenGym3Env:
    if name == "maze":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=2, fill_value=reward)
        env = Maze(reward, num, **kwargs)
    elif name == "miner":
        assert reward is not None
        if not isinstance(reward, np.ndarray):
            reward = np.full(shape=4, fill_value=reward)
        env = Miner(reward, num, **kwargs)
    elif name == "probe-1":
        env = Probe1(num=num, **kwargs)
    elif name == "probe-2":
        env = Probe2(num=num, **kwargs)
    else:
        env = ProcgenGym3Env(num=num, env_name=name)

    if extract_rgb:
        env = ExtractDictObWrapper(env, "rgb")
    return env


def get_root_env(env: Wrapper, max_layers: int = 100) -> Env:
    root_env = env
    layer = 0
    while isinstance(root_env, Wrapper) and layer < max_layers:
        root_env = root_env.env
        layer += 1
    if layer == max_layers:
        raise RuntimeError("Infinite loop looking for root_env")
    return root_env


def setup_env_folder(
    env_dir: Path, env: Type[FeatureEnv], n_reward_values: int, overwrite: bool = False
):
    env_dir = Path(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)

    rewards = env.make_reward_weights(n_reward_values)
    for i, reward in rewards:
        reward_dir = env_dir / str(i + 1)
        reward_dir.mkdir(parents=True, exist_ok=True)
        reward_path = reward_dir / "reward.npy"
        if overwrite or not reward_path.exists():
            np.save(reward_path, reward)

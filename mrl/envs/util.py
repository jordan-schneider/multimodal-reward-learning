import logging
from typing import Literal, Optional, Union, overload

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from gym3.env import Env
from gym3.wrapper import Wrapper
from mrl.envs import Maze, Miner
from mrl.envs.feature_envs import FeatureEnv  # type: ignore
from mrl.envs.probe_envs import OneActionNoObsOneTimestepOneReward as Probe1
from mrl.envs.probe_envs import OneActionTwoObsOneTimestepDeterministicReward as Probe2
from procgen.env import ProcgenGym3Env

ENV_NAMES = Literal[
    "maze", "miner", "maze-native", "miner-native", "probe-1", "probe-2"
]
FEATURE_ENV_NAMES = Literal["maze", "miner"]


@overload
def make_env(kind: FEATURE_ENV_NAMES, num: int, **kwargs) -> FeatureEnv:
    pass


@overload
def make_env(kind: ENV_NAMES, num: int, **kwargs) -> ProcgenGym3Env:
    pass


def make_env(
    kind: ENV_NAMES,
    num: int,
    reward: Optional[Union[float, np.ndarray]] = None,
    extract_rgb: bool = True,
    **kwargs
) -> ProcgenGym3Env:
    if kind == "maze":
        assert reward is not None
        if isinstance(reward, float):
            reward = np.full(shape=2, fill_value=reward)
        env = Maze(reward, num, **kwargs)
    elif kind == "miner":
        assert reward is not None
        if isinstance(reward, float):
            reward = np.full(shape=5, fill_value=reward)
        env = Miner(reward, num, **kwargs)
    elif kind == "probe-1":
        env = Probe1(num=num, **kwargs)
    elif kind == "probe-2":
        env = Probe2(num=num, **kwargs)
    else:
        env = ProcgenGym3Env(num=num, env_name=kind)

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

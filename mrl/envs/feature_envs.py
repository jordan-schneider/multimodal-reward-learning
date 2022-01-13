from abc import ABC
from typing import Generic, List, Tuple, TypeVar

import numpy as np
from procgen.env import ProcgenGym3Env


class StateInterface(ABC):
    grid: np.ndarray
    agent_pos: Tuple[int, int]


S = TypeVar("S", bound=StateInterface)


class FeatureEnv(ProcgenGym3Env, Generic[S]):

    N_FEATURES: int
    _reward_weights: np.ndarray
    features: np.ndarray

    def make_features(self) -> np.ndarray:
        raise NotImplementedError()

    def make_latent_states(self) -> List[S]:
        raise NotImplementedError()

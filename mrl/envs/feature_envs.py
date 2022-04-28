from abc import ABC
from typing import Generic, List, Sequence, Tuple, TypeVar

import numpy as np
from procgen.env import ProcgenGym3Env


class StateInterface(ABC):
    grid: np.ndarray
    agent_pos: Tuple[int, int]


S = TypeVar("S", bound=StateInterface)


class FeatureEnv(ProcgenGym3Env, Generic[S]):
    _reward_weights: np.ndarray
    features: np.ndarray

    def make_features(self) -> np.ndarray:
        raise NotImplementedError()

    def make_latent_states(self) -> List[S]:
        raise NotImplementedError()

    @property
    def n_features(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def make_reward_weights(
        values_per_dim: int = -1,
        feature_ranges: Sequence[Tuple[float, float]] = [],
    ) -> np.ndarray:
        raise NotImplementedError()

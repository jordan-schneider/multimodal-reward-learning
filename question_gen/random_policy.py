import gym3  # type: ignore
import numpy as np
from linear_procgen.util import get_root_env
from numpy.random import Generator


class RandomPolicy:
    def __init__(self, env: gym3.Env, rng: Generator) -> None:
        self.actions = env.ac_space.eltype.n
        self.num = env.num
        self.rng = rng

    def __call__(self, ob) -> np.ndarray:
        return self.rng.integers(low=0, high=self.actions - 1, size=(self.num,))


class RandomGridPolicy:
    def __init__(self, env: gym3.Env, rng: Generator) -> None:
        if not hasattr(get_root_env(env), "ACTION_DICT"):
            raise ValueError("env must have an ACTION_DICT")
        self.actions = np.array(list(get_root_env(env).ACTION_DICT.values()))
        self.num = env.num
        self.rng = rng

    def __call__(self, ob) -> np.ndarray:
        return self.actions[
            self.rng.integers(low=0, high=self.actions, size=(self.num,))
        ]

from typing import Dict, Tuple

import numpy as np
from gym3.types import DictType, Discrete, TensorType
from procgen import ProcgenGym3Env


class OneActionNoObsOneTimestepOneReward(ProcgenGym3Env):
    def __init__(self, num: int, **kwargs):
        self.num = num
        self.ob_space = DictType(rgb=TensorType(eltype=Discrete(256), shape=(64, 64, 3)))
        self.ac_space = TensorType(eltype=Discrete(1), shape=())

    def act(self, ac: np.ndarray) -> None:
        pass

    def observe(self) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        return (
            np.ones(shape=self.num),
            {"rgb": np.ones(shape=(self.num, 64, 64, 3))},
            np.ones(shape=self.num, dtype=bool),
        )

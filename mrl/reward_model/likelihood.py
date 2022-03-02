from typing import Protocol

import numpy as np


class Likelihood(Protocol):
    def __call__(
        self,
        reward: np.ndarray,
        diffs: np.ndarray,
        temperature: float = 1.0,
        approximate: bool = False,
    ) -> np.ndarray:
        ...

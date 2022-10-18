from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from linear_procgen.util import ENV_NAMES

DIFF_NORM_METHODS = Literal[
    "diff-length", "sum-length", "max-length", "log-diff-length", None
]


@dataclass
class EnvConfig:
    name: ENV_NAMES = "miner"
    n_envs: int = 100
    normalize_step: bool = False


class PreferenceNoise:
    pass


@dataclass
class FixedPreference(PreferenceNoise):
    temp: float = 0.001


@dataclass
class FlipProb(PreferenceNoise):
    name: str = "flip-prob"
    prob: float = 0.5
    calibration_prefs: int = 100
    init_state_temp: float = 1.0
    init_traj_temp: float = 1.0


@dataclass
class PreferenceConfig:
    prefs_per_trial: int = 1000
    normalize_differences: DIFF_NORM_METHODS = "sum-length"
    deduplicate: bool = True
    noise: PreferenceNoise = FlipProb()
    max_length: Optional[int] = None


class InferenceNoise:
    pass


@dataclass
class TrueInference(InferenceNoise):
    pass


@dataclass
class FixedInference(InferenceNoise):
    temp: float = 0.001


@dataclass
class GammaInference(InferenceNoise):
    k: float = 0.01
    theta: float = 0.01
    samples: int = 100


@dataclass
class InferenceConfig:
    noise: InferenceNoise = GammaInference()
    likelihood_fn: Literal["boltzmann", "hinge"] = "boltzmann"
    use_shift: bool = False
    save_all: bool = False


@dataclass
class ExperimentConfig:
    rootdir: str = "/home/joschnei/research/multimodal-reward-learning/data/miner/4/"
    ars_name: str = "ars.mixed.npy"
    n_trials: int = 1
    append: bool = False
    env: EnvConfig = EnvConfig()
    preference: PreferenceConfig = PreferenceConfig()
    inference: InferenceConfig = InferenceConfig()
    max_ram: str = "60G"
    seed: Optional[int] = None
    overwrite: bool = False
    verbosity: Literal["INFO", "DEBUG"] = "INFO"

    def validate(self) -> None:
        if self.overwrite and self.append:
            raise ValueError("Can only specify one of overwrite or append")

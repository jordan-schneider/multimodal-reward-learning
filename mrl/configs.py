from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml
from linear_procgen.util import ENV_NAMES

DIFF_NORM_METHODS = Literal[
    "diff-length", "sum-length", "max-length", "log-diff-length", None
]


class Config:
    def dump(self, outdir: Path) -> None:
        yaml.dump(self, stream=(outdir / "config.pkl").open("w"))

    def validate(self) -> None:
        raise NotImplementedError()


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
    noise: InferenceNoise = FixedInference()
    likelihood_fn: Literal["boltzmann", "hinge"] = "boltzmann"
    use_shift: bool = False
    save_all: bool = True
    reward_particles: int = 10
    short_traj_cutoff: Optional[int] = 100


@dataclass
class HumanExperimentConfig(Config):
    rootdir: str = "/nfs/data/joschnei/multimodal-reward-learning/data/miner/"
    git_dir: str = "/home/joschnei/multimodal-reward-learning"
    question_db_path: str = "/nfs/data/joschnei/experiment-server/experiments.db"
    inference: InferenceConfig = InferenceConfig()
    env: ENV_NAMES = "miner"
    # TODO: Dedup this with PreferenceConfig
    norm_mode: DIFF_NORM_METHODS = "sum-length"
    max_questions: Optional[int] = None

    n_shuffles: int = 2

    centroid_stats: bool = False
    mean_dispersion_stats: bool = True

    verbosity: Literal["INFO", "DEBUG"] = "INFO"
    seed: int = 23443394578

    def validate(self) -> None:
        pass


@dataclass
class EnvConfig:
    name: ENV_NAMES = "miner"
    # How many environments to run concurrently when generating questions, etc.
    n_envs: int = 10
    # Should the environment return features normalized between 0 and 1.
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
class SyntheticPreferenceConfig:
    prefs_per_trial: int = 1000
    normalize_differences: DIFF_NORM_METHODS = "sum-length"
    deduplicate: bool = True
    noise: PreferenceNoise = FlipProb()
    max_length: Optional[int] = None


@dataclass
class SimulationExperimentConfig(Config):
    rootdir: str = "/home/joschnei/multimodal-reward-learning/data/miner/4/"
    ars_name: str = "ars.mixed.npy"
    n_trials: int = 1
    append: bool = False
    env: EnvConfig = EnvConfig()
    preference: SyntheticPreferenceConfig = SyntheticPreferenceConfig()
    inference: InferenceConfig = InferenceConfig()
    max_ram: str = "60G"
    seed: Optional[int] = None
    overwrite: bool = False
    verbosity: Literal["INFO", "DEBUG"] = "INFO"

    def validate(self) -> None:
        if self.overwrite and self.append:
            raise ValueError("Can only specify one of overwrite or append")

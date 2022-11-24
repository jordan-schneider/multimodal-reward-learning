from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal, Optional

import tomli
import tomli_w
from linear_procgen.util import ENV_NAMES
from pydantic.dataclasses import dataclass

from mrl.strict_converter import StrictConverter

DIFF_NORM_METHODS = Literal[
    "diff-length", "sum-length", "max-length", "log-diff-length", None
]

# TODO: Add cattr deseralization that reads a type field where the nested dataclass type is ambiguous.


class Config:
    def load(self, path: Path):
        raise NotImplementedError()

    def dump(self, outdir: Path) -> None:
        tomli_w.dump(dataclasses.asdict(self), (outdir / "config.toml").open("wb"))

    def validate(self) -> None:
        raise NotImplementedError()


class InferenceNoise:
    pass


@dataclass
class TrueInference(InferenceNoise):
    pass


@dataclass
class FixedInference(InferenceNoise):
    temp: float


@dataclass
class GammaInference(InferenceNoise):
    k: float
    theta: float
    samples: int


@dataclass
class InferenceConfig:
    noise: InferenceNoise
    likelihood_fn: Literal["boltzmann", "hinge"]
    use_shift: bool
    save_all: bool
    reward_particles: int
    short_traj_cutoff: Optional[int]


@dataclass
class HumanExperimentConfig(Config):
    rootdir: str
    git_dir: str
    question_db_path: str
    inference: InferenceConfig
    env: ENV_NAMES
    norm_mode: DIFF_NORM_METHODS
    max_questions: Optional[int]

    n_shuffles: int

    centroid_stats: bool
    mean_dispersion_stats: bool

    verbosity: Literal["INFO", "DEBUG"]
    seed: int

    @staticmethod
    def load(path: Path) -> HumanExperimentConfig:
        config_dict = tomli.load(path.open("rb"))
        return StrictConverter().structure(config_dict, HumanExperimentConfig)

    def validate(self) -> None:
        pass


@dataclass
class EnvConfig:
    name: ENV_NAMES
    # How many environments to run concurrently when generating questions, etc.
    n_envs: int
    # Should the environment return features normalized between 0 and 1.
    normalize_step: bool


class PreferenceNoise:
    pass


@dataclass
class FixedPreference(PreferenceNoise):
    temp: float


@dataclass
class FlipProb(PreferenceNoise):
    name: str
    prob: float
    calibration_prefs: int
    init_state_temp: float
    init_traj_temp: float


@dataclass
class SyntheticPreferenceConfig:
    prefs_per_trial: int
    normalize_differences: DIFF_NORM_METHODS
    deduplicate: bool
    noise: PreferenceNoise
    max_length: Optional[int]


@dataclass
class SimulationExperimentConfig(Config):
    rootdir: str
    ars_name: str
    n_trials: int
    append: bool
    env: EnvConfig
    preference: SyntheticPreferenceConfig
    inference: InferenceConfig
    max_ram: str
    seed: Optional[int]
    overwrite: bool
    verbosity: Literal["INFO", "DEBUG"]

    @staticmethod
    def load(path: Path) -> SimulationExperimentConfig:
        config_dict = tomli.load(path.open("rb"))
        return StrictConverter().structure(config_dict, SimulationExperimentConfig)

    def validate(self) -> None:
        if self.overwrite and self.append:
            raise ValueError("Can only specify one of overwrite or append")

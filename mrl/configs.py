from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Type

import tomli
import tomli_w
from attrs import asdict, define
from linear_procgen.util import ENV_NAMES

from mrl.strict_converter import StrictConverter

DIFF_NORM_METHODS = Literal[
    "diff-length", "sum-length", "max-length", "log-diff-length", None
]


def remove_nones(d):
    if isinstance(d, dict):
        return {k: remove_nones(v) for k, v in d.items() if v is not None}
    return d


class ConfigConverter(StrictConverter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_structure_hook_func(
            lambda t: t == InferenceNoise, self._inference_noise_structure
        )

        # TODO: Add hooks for ambiguous simulation attrs classes.

    def _inference_noise_structure(
        self, val, type_: Type[InferenceNoise]
    ) -> InferenceNoise:
        if not isinstance(val, dict):
            raise ValueError(f"Expected dict, got {type_}")

        type_name = val.pop("type")
        if type_name == "true":
            return TrueInference()
        elif type_name == "fixed":
            return self.structure(val, FixedInference)
        elif type_name == "gamma":
            return self.structure(val, GammaInference)
        raise ValueError("Type field required to disambiguate InferenceNoise")


@define
class Config:
    def load(self, path: Path):
        raise NotImplementedError()

    def dump(self, outdir: Path) -> None:
        tomli_w.dump(remove_nones(asdict(self)), (outdir / "config.toml").open("wb"))

    def validate(self) -> None:
        raise NotImplementedError()


class InferenceNoise:
    pass


@define
class TrueInference(InferenceNoise):
    pass


@define
class FixedInference(InferenceNoise):
    temp: float


@define
class GammaInference(InferenceNoise):
    k: float
    theta: float
    samples: int


@define
class InferenceConfig:
    noise: InferenceNoise
    likelihood_fn: Literal["boltzmann", "hinge"]
    use_shift: bool
    save_all: bool
    reward_particles: int
    short_traj_cutoff: Optional[int]


@define
class HumanExperimentConfig(Config):
    rootdir: str
    git_dir: str
    question_db_path: str
    inference: InferenceConfig
    env: ENV_NAMES
    norm_mode: DIFF_NORM_METHODS

    n_shuffles: int

    centroid_stats: bool
    mean_dispersion_stats: bool

    verbosity: Literal["INFO", "DEBUG"]
    seed: int

    max_questions: Optional[int] = None

    @staticmethod
    def load(path: Path) -> HumanExperimentConfig:
        config_dict = tomli.load(path.open("rb"))
        return ConfigConverter().structure(config_dict, HumanExperimentConfig)

    def validate(self) -> None:
        pass


@define
class EnvConfig:
    name: ENV_NAMES
    # How many environments to run concurrently when generating questions, etc.
    n_envs: int
    # Should the environment return features normalized between 0 and 1.
    normalize_step: bool


class PreferenceNoise:
    pass


@define
class FixedPreference(PreferenceNoise):
    temp: float


@define
class FlipProb(PreferenceNoise):
    name: str
    prob: float
    calibration_prefs: int
    init_state_temp: float
    init_traj_temp: float


@define
class SyntheticPreferenceConfig:
    prefs_per_trial: int
    normalize_differences: DIFF_NORM_METHODS
    deduplicate: bool
    noise: PreferenceNoise
    max_length: Optional[int]


@define
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from mrl.envs.util import FEATURE_ENV_NAMES


@dataclass
class EnvConfig:
    name: FEATURE_ENV_NAMES = "miner"
    n_envs: int = 100
    normalize_step: bool = False


@dataclass
class PreferenceNoise:
    name: str


@dataclass
class FixedPreference(PreferenceNoise):
    name: str = "fixed"
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
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = "sum-length"
    deduplicate: bool = True
    noise: PreferenceNoise = MISSING


@dataclass
class InferenceNoise:
    name: str


@dataclass
class TrueInference(InferenceNoise):
    name: str = "gt"


@dataclass
class FixedInference(InferenceNoise):
    name: str = "fixed"
    temp: float = 0.001


@dataclass
class GammaInference(InferenceNoise):
    name: str = "gamma"
    k: float = 0.01
    theta: float = 0.01
    samples: int = 100


@dataclass
class InferenceConfig:
    noise: InferenceNoise = MISSING
    likelihood_fn: Literal["boltzmann", "hinge"] = "boltzmann"
    use_shift: bool = False


@dataclass
class ExperimentConfig:
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"preference/noise": "flip-prob"},
            {"inference/noise": "gamma"},
            {"override hydra/hydra_logging": "none"},
            {"override hydra/job_logging": "none"},
        ]
    )
    rootdir: str = (
        "/home/joschnei/multimodal-reward-learning/data/miner/near-original-reward/7"
    )
    ars_name: str = "ars.mixed.npy"
    n_trials: int = 1
    append_trials: bool = False
    env: EnvConfig = EnvConfig()
    preference: PreferenceConfig = PreferenceConfig()
    inference: InferenceConfig = InferenceConfig()
    max_ram: str = "100G"
    seed: Optional[int] = None
    overwrite: bool = False
    verbosity: Literal["INFO", "DEBUG"] = "INFO"
    hydra: Any = field(
        default_factory=lambda: {
            "output_subdir": None,
            "run": {
                "dir": ".",
            },
            "sweep": {
                "dir": ".",
            },
        }
    )


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="experiment", node=ExperimentConfig)
    cs.store(
        group="preference/noise",
        name="fixed",
        node=FixedPreference,
    )
    cs.store(
        group="preference/noise",
        name="flip-prob",
        node=FlipProb,
    )
    cs.store(
        group="inference/noise",
        name="fixed",
        node=FixedInference,
    )
    cs.store(group="inference/noise", name="gt", node=TrueInference)
    cs.store(
        group="inference/noise",
        name="gamma",
        node=GammaInference,
    )

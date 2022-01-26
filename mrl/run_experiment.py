import logging
from pathlib import Path
from typing import Literal, Optional

import fire  # type: ignore

from mrl.aligned_rewards.make_ars import main as make_ars
from mrl.dataset.preferences import (
    gen_preferences,
    gen_state_preferences,
    gen_traj_preferences,
)
from mrl.envs.util import FEATURE_ENV_NAMES
from mrl.inference.posterior import compare_modalities
from mrl.util import setup_logging


def main(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    prefs_per_trial: int = 1000,
    temperature: Optional[float] = None,
    flip_prob: Optional[float] = None,
    calibration_prefs: int = 100,
    init_state_temp: float = 1.0,
    init_traj_temp: float = 1.0,
    n_envs: int = 100,
    normalize_step: bool = False,
    normalize_differences: Literal["diff-length", "sum-length", None] = None,
    use_hinge: bool = False,
    use_shift: bool = False,
    n_trials: int = 1,
    max_ram: str = "100G",
    seed: int = 0,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    if temperature is None and flip_prob is None:
        raise ValueError("Must specify one of temperature or flip_prob")
    if temperature is not None and flip_prob is not None:
        raise ValueError("Cannot specify both temperature and flip_prob")

    rootdir = Path(rootdir)
    if temperature is not None:
        temperature = float(temperature)
    elif flip_prob is not None:
        flip_prob = float(flip_prob)
    inference_outdir = make_inference_outdir(rootdir, temperature, flip_prob)
    setup_logging(level=verbosity, outdir=inference_outdir)

    n_prefs = prefs_per_trial * n_trials

    data_outname = make_pref_outname(n_prefs, normalize_differences)

    if temperature is not None:
        state_path = gen_state_preferences(
            rootdir=rootdir,
            env=env,
            n_states=n_prefs,
            n_parallel_envs=n_envs,
            outname=data_outname,
            temperature=temperature,
            normalize_step_features=normalize_step,
            normalize_differences=normalize_differences,
            overwrite=overwrite,
            verbosity=verbosity,
        )
        logging.debug(f"state_path returned: {state_path}")
        traj_path = gen_traj_preferences(
            rootdir=rootdir,
            env=env,
            n_trajs=n_prefs,
            n_parallel_envs=n_envs,
            outname=data_outname,
            temperature=temperature,
            normalize_step_features=normalize_step,
            normalize_differences=normalize_differences,
            overwrite=overwrite,
            verbosity=verbosity,
        )
        logging.debug(f"traj_path returned: {traj_path}")
    else:
        assert flip_prob is not None
        flip_prob = float(flip_prob)
        state_path, traj_path = gen_preferences(
            rootdir=rootdir,
            env=env,
            outname=data_outname,
            n_prefs=n_prefs,
            n_calibration_prefs=calibration_prefs,
            n_envs=n_envs,
            flip_prob=flip_prob,
            init_state_temp=init_state_temp,
            init_traj_temp=init_traj_temp,
            normalize_step=normalize_step,
            normalize_differences=normalize_differences,
            overwrite=overwrite,
            verbosity=verbosity,
        )

    reward_path = rootdir / "reward.npy"
    ars_path = rootdir / "aligned_reward_set.npy"
    if not ars_path.exists():
        make_ars(
            reward_path=reward_path,
            env_name=env,
            outdir=rootdir,
            seed=seed,
            verbosity=verbosity,
        )

    state_temp = float(state_path.parent.name)
    traj_temp = float(traj_path.parent.name)

    compare_modalities(
        outdir=inference_outdir,
        data_rootdir=rootdir,
        state_temp=state_temp,
        traj_temp=traj_temp,
        state_name=state_path.name,
        traj_name=traj_path.name,
        max_comparisons=prefs_per_trial,
        norm_diffs=normalize_differences,
        use_hinge=use_hinge,
        use_shift=use_shift,
        n_trials=n_trials,
        max_ram=max_ram,
        seed=seed,
        verbosity=verbosity,
    )


def make_pref_outname(
    n_prefs: int, normalize_differences: Literal["diff-length", "sum-length", None]
) -> str:
    outname = str(n_prefs)
    if normalize_differences == "diff-length":
        outname += ".noise-norm-diff"
    elif normalize_differences == "sum-length":
        outname += ".noise-norm-sum"
    return outname


def make_inference_outdir(
    rootdir: Path, temperature: Optional[float], flip_prob: Optional[float]
) -> Path:
    outdir = rootdir / "compare"
    if temperature is not None:
        outdir /= str(temperature)
    elif flip_prob is not None:
        outdir /= f"flip-{flip_prob}"
    return outdir


if __name__ == "__main__":
    fire.Fire(main)

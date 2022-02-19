import logging
from pathlib import Path
from typing import Literal, Optional, Union

import fire  # type: ignore

from mrl.aligned_rewards.make_ars import main as make_ars
from mrl.dataset.preferences import (
    gen_preferences,
    gen_state_preferences,
    gen_traj_preferences,
)
from mrl.envs.util import FEATURE_ENV_NAMES
from mrl.folders import HyperFolders
from mrl.inference.posterior import compare_modalities
from mrl.util import setup_logging


def main(
    rootdir: Path,
    env: FEATURE_ENV_NAMES,
    prefs_per_trial: int = 1000,
    n_trials: int = 1,
    pref_temp: Optional[float] = None,
    flip_prob: Optional[float] = None,
    calibration_prefs: int = 100,
    init_state_temp: float = 1.0,
    init_traj_temp: float = 1.0,
    inference_temp: Union[float, Literal["gt"]] = 1.0,
    deduplicate: bool = False,
    n_envs: int = 100,
    normalize_step: bool = False,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    use_hinge: bool = False,
    use_shift: bool = False,
    max_ram: str = "100G",
    seed: int = 0,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    if pref_temp is None and flip_prob is None:
        raise ValueError("Must specify one of pref_temperature or flip_prob")
    if pref_temp is not None and flip_prob is not None:
        raise ValueError(
            f"Cannot specify both temperature {pref_temp} and flip_prob {flip_prob}"
        )

    rootdir = Path(rootdir)
    if pref_temp is not None:
        pref_temp = float(pref_temp)
    elif flip_prob is not None:
        flip_prob = float(flip_prob)
    inference_outdir = make_inference_outdir(
        rootdir=rootdir,
        data_temp=pref_temp,
        flip_prob=flip_prob,
        inference_fn="hinge" if use_hinge else "boltzmann",
        inference_temp=inference_temp,
        dedup=deduplicate,
        normalization=normalize_differences,
    )
    setup_logging(level=verbosity, outdir=inference_outdir)

    data_outname = make_pref_outname(prefs_per_trial * n_trials, normalize_differences)

    if pref_temp is not None:
        state_path = gen_state_preferences(
            rootdir=rootdir,
            env=env,
            prefs_per_trial=prefs_per_trial,
            n_trials=n_trials,
            n_parallel_envs=n_envs,
            outname=data_outname,
            temperature=pref_temp,
            deduplicate=deduplicate,
            normalize_step_features=normalize_step,
            normalize_differences=normalize_differences,
            overwrite=overwrite,
            verbosity=verbosity,
        )
        logging.debug(f"state_path returned: {state_path}")
        traj_path = gen_traj_preferences(
            rootdir=rootdir,
            env=env,
            prefs_per_trial=prefs_per_trial,
            n_trials=n_trials,
            n_parallel_envs=n_envs,
            outname=data_outname,
            temperature=pref_temp,
            deduplicate=deduplicate,
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
            prefs_per_trial=prefs_per_trial,
            n_trials=n_trials,
            n_calibration_prefs=calibration_prefs,
            n_envs=n_envs,
            flip_prob=flip_prob,
            init_state_temp=init_state_temp,
            init_traj_temp=init_traj_temp,
            deduplicate=deduplicate,
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

    state_temp = float(state_path.parts[-3])
    traj_temp = float(traj_path.parts[-3])

    compare_modalities(
        outdir=inference_outdir,
        data_rootdir=rootdir,
        state_temp=state_temp,
        traj_temp=traj_temp,
        state_name=state_path.name,
        traj_name=traj_path.name,
        max_comparisons=prefs_per_trial,
        deduplicate=deduplicate,
        norm_diffs=normalize_differences,
        use_hinge=use_hinge,
        use_shift=use_shift,
        inference_temp=inference_temp,
        n_trials=n_trials,
        max_ram=max_ram,
        seed=seed,
        verbosity=verbosity,
    )


def make_pref_outname(
    n_prefs: int,
    normalize_differences: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ],
) -> str:
    outname = str(n_prefs)
    if normalize_differences == "diff-length":
        outname += ".noise-norm-diff"
    elif normalize_differences == "sum-length":
        outname += ".noise-norm-sum"
    return outname


def make_inference_outdir(
    rootdir: Path,
    data_temp: Optional[float],
    flip_prob: Optional[float],
    inference_fn: Literal["boltzmann", "hinge"],
    inference_temp: Union[float, Literal["gt"]],
    dedup: bool,
    normalization: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ],
) -> Path:
    folders = HyperFolders(
        rootdir / "compare",
        schema=[
            "data-noise",
            "inference-fn",
            "inference-temp",
            "normalization",
            "dedup",
        ],
    )
    if data_temp is not None:
        data_noise = f"pref-{data_temp}"
    elif flip_prob is not None:
        data_noise = f"flip-{flip_prob}"

    if normalization is None:
        norm_str = "no-norm"
    elif not (
        normalization in ["diff-length", "sum-length", "max-length", "log-diff-length"]
    ):
        raise ValueError(f"Invalid normalization: {normalization}")
    else:
        norm_str = normalization

    dedup_str = "dedup" if dedup else "no-dedup"

    return folders.add_experiment(
        {
            "data-noise": data_noise,
            "inference-fn": inference_fn,
            "inference-temp": f"inference-{inference_temp}",
            "normalization": norm_str,
            "dedup": dedup_str,
        }
    )


if __name__ == "__main__":
    fire.Fire(main)

import logging
from pathlib import Path
from typing import Tuple, cast

import hydra
import numpy as np
from omegaconf import MISSING, OmegaConf

from mrl.aligned_rewards.aligned_reward_set import AlignedRewardSet
from mrl.aligned_rewards.make_ars import main as make_ars
from mrl.configs import ExperimentConfig, FixedPreference, FlipProb, register_configs
from mrl.dataset.preferences import gen_preferences, gen_preferences_flip_prob
from mrl.experiment_db.experiment import ExperimentDB
from mrl.inference.analysis import analysis
from mrl.inference.plots import plot_comparisons
from mrl.inference.posterior import compare_modalities
from mrl.inference.results import Results
from mrl.util import load, setup_logging


@hydra.main(config_path=None, config_name="experiment")
def main(config: ExperimentConfig):
    if config.preference.noise is MISSING:
        raise ValueError("Must specify preference noise model")
    if config.inference.noise is MISSING:
        raise ValueError("Must specify inferene noise model")
    if config.inference.noise.name not in ["fixed", "gt"]:
        raise NotImplementedError("Prior noise models not implemented")
    if config.overwrite and config.append:
        raise ValueError("Can only specify one of overwrite or append")
    rootdir = Path(config.rootdir)
    experiment_db = ExperimentDB(git_dir=Path())
    inference_outdir = experiment_db.add(rootdir / "inference", config)
    setup_logging(level=config.verbosity, outdir=inference_outdir, force=True)

    write_config(config, inference_outdir)

    rng = np.random.default_rng(seed=config.seed)

    ((state_path, state_start_trial), (traj_path, traj_start_trial)) = get_prefs(
        config, rootdir, inference_outdir, rng=rng
    )

    reward_path = rootdir / "reward.npy"
    ars_path = rootdir / config.ars_name
    if not ars_path.exists():
        make_ars(
            reward_path=reward_path,
            env_name=config.env.name,
            outdir=rootdir,
            outname=config.ars_name,
            seed=config.seed if config.seed is not None else 0,
            verbosity=config.verbosity,
        )

    results = Results(inference_outdir / "trials", load_contents=config.append)

    results = compare_modalities(
        outdir=inference_outdir,
        reward_path=rootdir / "reward.npy",
        state_path=state_path,
        traj_path=traj_path,
        results=results,
        max_comparisons=config.preference.prefs_per_trial,
        norm_diffs=config.preference.normalize_differences,
        use_hinge=config.inference.likelihood_fn == "hinge",
        use_shift=config.inference.use_shift,
        inference_temp=config.inference.noise,
        n_trials=config.n_trials,
        save_all=config.inference.save_all,
        state_start_trial=state_start_trial,
        traj_start_trial=traj_start_trial,
        verbosity=config.verbosity,
        rng=rng,
    )

    results.start("")
    true_reward = results.get("true_reward")
    results.close()

    analysis(
        results=results,
        aligned_reward_set=AlignedRewardSet(
            path=rootdir / config.ars_name, true_reward=true_reward
        ),
    )
    plot_comparisons(results=results, outdir=inference_outdir)


def get_prefs(
    config: ExperimentConfig,
    rootdir: Path,
    inference_outdir: Path,
    rng: np.random.Generator,
) -> Tuple[Tuple[Path, int], Tuple[Path, int]]:
    noise = config.preference.noise
    search_results_path = inference_outdir / "search_results.pkl"
    if noise.name == "fixed" or (
        noise.name == "flip-prob" and search_results_path.exists()
    ):
        if noise.name == "flip-prob":
            cast(FlipProb, noise)
            logging.info("Temperature search already done")
            search_results = load(search_results_path)
            state_temp = search_results["state"]
            traj_temp = search_results["traj"]
        else:
            noise = cast(FixedPreference, noise)
            state_temp = noise.temp
            traj_temp = noise.temp

        state_path, state_start_trial = gen_preferences(
            rootdir=rootdir,
            env=config.env.name,
            modality="state",
            prefs_per_trial=config.preference.prefs_per_trial,
            n_trials=config.n_trials,
            n_parallel_envs=config.env.n_envs,
            outname="prefs",
            temperature=state_temp,
            deduplicate=config.preference.deduplicate,
            normalize_step_features=config.env.normalize_step,
            normalize_differences=config.preference.normalize_differences,
            append=config.append,
            overwrite=config.overwrite,
            verbosity=config.verbosity,
            rng=rng,
        )
        logging.debug(f"state_path returned: {state_path}")
        traj_path, traj_start_trial = gen_preferences(
            rootdir=rootdir,
            env=config.env.name,
            modality="traj",
            prefs_per_trial=config.preference.prefs_per_trial,
            n_trials=config.n_trials,
            n_parallel_envs=config.env.n_envs,
            outname="prefs",
            max_length=config.preference.max_length,
            temperature=traj_temp,
            deduplicate=config.preference.deduplicate,
            normalize_step_features=config.env.normalize_step,
            normalize_differences=config.preference.normalize_differences,
            append=config.append,
            overwrite=config.overwrite,
            verbosity=config.verbosity,
            rng=rng,
        )
        logging.debug(f"traj_path returned: {traj_path}")
    elif noise.name == "flip-prob":
        noise = cast(FlipProb, noise)

        (state_path, state_start_trial), (
            traj_path,
            traj_start_trial,
        ) = gen_preferences_flip_prob(
            rootdir=rootdir,
            env=config.env.name,
            outname="prefs",
            prefs_per_trial=config.preference.prefs_per_trial,
            n_trials=config.n_trials,
            n_calibration_prefs=noise.calibration_prefs,
            n_envs=config.env.n_envs,
            flip_prob=noise.prob,
            init_state_temp=noise.init_state_temp,
            init_traj_temp=noise.init_traj_temp,
            deduplicate=config.preference.deduplicate,
            normalize_step=config.env.normalize_step,
            normalize_differences=config.preference.normalize_differences,
            max_length=config.preference.max_length,
            append=config.append,
            overwrite=config.overwrite,
            verbosity=config.verbosity,
            rng=rng,
        )
    return (state_path, state_start_trial), (traj_path, traj_start_trial)


def write_config(config: ExperimentConfig, inference_outdir: Path) -> None:
    config_yaml = OmegaConf.to_yaml(config)
    (inference_outdir / "config.yaml").open("w").write(config_yaml)


if __name__ == "__main__":
    register_configs()
    main()

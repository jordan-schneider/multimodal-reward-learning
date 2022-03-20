import logging
from pathlib import Path
from typing import Literal, cast

import hydra
from omegaconf import MISSING, OmegaConf

from mrl.aligned_rewards.aligned_reward_set import AlignedRewardSet
from mrl.aligned_rewards.make_ars import main as make_ars
from mrl.configs import (
    ExperimentConfig,
    FixedInference,
    FixedPreference,
    FlipProb,
    GammaInference,
    register_configs,
)
from mrl.dataset.preferences import (
    gen_preferences,
    gen_state_preferences,
    gen_traj_preferences,
)
from mrl.folders import HyperFolders
from mrl.inference.analysis import analysis
from mrl.inference.plots import plot_comparisons
from mrl.inference.posterior import compare_modalities
from mrl.inference.results import Results
from mrl.util import dump, load, setup_logging


@hydra.main(config_path=None, config_name="experiment")
def main(config: ExperimentConfig):
    if config.preference.noise is MISSING:
        raise ValueError("Must specify preference noise model")
    if config.inference.noise is MISSING:
        raise ValueError("Must specify inferene noise model")
    if config.inference.noise.name not in ["fixed", "gt"]:
        raise NotImplementedError("Prior noise models not implemented")

    rootdir = Path(config.rootdir)
    inference_outdir = make_inference_outdir(config)
    setup_logging(level=config.verbosity, outdir=inference_outdir, force=True)

    write_config(config, inference_outdir)

    data_outname = make_pref_outname(
        config.preference.prefs_per_trial * config.n_trials,
        config.preference.normalize_differences,
    )

    noise = config.preference.noise
    search_results_path = inference_outdir / "search_results.pkl"
    if noise.name == "fixed" or (
        noise.name == "flip-prob" and search_results_path.exists()
    ):
        if noise.name == "flip-prob":
            cast(FlipProb, noise)
            search_results = load(search_results_path)
            state_temp = get_temp_from_pref_path(search_results["state"])
            traj_temp = get_temp_from_pref_path(search_results["traj"])
        else:
            noise = cast(FixedPreference, noise)
            state_temp = noise.temp
            traj_temp = noise.temp

        state_path = gen_state_preferences(
            rootdir=rootdir,
            env=config.env.name,
            prefs_per_trial=config.preference.prefs_per_trial,
            n_trials=config.n_trials,
            n_parallel_envs=config.env.n_envs,
            outname=data_outname,
            temperature=state_temp,
            deduplicate=config.preference.deduplicate,
            normalize_step_features=config.env.normalize_step,
            normalize_differences=config.preference.normalize_differences,
            overwrite=config.overwrite,
            verbosity=config.verbosity,
        )
        logging.debug(f"state_path returned: {state_path}")
        traj_path = gen_traj_preferences(
            rootdir=rootdir,
            env=config.env.name,
            prefs_per_trial=config.preference.prefs_per_trial,
            n_trials=config.n_trials,
            n_parallel_envs=config.env.n_envs,
            outname=data_outname,
            temperature=traj_temp,
            deduplicate=config.preference.deduplicate,
            normalize_step_features=config.env.normalize_step,
            normalize_differences=config.preference.normalize_differences,
            overwrite=config.overwrite,
            verbosity=config.verbosity,
        )
        logging.debug(f"traj_path returned: {traj_path}")
    elif noise.name == "flip-prob":
        noise = cast(FlipProb, noise)

        state_path, traj_path = gen_preferences(
            rootdir=rootdir,
            env=config.env.name,
            outname=data_outname,
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
            overwrite=config.overwrite,
            verbosity=config.verbosity,
        )
        search_results = {
            "state": state_path,
            "traj": traj_path,
        }
        dump(search_results, inference_outdir / "search_results.pkl")

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

    state_temp = get_temp_from_pref_path(state_path)
    traj_temp = get_temp_from_pref_path(traj_path)

    results = Results(inference_outdir / "trials", load_contents=config.append_trials)

    results = compare_modalities(
        outdir=inference_outdir,
        data_rootdir=rootdir,
        state_temp=state_temp,
        traj_temp=traj_temp,
        state_name=state_path.name,
        traj_name=traj_path.name,
        results=results,
        max_comparisons=config.preference.prefs_per_trial,
        deduplicate=config.preference.deduplicate,
        norm_diffs=config.preference.normalize_differences,
        use_hinge=config.inference.likelihood_fn == "hinge",
        use_shift=config.inference.use_shift,
        inference_temp=config.inference.noise,
        n_trials=config.n_trials,
        max_ram=config.max_ram,
        seed=config.seed,
        verbosity=config.verbosity,
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


def write_config(config: ExperimentConfig, inference_outdir: Path) -> None:
    config_yaml = OmegaConf.to_yaml(config)
    (inference_outdir / "config.yaml").open("w").write(config_yaml)


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


def make_inference_outdir(config: ExperimentConfig) -> Path:
    folders = HyperFolders(
        Path(config.rootdir) / "compare",
        schema=[
            "data-noise",
            "inference-fn",
            "inference-temp",
            "ars",
            "normalization",
            "dedup",
        ],
    )
    preference_noise = config.preference.noise
    if preference_noise.name == "fixed":
        preference_noise = cast(FixedPreference, preference_noise)
        data_noise = f"pref-{preference_noise.temp}"
    elif preference_noise.name == "flip-prob":
        preference_noise = cast(FlipProb, preference_noise)
        data_noise = f"flip-{preference_noise.prob}"

    inference_noise = config.inference.noise
    if inference_noise.name == "gamma":
        inference_noise = cast(GammaInference, inference_noise)
        k = inference_noise.k
        theta = inference_noise.theta
        inference_temp_str = f"gamma({k}, {theta})"
    elif inference_noise.name == "fixed":
        inference_noise = cast(FixedInference, inference_noise)
        inference_temp_str = f"fixed-{inference_noise.temp}"
    elif inference_noise.name == "gt":
        inference_temp_str = "gt"

    ars_name = config.ars_name.strip(".npy").replace(".", "-")

    if (normalize := config.preference.normalize_differences) is None:
        norm_str = "no-norm"
    elif normalize not in [
        "diff-length",
        "sum-length",
        "max-length",
        "log-diff-length",
    ]:
        raise ValueError(
            f"Invalid normalization: {config.preference.normalize_differences}"
        )
    else:
        norm_str = normalize

    dedup_str = "dedup" if config.preference.deduplicate else "no-dedup"

    return folders.add_experiment(
        {
            "data-noise": data_noise,
            "inference-fn": config.inference.likelihood_fn,
            "inference-temp": f"inference-{inference_temp_str}",
            "ars": ars_name,
            "normalization": norm_str,
            "dedup": dedup_str,
        }
    )


def get_temp_from_pref_path(state_path):
    float(state_path.parts[-3])


if __name__ == "__main__":
    register_configs()
    main()

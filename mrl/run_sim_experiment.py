from pathlib import Path
from typing import Tuple

import numpy as np

from mrl.aligned_rewards.aligned_reward_set import AlignedRewardSet
from mrl.aligned_rewards.make_ars import main as make_ars
from mrl.configs import FixedPreference, FlipProb, SimulationExperimentConfig
from mrl.dataset.preferences import PreferenceGenerator
from mrl.experiment_db.experiment import ExperimentDB
from mrl.inference.analysis import analysis
from mrl.inference.plots import plot_comparisons
from mrl.inference.posterior import compare_modalities
from mrl.inference.results import Results
from mrl.util import setup_logging


def main() -> None:
    """Runs a reward inference experiment with simulated preferences according to a given ground truth reward. See ExperimentConfig for details."""
    config = SimulationExperimentConfig()
    config.validate()

    rootdir = Path(config.rootdir)
    experiment_db = ExperimentDB(git_dir=Path())
    inference_outdir = experiment_db.add(rootdir / "inference", config)
    setup_logging(level=config.verbosity, outdir=inference_outdir, force=True)
    config.dump(inference_outdir)

    rng = np.random.default_rng(seed=config.seed)

    ((state_path, state_start_trial), (traj_path, traj_start_trial)) = get_prefs(
        config, rng=rng
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
            diffs=np.load(rootdir / config.ars_name), true_reward=true_reward
        ),
    )
    plot_comparisons(results=results, outdir=inference_outdir)


def get_prefs(
    config: SimulationExperimentConfig,
    rng: np.random.Generator,
) -> Tuple[Tuple[Path, int], Tuple[Path, int]]:
    """Generate state and trajectory preferences according to configuration variables.

    Args:
        config (ExperimentConfig): Configuration for preference generation. See ExperimentConfig and PreferenceGenerator for details.
        rng (np.random.Generator): Numpy random number generator.

    Returns:
        Tuple[Tuple[Path, int], Tuple[Path, int]]: Paths to state and trajectory preference files, and the trial number at which to start inference.
    """
    noise = config.preference.noise

    generator = PreferenceGenerator(
        rootdir=Path(config.rootdir),
        env=config.env.name,
        outname="prefs",
        rng=rng,
        prefs_per_trial=config.preference.prefs_per_trial,
        n_trials=config.n_trials,
        n_envs=config.env.n_envs,
        deduplicate=config.preference.deduplicate,
        normalize_step=config.env.normalize_step,
        normalize_differences=config.preference.normalize_differences,
        append=config.append,
        overwrite=config.overwrite,
        verbosity=config.verbosity,
    )

    if isinstance(noise, FixedPreference):
        state_temp = noise.temp
        traj_temp = noise.temp
        state_path, state_start_trial = generator.gen_preferences(
            modality="state", temperature=state_temp
        )
        traj_path, traj_start_trial = generator.gen_preferences(
            modality="traj",
            temperature=traj_temp,
        )
    elif isinstance(noise, FlipProb):
        (state_path, state_start_trial), (
            traj_path,
            traj_start_trial,
        ) = generator.gen_preferences_flip_prob(
            n_calibration_prefs=noise.calibration_prefs,
            flip_prob=noise.prob,
            init_state_temp=noise.init_state_temp,
            init_traj_temp=noise.init_traj_temp,
        )
    return (state_path, state_start_trial), (traj_path, traj_start_trial)


if __name__ == "__main__":
    main()

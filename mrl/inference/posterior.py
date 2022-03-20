import logging
from math import sqrt
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Tuple, cast

import bitmath  # type: ignore
import fire  # type: ignore
import numpy as np
from mrl.configs import FixedInference, InferenceNoise, TrueInference
from mrl.inference.results import Results
from mrl.reward_model.boltzmann import boltzmann_likelihood
from mrl.reward_model.hinge import hinge_likelihood
from mrl.reward_model.likelihood import Likelihood
from mrl.reward_model.logspace import cum_likelihoods
from mrl.util import normalize_diffs, normalize_vecs, np_gather, setup_logging


def compare_modalities(
    outdir: Path,
    data_rootdir: Path,
    state_temp: float,
    traj_temp: float,
    state_name: str,
    traj_name: str,
    results: Results,
    n_samples: int = 100_000,
    max_comparisons: int = 1000,
    deduplicate: bool = False,
    norm_diffs: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
    use_hinge: bool = False,
    use_shift: bool = False,
    inference_temp: InferenceNoise = TrueInference(),
    n_trials: int = 1,
    plot_individual: bool = False,
    save_all: bool = False,
    state_start_trial: int = 0,
    traj_start_trial: int = 0,
    max_ram: str = "100G",
    seed: Optional[int] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> Results:
    try:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        setup_logging(outdir=outdir, level=verbosity)

        logging.info(
            f"""outdir={outdir},
data_rootdir={data_rootdir},
{state_temp=},
{traj_temp=},
{state_name=},
{traj_name=},
n_samples={n_samples},
max_comparisons={max_comparisons},
{deduplicate=},
norm_diffs={norm_diffs},
use_hinge={use_hinge},
use_shift={use_shift},
{inference_temp=},
n_trials={n_trials},
{plot_individual=},
save_all={save_all},
{max_ram=},
seed={seed},
verbosity={verbosity}"""
        )

        rng = np.random.default_rng(seed=seed)

        paths = collect_paths(
            data_rootdir,
            state_temp=state_temp,
            traj_temp=traj_temp,
            state_name=state_name,
            traj_name=traj_name,
            dedup=deduplicate,
            state_start_trial=state_start_trial,
            traj_start_trial=traj_start_trial,
            norm=norm_diffs,
        )

        results.start("")

        if inference_temp.name == "gt":
            temps = {
                "state": state_temp,
                "traj": traj_temp,
                "joint": sqrt(state_temp * traj_temp),
            }
        elif inference_temp.name == "fixed":
            inference_temp = cast(FixedInference, inference_temp)
            temps = {
                "state": inference_temp.temp,
                "traj": inference_temp.temp,
                "joint": inference_temp.temp,
            }
        elif inference_temp.name == "gamma":
            raise NotImplementedError("Gamma inference not implemented")
        else:
            raise ValueError(f"Inference temp {inference_temp} not recognized")

        results.update("temp", temps)

        reward_path = data_rootdir / "reward.npy"
        true_reward = load_ground_truth(reward_path=reward_path)
        logging.debug(f"{true_reward=} from {reward_path=}")

        max_ram_nbytes = int(bitmath.parse_string_unsafe(max_ram).bytes)
        trial_batches = load_comparison_data(
            paths=paths,
            max_comparisons=max_comparisons,
            n_trials=n_trials,
            ram_free=max_ram_nbytes,
            rng=rng,
        )

        logging.info(f"Generating {n_samples} reward samples on the sphere")
        reward_samples = cover_sphere(
            n_samples=n_samples, ndims=get_reward_ndims(outdir), rng=rng
        )

        if true_reward is not None:
            results.update("true_reward", true_reward)
            logging.info("Adding ground truth reward as reward sample")
            # We're doing this so we can evalute the likelihood of the gt reward later.
            reward_samples = np.concatenate((reward_samples, [true_reward]), axis=0)

        results.update("reward_sample", reward_samples)

        last_old_trial = get_last_trial(results.experiment_names())

        for trial_offset, features in enumerate(trial_batches):
            trial = last_old_trial + trial_offset + 1
            logging.info(f"Starting trial-{trial}")
            results.start(f"trial-{trial}")

            diffs = {
                key: normalize_diffs(feature, mode=norm_diffs)
                for key, feature in features.items()
            }

            try:
                # TODO: Write function that processes one modality at a time to make multiple
                # temperature values easy to handle.
                results = make_likelihoods(
                    reward_samples=reward_samples,
                    diffs=diffs,
                    reward_likelihood=hinge_likelihood
                    if use_hinge
                    else boltzmann_likelihood,
                    use_shift=use_shift,
                    results=results,
                    temps=temps,
                    save_all=save_all,
                )
            except AssertionError as e:
                logging.exception(e)
            results.close()

        logging.info("Finished all trials")
    except Exception as e:
        logging.exception(e)

    return results


def get_last_trial(experiment_names: List[str]) -> int:
    trial = -1
    for name in experiment_names:
        if name.startswith("trial-"):
            trial = max(trial, int(name[6:]))
    return trial


def load_ground_truth(
    reward_path: Path,
) -> Optional[np.ndarray]:
    true_reward = None
    if reward_path.exists():
        logging.info(f"Loading ground truth reward from {reward_path}")
        true_reward = np.load(reward_path)
    else:
        logging.warning("No ground truth reward provided")

    return true_reward


def collect_paths(
    rootdir: Path,
    state_temp: float,
    traj_temp: float,
    state_name: str,
    traj_name: str,
    state_start_trial: int,
    traj_start_trial: int,
    dedup: bool,
    norm: str,
) -> Dict[str, Tuple[Path, int]]:
    dedup_str = "dedup" if dedup else "no-dedup"
    paths: Dict[str, Tuple[Path, int]] = {}
    state_root = (
        rootdir
        / f"prefs/state/{state_temp}/{dedup_str}/norm-{norm}/{state_name}.features"
    )
    state_path = Path(str(state_root) + ".npy")

    traj_root = (
        rootdir / f"prefs/traj/{traj_temp}/{dedup_str}/norm-{norm}/{traj_name}.features"
    )
    traj_path = Path(str(traj_root) + ".npy")

    if state_path.exists():
        logging.info(f"Loading states from {state_path}")
        paths["state"] = (state_root, state_start_trial)
    else:
        logging.warning(f"No file found at {state_path}")

    if traj_path.exists():
        logging.info(f"Loading trajectories from {traj_path}")
        paths["traj"] = (traj_root, traj_start_trial)
    else:
        logging.warning(f"No file found at {traj_path}")

    assert len(paths) > 0

    return paths


def get_reward_ndims(path: Path) -> int:
    if "miner" in str(path):
        return 4
    elif "maze" in str(path):
        return 2
    else:
        raise ValueError(f"path {path} must include either miner or maze folder")


def load_comparison_data(
    paths: Dict[str, Tuple[Path, int]],
    max_comparisons: int,
    n_trials: int,
    ram_free: int,
    rng: np.random.Generator,
) -> Generator[Dict[str, np.ndarray], None, None]:
    all_data: Dict[str, np.ndarray] = {}
    logging.debug(f"paths={paths}")
    for key, (path, start_trial) in paths.items():
        # TODO: Remove np_gather calls here, nothing is sharded anymore.
        all_trials = np_gather(
            indir=path.parent,
            name=path.name,
            max_nbytes=ram_free // len(paths),
        )
        if n_trials > all_trials.shape[0]:
            raise RuntimeError(
                f"{key} only has data for {all_trials.shape[0]} trials, but need {n_trials}"
            )
        elif max_comparisons > all_trials.shape[1]:
            raise RuntimeError(
                f"{key} only has {all_trials.shape[1]} prefs per trial, need {max_comparisons}"
            )
        all_data[key] = all_trials[start_trial:]

    logging.debug(f"data={all_data}")

    for trial in range(n_trials):
        trial_data: Dict[str, np.ndarray] = {
            key: value[trial] for key, value in all_data.items()
        }

        trial_data["joint"] = np.concatenate(
            [
                features[: max_comparisons // len(trial_data.keys())]
                for features in trial_data.values()
            ]
        )
        for key, features in trial_data.items():
            perm = rng.permutation(features.shape[0])
            trial_data[key] = features[perm]

        yield trial_data


def make_likelihoods(
    reward_samples: np.ndarray,
    diffs: Dict[str, np.ndarray],
    reward_likelihood: Likelihood,
    use_shift: bool,
    results: Results,
    temps: Optional[Dict[str, float]] = None,
    save_all: bool = False,
) -> Results:
    if temps is None:
        temps = {key: 1.0 for key in diffs}

    logging.info("Computing p(reward|diff)")
    log_likelihoods = {
        key: reward_likelihood(
            reward=reward_samples, diffs=diff, temperature=temps[key]
        )
        for key, diff in diffs.items()
    }

    if save_all:
        results.update("log_likelihoods", log_likelihoods)

    logging.info("Normalizing and totaling likelihoods")
    likelihoods = {
        key: cum_likelihoods(log_likelihoods=l, shift=use_shift)
        for key, l in log_likelihoods.items()
    }

    logging.info("Saving total likelihoods")
    save_likelihoods(likelihoods, results, save_all)

    return results


def save_likelihoods(
    likelihoods: Dict[str, np.ndarray], results: Results, save_all: bool = False
) -> Results:
    results.update("likelihood", likelihoods, save=save_all)
    if not save_all:
        # Subsample 1% of the sampled rewards and reduce precision to save disk space by default.
        likelihood_samples = {
            k: v[::100].astype(np.float32) for k, v in likelihoods.items()
        }
        results.update("likelihood_sample", likelihood_samples)
    return results


def cover_sphere(n_samples: int, ndims: int, rng: np.random.Generator) -> np.ndarray:
    samples = rng.multivariate_normal(
        mean=np.zeros(ndims), cov=np.eye(ndims), size=(n_samples)
    )
    samples = normalize_vecs(samples)
    return samples


def infogain_sort(rewards: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    """Sort difference vector greedily by expected information gain.

    Args:
        rewards (np.ndarray): Reward samples used to evaluate infogain.
        diffs (np.ndarray): Reward feature difference vectors.

    Returns:
        np.ndarray: A copy of diffs in the order of max expected infogain
    """
    current_diffs = [diffs[0]]
    for i in range(1, len(diffs)):
        likelihoods = None
    exit()
    # TODO: Finish


if __name__ == "__main__":
    fire.Fire(
        {
            "compare": compare_modalities,
        }
    )

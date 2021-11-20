import logging
import pickle as pkl
from math import ceil
from pathlib import Path
from typing import Any, Dict, Final, Literal, Optional, Sequence, Union

import fire  # type: ignore
import joypy  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from tqdm import trange  # type: ignore

from mrl.reward_model.boltzmann import boltzmann_likelihood
from mrl.reward_model.hinge import hinge_likelihood
from mrl.reward_model.logspace import cum_likelihoods
from mrl.sphere import find_centroid
from mrl.util import dump, np_gather, setup_logging


def one_modality_analysis(
    in_path: Path,
    outdir: Path,
    n_samples: int = 1000,
    max_preferences: int = 1000,
    frac_complete: float = 0.1,
    temperature: float = 1.0,
    approximate: bool = False,
    norm_diffs: bool = False,
    use_hinge: bool = False,
    reward_path: Optional[Path] = None,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    in_path, outdir = Path(in_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diffs = np_gather(in_path.parent, in_path.name, max_preferences, frac_complete)
    true_reward = np.load(reward_path) if reward_path is not None else None

    analysis(
        outdir=outdir,
        diffs=diffs,
        n_reward_samples=n_samples,
        rng=np.random.default_rng(seed=seed),
        temperature=temperature,
        approximate=approximate,
        norm_diffs=norm_diffs,
        use_hinge=use_hinge,
        true_reward=true_reward,
    )


def analysis_joint(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    action_path: Optional[Path] = None,
    reward_path: Optional[Path] = None,
    n_reward_samples: int = 100_000,
    data_per_modality: int = 1000,
    frac_complete: float = 0.1,
    temperature: float = 1.0,
    approximate: bool = False,
    norm_diffs: bool = False,
    use_hinge: bool = False,
    seed: int = 0,
    verbosity: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=seed)
    setup_logging(level=verbosity)

    outname = ""
    paths = []
    if traj_path is not None:
        paths.append(Path(traj_path))
        outname += "traj."
    if state_path is not None:
        paths.append(Path(state_path))
        outname += "state."
    if action_path is not None:
        paths.append(Path(action_path))
        outname += "action."
    outname += "dispersion.png"

    diffs = np.concatenate(
        [
            np_gather(path.parent, path.name, n=data_per_modality, frac_complete=frac_complete)
            for path in paths
        ]
    )
    rng.shuffle(diffs)

    true_reward = np.load(reward_path) if reward_path is not None else None

    analysis(
        outdir,
        diffs,
        n_reward_samples,
        rng,
        temperature=temperature,
        approximate=approximate,
        norm_diffs=norm_diffs,
        use_hinge=use_hinge,
        true_reward=true_reward,
    )


def compare_modalities(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    reward_path: Optional[Path] = None,
    n_samples: int = 100_000,
    max_comparisons: int = 1000,
    frac_complete: float = 0.1,
    norm_diffs: bool = False,
    use_hinge: bool = False,
    use_shift: bool = False,
    n_trials: int = 1,
    plot_individual: bool = False,
    save_all: bool = False,
    seed: int = 0,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir=outdir, level=verbosity)

    logging.info(
        f"outdir={outdir}, traj_path={traj_path}, state_path={state_path}, reward_path={reward_path}, n_samples={n_samples}, max_comparisons={max_comparisons}, frac_complete={frac_complete}, norm_diffs={norm_diffs}, use_hinge={use_hinge}, use_shift={use_shift}, n_trials={n_trials}, save_all={save_all}, seed={seed}, verbosity={verbosity}"
    )

    rng = np.random.default_rng(seed=seed)

    paths: Dict[str, Path] = {}
    if traj_path is not None:
        logging.info(f"Loading trajectories from {traj_path}")
        paths["traj"] = Path(traj_path)
    if state_path is not None:
        logging.info(f"Loading states from {state_path}")
        paths["state"] = Path(state_path)

    if reward_path is not None:
        logging.info(f"Loading ground truth reward from {reward_path}")
        true_reward = np.load(reward_path)

    logging.info(f"Generating {n_samples} reward samples on the sphere")
    NDIMS: Final = 5
    reward_samples = cover_sphere(n_samples, NDIMS, rng)

    if reward_path is not None:
        logging.info("Adding ground truth reward as reward sample")
        reward_samples = np.concatenate((reward_samples, [true_reward]), axis=0)

    np.save(outdir / "reward_samples.npy", reward_samples)

    results = Results(outdir)
    for trial in range(n_trials):
        logging.info(f"Starting trial-{trial}")
        results.start(f"trial-{trial}")

        diffs = load_comparison_diffs(paths, max_comparisons, frac_complete, rng)

        if norm_diffs:
            logging.info("Normalizing difference vectors")
            diffs = {key: normalize(diff) for key, diff in diffs.items()}

        shuffled_diffs = {}
        for key, diff in diffs.items():
            shuffled_diffs[key] = diff[rng.permutation(diffs["joint"].shape[0])]

        try:
            results = comparison_analysis(
                reward_samples=reward_samples,
                diffs=shuffled_diffs,
                use_hinge=use_hinge,
                use_shift=use_shift,
                results=results,
                true_reward=true_reward if reward_path is not None else None,
                save_all=save_all,
            )
            if plot_individual:
                plot_comparison(results, use_gt=reward_path is not None)
        except AssertionError as e:
            logging.exception(e)
        results.close()

    logging.info("Finished all trials, plotting aggregate results")

    dispersion_gt = results.getall("dispersion_gt")
    sns.relplot(
        data=dispersion_gt, x="time", y="dispersion_gt", hue="modality", kind="line"
    ).savefig(outdir / "dispersion_gt.png")
    plt.close()

    likelihoods_gt = results.getall_gt_likelihood()
    sns.relplot(
        data=likelihoods_gt, x="time", y="likelihood_gt", hue="modality", kind="line"
    ).savefig(outdir / "likelihood_gt.png")
    plt.close()

    entropies = results.getall("entropies")
    sns.relplot(data=entropies, x="time", y="entropies", hue="modality", kind="line").savefig(
        outdir / "entropy.png"
    )


def load_comparison_diffs(
    paths: Dict[str, Path],
    max_comparisons: int,
    frac_complete: float,
    rng: np.random.Generator,
):
    diffs = {
        key: np_gather(
            in_path.parent,
            in_path.name,
            n=max_comparisons,
            rng=rng,
            frac_complete=frac_complete,
        )
        for key, in_path in paths.items()
    }
    diffs["joint"] = np.concatenate(
        [diff[: max_comparisons // len(paths)] for diff in diffs.values()]
    )
    rng.shuffle(diffs["joint"])
    return diffs


class Results:
    current_experiment: Optional[str]

    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.experiments: Dict[str, Dict[str, Any]] = {}

    def start(self, experiment_name: str):
        self.experiments[experiment_name] = self.experiments.get(experiment_name, {})
        self.current_experiment = experiment_name
        (self.outdir / self.current_experiment).mkdir(exist_ok=True)

    def update(self, name: str, value: Any) -> None:
        assert self.current_experiment is not None, "No current experiment"
        self.experiments[self.current_experiment][name] = value
        dump(value, self.outdir / self.current_experiment / name)

    def get(self, name: str) -> Any:
        assert self.current_experiment is not None, "No current experiment"
        return self.experiments[self.current_experiment].get(name)

    def getall(self, name: str) -> pd.DataFrame:
        if name in ["centroids", "mean_rewards"]:
            raise NotImplementedError("Support for converting reward data implemented.")

        out = pd.DataFrame(columns=["trial", "time", name])
        for exp_name, exp in self.experiments.items():
            value = exp[name]
            if isinstance(value, np.ndarray):
                df = self.__make_df(name, value)
                df["trial"] = exp_name
                out = out.append(df.copy())
            elif isinstance(value, dict):
                for modality, v in value.items():
                    df = self.__make_df([name], v)
                    df["trial"] = exp_name
                    df["modality"] = modality
                    out = out.append(df.copy())

        out = out.reset_index(drop=True)
        return out

    def getall_gt_likelihood(self) -> pd.DataFrame:
        out = pd.DataFrame(columns=["trial", "time", "likelihood_gt"])
        for exp_name, exp in self.experiments.items():
            likelihoods = exp["likelihoods"]
            if isinstance(likelihoods, np.ndarray):
                df = self.__make_df(["likelihood_gt"], likelihoods[-1])
                df["trial"] = exp_name
                out = out.append(df.copy())
            elif isinstance(likelihoods, dict):
                for modality, v in likelihoods.items():
                    df = self.__make_df(["likelihood_gt"], v[-1])
                    df["trial"] = exp_name
                    df["modality"] = modality
                    out = out.append(df.copy())
        out = out.reset_index(drop=True)
        return out

    @staticmethod
    def __make_df(columns: Sequence[str], value: np.ndarray) -> pd.DataFrame:
        assert value.ndim == 1, "Only 1D arrays supported"
        df = pd.DataFrame(value, columns=columns)
        df.index.name = "time"
        df = df.reset_index()
        return df

    def close(self):
        self.current_experiment = None


def comparison_analysis(
    reward_samples: np.ndarray,
    diffs: Dict[str, np.ndarray],
    use_hinge: bool,
    use_shift: bool,
    results: Results,
    true_reward: Optional[np.ndarray] = None,
    save_all: bool = False,
) -> Results:
    logging.debug(diffs["joint"])
    reward_likelihood = hinge_likelihood if use_hinge else boltzmann_likelihood

    log_likelihoods = {
        key: reward_likelihood(reward=reward_samples, diffs=diff) for key, diff in diffs.items()
    }

    if save_all:
        results.update("log_likelihoods", log_likelihoods)

    likelihoods = {
        key: cum_likelihoods(log_likelihoods=l, shift=use_shift)
        for key, l in log_likelihoods.items()
    }

    if not save_all:
        # Subsample 1% of the sampled rewards and reduce precision to save disk space by default.
        samples = {k: v[::100].astype(np.float32) for k, v in likelihoods.items()}
        results.update("likelihoods", samples)
    else:
        results.update("likelihoods", likelihoods)

    entropies = {key: entropy(l) for key, l in likelihoods.items()}
    results.update("entropies", entropies)

    counts = {key: np.sum(l > 0.0, axis=0) for key, l in likelihoods.items()}
    results.update("counts", counts)

    centroid_per_modality = {}
    dispersion_centroid_per_modality = {}
    mean_rewards = {}
    proj_mean_rewards = {}
    for key, l in likelihoods.items():
        centroids = []
        dispersions = []
        mean_rewards[key] = find_means(rewards=reward_samples, likelihoods=l)
        proj_mean_rewards[key] = normalize(mean_rewards[key])
        for t in trange(l.shape[1]):
            centroid, dist = find_centroid(
                points=reward_samples,
                weights=l[:, t],
                max_iter=10,
                init=proj_mean_rewards[key][t],
            )
            assert np.allclose(
                np.linalg.norm(centroid), 1.0
            ), f"centroid={centroid} has norm={np.linalg.norm(centroid)} far from 1."
            centroids.append(centroid)
            dispersions.append(dist)

        centroid_per_modality[key] = np.stack(centroids)
        assert np.allclose(np.linalg.norm(centroid_per_modality[key], axis=1), 1.0)
        dispersion_centroid_per_modality[key] = np.array(dispersions)

    results.update("centroid_per_modality", centroid_per_modality)
    results.update("mean_rewards", mean_rewards)
    results.update("proj_mean_rewards", proj_mean_rewards)
    results.update("dispersions_centroid", dispersion_centroid_per_modality)

    dispersion_mean = {
        key: mean_geodesic_dispersion(
            reward_samples=reward_samples,
            likelihoods=likelihoods[key],
            target_rewards=proj_mean_rewards[key],
            expect_monotonic=False,
        )
        for key in likelihoods.keys()
    }
    results.update("dispersion_mean", dispersion_mean)

    if true_reward is not None:
        true_reward_index = np.where(np.all(reward_samples == true_reward, axis=1))[0][0]
        assert true_reward_index == len(list(likelihoods.values())[0]) - 1

        n_comparisons = list(diffs.values())[0].shape[0]

        true_reward_copies = np.tile(true_reward, (n_comparisons, 1))
        dispersions_gt = {
            key: mean_geodesic_dispersion(
                reward_samples=reward_samples,
                likelihoods=l,
                target_rewards=true_reward_copies,
            )
            for key, l in likelihoods.items()
        }
        results.update("dispersion_gt", dispersions_gt)

    return results


def plot_comparison(results: Results, use_gt: bool = False) -> None:
    assert results.current_experiment is not None, "No current experiment"
    outdir = results.outdir / results.current_experiment
    if likelihoods := results.get("likelihoods"):
        plot_liklihoods(likelihoods, outdir)

        if use_gt:
            plot_gt_likelihood(likelihoods, outdir)

    if entropies := results.get("entropies"):
        plot_entropies(entropies, outdir)
    if counts := results.get("counts"):
        plot_counts(counts, outdir)

    if dispersion_mean := results.get("dispersion_mean"):
        plot_dispersions(dispersion_mean, outdir, outname="dispersion_mean")

    if centroid_per_modality := results.get("centroid_per_modality"):
        plot_rewards(rewards=centroid_per_modality, outdir=outdir, outname="centroids")

    if mean_rewards := results.get("mean_rewards"):
        plot_rewards(rewards=mean_rewards, outdir=outdir, outname="mean_rewards")

    if dispersion_centroid_per_modality := results.get("dispersions_centroid"):
        plot_dispersions(dispersion_centroid_per_modality, outdir, outname="dispersion_centroid")

    if dispersions_gt := results.get("dispersion_gt"):
        assert use_gt
        plot_dispersions(dispersions_gt, outdir, outname="dispersion_gt")


def analysis(
    outdir: Path,
    diffs: np.ndarray,
    n_reward_samples: int,
    rng: np.random.Generator,
    temperature: float,
    approximate: bool,
    norm_diffs: bool,
    use_hinge: bool,
    true_reward: Optional[np.ndarray] = None,
) -> None:
    # TODO: Try normalizing diffs
    if norm_diffs:
        diffs = normalize(diffs)
    ndims = diffs.shape[1]

    reward_likelihood = boltzmann_likelihood if not use_hinge else hinge_likelihood

    samples = cover_sphere(n_samples=n_reward_samples, ndims=ndims, rng=rng)

    if true_reward is not None:
        samples = np.concatenate((samples, true_reward.reshape(1, -1)))

    per_diff_likelihoods = reward_likelihood(
        reward=samples, diffs=diffs, temperature=temperature, approximate=approximate
    )
    likelihoods = cum_likelihoods(per_diff_likelihoods, shift=True)

    plot_counts(counts=np.sum(likelihoods > 0.0, axis=0), outdir=outdir)

    map_rewards = samples[np.argmax(likelihoods, axis=0)]
    plot_rewards(map_rewards, outdir, outname="map_reward")

    if true_reward is not None:
        plot_gt_likelihood(likelihoods, outdir)

        true_reward_copies = np.tile(true_reward, (len(diffs), 1))
        dispersions_gt = mean_geodesic_dispersion(
            reward_samples=samples,
            likelihoods=likelihoods,
            target_rewards=true_reward_copies,
        )
        np.save(outdir / "dispersion_gt.npy", dispersions_gt)
        plot_dispersions(dispersions_gt, outdir, outname="dispersion_gt")

    mean_rewards = find_means(rewards=samples, likelihoods=likelihoods)
    proj_mean_rewards = normalize(mean_rewards)

    plot_rewards(mean_rewards, outdir, outname="mean_reward")
    plot_rewards(proj_mean_rewards, outdir, outname="proj_mean_reward")

    log_big_shifts(diffs, mean_rewards)

    # TODO: Try not exponetiation and using logits as raw probabilities
    dispersions_mean = mean_geodesic_dispersion(
        reward_samples=samples,
        likelihoods=likelihoods,
        target_rewards=proj_mean_rewards,
        expect_monotonic=False,
    )

    np.save(outdir / "dispersion_mean.npy", dispersions_mean)
    plot_dispersions(dispersions_mean, outdir, outname="dispersion_mean")


def log_big_shifts(diffs: np.ndarray, mean_rewards: np.ndarray) -> None:
    delta_mean_reward = np.max(np.abs(mean_rewards[:-1] - mean_rewards[1:]), axis=1)
    assert delta_mean_reward.shape == (
        len(diffs) - 1,
    ), f"delta mean rewards is {delta_mean_reward.shape} expected {(len(diffs) - 1,)}"
    big_shift = np.concatenate(([False], delta_mean_reward > 0.01))
    if np.any(big_shift):
        logging.info(
            f"There are some big one timestep shifts. The following are the preferences that caused them:\n{diffs[big_shift]}\nat {np.where(big_shift)}"
        )
        logging.info(
            f"For comparison, here are some small shifts:\n{diffs[np.logical_not(big_shift)][:5]}"
        )


def cover_sphere(n_samples: int, ndims: int, rng: np.random.Generator) -> np.ndarray:
    samples = rng.multivariate_normal(mean=np.zeros(ndims), cov=np.eye(ndims), size=(n_samples))
    samples = normalize(samples)
    return samples


def mean_geodesic_dispersion(
    reward_samples: np.ndarray,
    likelihoods: np.ndarray,
    target_rewards: np.ndarray,
    expect_monotonic: bool = False,
) -> np.ndarray:
    # Arc length is angle times radius, and the radius is 1, so the arc length between the mean
    # reward and each sample is just the angle between them, which you can get using the standard
    # cos(theta) = a @ b / (|a| * |b|) trick, and the norm of all vectors is 1.
    logging.debug(
        f"reward samples shape={reward_samples.shape}, target rewards shape={target_rewards.shape}"
    )
    assert np.allclose(np.linalg.norm(reward_samples, axis=1), 1)
    assert np.allclose(np.linalg.norm(target_rewards, axis=1), 1)
    dots = np.clip(reward_samples @ target_rewards.T, -1.0, 1.0)

    dists = np.arccos(dots)
    logging.debug(
        f"samples {reward_samples.shape} likelihoods {likelihoods.shape} targets {target_rewards.shape} dists {dists.shape}"
    )
    assert dists.shape == (
        len(reward_samples),
        len(target_rewards),
    ), f"dists shape={dists.shape}, expected {(len(reward_samples), len(target_rewards))}"
    assert not np.any(np.all(dists == 0.0, axis=0))

    weighted_dists = np.stack(
        [np.average(dists[:, i], weights=likelihoods[:, i]) for i in range(dists.shape[1])]
    )

    if expect_monotonic and __debug__:
        for t in range(len(weighted_dists) - 1):
            if weighted_dists[t] < weighted_dists[t + 1]:
                # This should never happen. Let's figure out why

                d_1 = dists[:, t]
                d_2 = dists[:, t + 1]

                l_1 = likelihoods[:, t]
                l_2 = likelihoods[:, t + 1]

                denominator_1 = np.sum(l_1)
                denominator_2 = np.sum(l_2)

                assert (
                    denominator_2 - denominator_1 <= 1e-8
                ), f"Total likelihood increased between t={t} and {t+1} from {denominator_1} to {denominator_2}"

                # What rewards are most contributing to the distances?
                contributions_2 = d_2 * l_2 / denominator_2
                assert np.allclose(np.sum(contributions_2), weighted_dists[t + 1])
                big_indices_2 = np.argsort(contributions_2)[::-1][:10]
                big_rewards_2 = reward_samples[big_indices_2]
                big_dists_2 = d_2[big_indices_2]
                big_likelihoods_2 = l_2[big_indices_2]
                big_contributions_2 = contributions_2[big_indices_2]

                logging.info(f"{np.sum(l_2 == 1)} likelihoods are 1.")

                logging.warning(
                    f"Average dispersion went up from {weighted_dists[t]} to {weighted_dists[t+1]} on timestep={t}.\nThe largest terms are rewards=\n{big_rewards_2}\ntarget={target_rewards[t+1]}\ndists=\n{big_dists_2}\nlikelihoods=\n{big_likelihoods_2}\ndenom={denominator_2}\ncontributions=\n{big_contributions_2}"
                )

                # What rewards's contributions changed the most from the last timestep
                contributions_1 = d_1 * l_1 / denominator_1
                assert np.allclose(np.sum(contributions_1), weighted_dists[t])
                diffs = contributions_2 - contributions_1
                assert np.allclose(np.sum(diffs), np.sum(contributions_2) - np.sum(contributions_1))
                big_diff_indices = np.argsort(diffs)[::-1][:10]
                big_diff_rewards = reward_samples[big_diff_indices]
                big_diff_dists = d_2[big_diff_indices] - d_1[big_diff_indices]
                big_diff_likelihoods = l_2[big_diff_indices] - l_1[big_diff_indices]
                big_diffs = diffs[big_diff_indices]
                logging.warning(
                    f"The largest difference is due to rewards=\n{big_diff_rewards}\n with differences of targets={target_rewards[t+1] - target_rewards[t]}\ndists=\n{big_diff_dists}\nlikelihoods=\n{big_diff_likelihoods}\ndenom={denominator_1 - denominator_2}\ncontributions=\n{big_diffs}"
                )

    return weighted_dists


def entropy(likelihoods: np.ndarray) -> np.ndarray:
    """Compute the entropy of a set of likelihoods.

    Args:
        likelihoods (np.ndarray): A set of likelihoods.

    Returns:
        np.ndarray: The entropy of each likelihood.
    """
    l = np.ma.masked_equal(likelihoods, 0.0)
    return -np.sum(l * np.log(l), axis=0)


def find_means(rewards: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    mean_rewards = np.stack(
        [
            np.average(rewards, weights=likelihoods[:, t], axis=0)
            for t in range(likelihoods.shape[1])
        ]
    )
    assert mean_rewards.shape == (
        likelihoods.shape[1],
        rewards.shape[1],
    ), f"mean_rewards shape={mean_rewards.shape}, expected {(likelihoods.shape[1],rewards.shape[1])}"
    assert np.all(np.isfinite(mean_rewards))
    return mean_rewards


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes the vectors in an array on the 1th axis.

    Args:
        x (np.ndarray): 2D array of vectors.

    Returns:
        np.ndarray: 2D array x such that np.linalg.norm(x, axis=1) == 1
    """
    shape = x.shape
    out: np.ndarray = (x.T / np.linalg.norm(x, axis=1)).T
    assert out.shape == shape, f"shape: expected={shape} actual={out.shape}"
    assert np.allclose(
        np.linalg.norm(out, axis=1), 1
    ), f"norm: expected={1} actual={np.linalg.norm(out, axis=1)}"
    return out


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


def plot_gt_likelihood(likelihoods: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path) -> None:
    logging.info("Plotting likelihood")
    if isinstance(likelihoods, dict):
        for name, l in likelihoods.items():
            plt.plot(l[-1], label=name)
        plt.legend()
    else:
        plt.plot(likelihoods[-1])
    plt.title("Ground truth posterior likelihood")
    plt.xlabel("Human preferences")
    plt.ylabel("Likelihood of true reward")
    plt.savefig(outdir / "gt_likelihood.png")
    plt.close()


def plot_dispersions(
    dispersions: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, outname: str
) -> None:
    if isinstance(dispersions, dict):
        for name, d in dispersions.items():
            plt.plot(d, label=name)
        plt.legend()
    else:
        plt.plot(dispersions)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / f"{outname}.png")
    plt.close()

    if isinstance(dispersions, dict):
        for name, d in dispersions.items():
            log_dispersion = np.log(d)
            log_dispersion[log_dispersion == -np.inf] = None
            plt.plot(log_dispersion, label=outname)
        plt.legend()
    else:
        log_dispersion = np.log(dispersions)
        log_dispersion[log_dispersion == -np.inf] = None
        plt.plot(log_dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("log(mean dispersion)")
    plt.title("Log-concentration of posterior with data")
    plt.savefig(outdir / f"log_{outname}.png")
    plt.close()


def plot_counts(
    counts: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, threshold: int = 200
) -> None:
    max_count = (
        max(np.max(c) for c in counts.values()) if isinstance(counts, dict) else np.max(counts)
    )
    if isinstance(counts, dict):
        for name, c in counts.items():
            plt.plot(c, label=name)
        plt.legend()
    else:
        plt.plot(counts)
    plt.title("Number of rewards with nonzero likelihood")
    plt.xlabel("Number of preferences")
    plt.ylabel("Count")
    plt.ylim((0, max_count * 1.05))
    plt.savefig(outdir / "counts.png")
    plt.close()

    if isinstance(counts, dict):
        plot_small = any(np.any(c < threshold) for c in counts.values())
    else:
        plot_small = bool(np.any(counts < threshold))
    if plot_small:
        if isinstance(counts, dict):
            for name, c in counts.items():
                plt.plot(c[c < threshold], label=name)
            plt.legend()
        else:
            plt.plot(counts[counts < threshold])
        plt.title("Number of rewards with nonzero likelihood")
        plt.xlabel("Number of preferences")
        plt.ylabel("Count")
        plt.savefig(outdir / "small_counts.png")
        plt.close()


def plot_rewards(
    rewards: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path, outname: str
) -> None:
    ndims = list(rewards.values())[0].shape[1] if isinstance(rewards, dict) else rewards.shape[1]
    for dim in range(ndims):
        if isinstance(rewards, dict):
            for name, r in rewards.items():
                plt.plot(r[:, dim], label=name)
            plt.legend()
        else:
            plt.plot(rewards[:, dim])
        plt.ylim(-1, 1)
        plt.xlabel("Preferences")
        plt.ylabel(f"{dim}-th dimension of reward")
        plt.title(f"{dim}-th dimension of reward")
        plt.savefig(outdir / f"{dim}.{outname}.png")
        plt.close()


def plot_liklihoods(likelihoods: Union[Dict[str, np.ndarray], np.ndarray], outdir: Path) -> None:
    def plot(likelihoods: pd.DataFrame, outdir: Path, name: str) -> None:
        df = pd.DataFrame(likelihoods, dtype=np.float128)
        assert df.notnull().all().all()
        assert (df < np.inf).all().all()
        df = df.melt(
            value_vars=range(likelihoods.shape[1]), var_name="timestep", value_name="likelihood"
        )

        n_plots = min(10, likelihoods.shape[1])
        logging.debug(f"n_plots={n_plots}")
        timesteps = np.arange(0, likelihoods.shape[1], ceil(likelihoods.shape[1] / n_plots))
        assert len(timesteps) == n_plots, f"{len(timesteps)} != {n_plots}"

        df = df.loc[df["timestep"].isin(timesteps)]
        df = df.loc[df.timestep > 1e-3]

        small_df = df.loc[df.likelihood < 0.1]
        large_df = df.loc[df.likelihood >= 0.1]

        assert large_df.notnull().all().all()
        assert (large_df.abs() < np.inf).all().all()

        logging.debug(f"small_df min={small_df.likelihood.min()} max={small_df.likelihood.max()}")
        logging.debug(f"large_df min={large_df.likelihood.min()} max={large_df.likelihood.max()}")

        if len(small_df) > 0:
            fig, _ = joypy.joyplot(small_df, hist=True, by="timestep", overlap=0, bins=100)
            fig.savefig(outdir / f"{name}.small.png")
            plt.close(fig)

        if len(large_df) > 0:
            fig, _ = joypy.joyplot(large_df, hist=True, by="timestep", overlap=0, bins=100)
            fig.savefig(outdir / f"{name}.large.png")
            plt.close(fig)

    if isinstance(likelihoods, dict):
        for name, l in likelihoods.items():
            plot(l, outdir, name=f"likelihood_hist.{name}")
    else:
        plot(likelihoods, outdir, name="likelihood_hist")


def plot_entropies(entropies: Dict[str, np.ndarray], outdir: Path) -> None:
    for name, e in entropies.items():
        plt.plot(e, label=name)
    plt.legend()
    plt.title("Posterior entropy")
    plt.ylabel("Entropy")
    plt.xlabel("Human preferences")
    plt.savefig(outdir / "entropy.png")
    plt.close()


if __name__ == "__main__":
    fire.Fire(
        {"single": one_modality_analysis, "compare": compare_modalities, "joint": analysis_joint}
    )

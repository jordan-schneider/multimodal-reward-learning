import logging
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from numpy.lib.function_base import disp
from numpy.lib.ufunclike import isneginf
from scipy.optimize import linprog  # type: ignore
from scipy.spatial.distance import cosine  # type: ignore
from scipy.special import logsumexp  # type: ignore
from tqdm import tqdm  # type: ignore


def dedup(normals: np.ndarray, precision: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Remove halfspaces that have small cosine similarity to another."""
    out: List[np.ndarray] = []
    counts: List[int] = []
    # Remove exact duplicates
    _, indices = np.unique(normals, return_index=True, axis=0)

    for normal in normals[indices]:
        unique = True
        for j, accepted_normal in enumerate(out):
            if cosine(normal, accepted_normal) < precision:
                counts[j] += 1
                unique = False
                break
        if unique:
            out.append(normal)
            counts.append(1)

    return np.array(out).reshape(-1, normals.shape[1]), np.array(counts)


def log_normalize_logs(x: np.ndarray) -> np.ndarray:
    logging.debug(f"min={np.min(x)}, max={np.max(x)}, ratio={np.max(x) - np.min(x)}")
    denom = logsumexp(x, axis=0)
    logging.debug(f"min denom={np.min(denom)}, max={np.max(denom)}")
    out = x - denom
    if np.any(np.isneginf(out)):
        logging.warning("Some counted halfplanes have -inf log likelihood")
    if np.any(np.exp(out) == 0):
        logging.warning("Some counted halfplanes have 0 likelihood")
    return out


def reward_prop_likelihood_by_diff(
    reward: np.ndarray,
    diffs: np.ndarray,
    temperature: float = 1.0,
    approximate: bool = False,
) -> np.ndarray:
    """Return the proportional likelihood of a reward given a set of reward feature differences.

    Args:
        reward (np.ndarray): Reward or batch of rewards to determine likelihood of.
        diffs (np.ndarray): Differences between features of preferred and dispreffered objects.
        weights (np.ndarray, optional): How many copies of each difference vector are present. Defaults to None, in which case all counts are assumed to be 1.

    Returns:
        np.ndarray: (Batch of) proportional likelihoods of each reward under each halfplane
    """
    if len(reward.shape) == 1:
        reward = reward.reshape(1, -1)
    assert len(diffs) > 0

    # This function assumes that the reward posterior is defined on the unit sphere by restricting
    # the given likelihood to exactly the sphere, rather than taking a quotient space (by projecting
    # the likelihood for all rewards on every ray to their unit length point. If I ever want to do
    # that instead, the likelihood is |w| * log(1/2 * (1 + exp(w @ diffs))) / (w @ diffs) in general
    # and (log(1/2) + log1p(exp(w @ diffs))) / (w @ diffs) in our case, as |w|=1.
    strengths = temperature * (reward @ diffs.T).T
    if approximate:
        log_likelihoods = strengths
    else:
        exp_strengths = np.exp(-strengths)

        infs = np.isinf(exp_strengths)
        not_infs = np.logical_not(infs)

        log_likelihoods = np.empty((len(diffs), len(reward)))

        # If np.exp(...) is inf, then 1 + np.exp(...) is approximately np.exp(...)
        # so log1p(exp(-reward @ diffs))) \approx rewards @ diffs
        log_likelihoods[infs] = strengths[infs]
        log_likelihoods[not_infs] = -np.log1p(exp_strengths[not_infs])

    log_likelihoods = log_likelihoods.T
    assert log_likelihoods.shape == (len(reward), len(diffs))

    if np.any(np.isneginf(log_likelihoods)):
        logging.warning("Some counted halfplanes have -inf log likelihood")
    if np.any(np.exp(log_likelihoods) == 0):
        logging.warning("Some counted halfplanes have 0 likelihood")

    return log_likelihoods


def cover_sphere(n_samples: int, ndims: int, rng: np.random.Generator) -> np.ndarray:
    samples = rng.multivariate_normal(mean=np.zeros(ndims), cov=np.eye(ndims), size=(n_samples))
    samples = (samples.T / np.linalg.norm(samples, axis=1)).T
    assert np.allclose(np.linalg.norm(samples, axis=1), 1.0)
    return samples


def cum_reward_likelihoods(
    reward_samples: np.ndarray,
    diffs: np.ndarray,
    temperature: float = 1.0,
    approximate: bool = False,
):
    log_likelihoods = reward_prop_likelihood_by_diff(
        reward_samples, diffs, temperature, approximate
    )

    log_total_likelihoods = np.cumsum(log_likelihoods, axis=1)
    assert log_total_likelihoods.shape == (len(reward_samples), len(diffs))

    if np.any(np.isneginf(log_total_likelihoods)):
        logging.warning("Some rewards have -inf log total likelihood")
    if np.any(np.exp(log_total_likelihoods) == 0):
        logging.warning("Some rewards have 0 total unnormalized likelihood")

    log_total_likelihoods = log_normalize_logs(log_total_likelihoods)

    log_total_likelihoods = log_shift(log_total_likelihoods)

    total_likelihoods = np.exp(log_total_likelihoods)
    assert total_likelihoods.shape == (len(reward_samples), len(diffs))

    if np.any(total_likelihoods == 0):
        logging.warning("Some rewards have 0 total likelihood")
    assert np.all(np.isfinite(total_likelihoods))
    return total_likelihoods


def mean_l2_dispersions(
    reward_samples: np.ndarray,
    likelihoods: np.ndarray,
    target_rewards: np.ndarray,
) -> np.ndarray:
    # Arc length is angle times radius, and the radius is 1, so the arc length between the mean
    # reward and each sample is just the angle between them, which you can get using the standard
    # cos(theta) = a @ b / (|a| * |b|) trick, and the norm of all vectors is 1.
    logging.debug(
        f"reward samples shape={reward_samples.shape}, target rewards shape={target_rewards.shape}"
    )
    dots = reward_samples @ target_rewards.T
    dots[dots < -1.0] = -1.0
    dots[dots > 1.0] = 1.0

    # TODO: Try entropy or exp entropy or something

    dists = np.arccos(dots)
    logging.debug(
        f"samples {reward_samples.shape} likelihoods {likelihoods.shape} targets {target_rewards.shape} dists {dists.shape}"
    )
    assert dists.shape == (
        len(reward_samples),
        len(target_rewards),
    ), f"dists shape={dists.shape}, expected {(len(reward_samples), len(target_rewards))}"
    assert not np.any(np.all(dists == 0.0, axis=0))

    weighted_dists = np.zeros((len(target_rewards),))
    for i in range(len(target_rewards)):
        # TODO: There might be a less terrible way to do this, but np.average can't handle it
        weighted_dists[i] = np.average(dists[:, i], weights=likelihoods[:, i], axis=0)

    return weighted_dists


def find_means(rewards: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    mean_rewards = np.stack(
        [
            np.average(rewards, weights=likelihoods[:, i], axis=0)
            for i in range(likelihoods.shape[1])
        ]
    )
    assert mean_rewards.shape == (
        likelihoods.shape[1],
        rewards.shape[1],
    ), f"mean_rewards shape={mean_rewards.shape}, expected {(likelihoods.shape[1],rewards.shape[1])}"
    assert np.all(np.isfinite(mean_rewards))
    return mean_rewards


def log_shift(log_total_likelihoods: np.ndarray) -> np.ndarray:
    smallest_meaningful_log = np.log(np.finfo(np.float64).tiny)
    largest_meainingful_log = np.log(np.finfo(np.float64).max)
    max_log_shift = largest_meainingful_log - np.max(log_total_likelihoods) - 100
    assert max_log_shift > 0
    ideal_log_shift = smallest_meaningful_log - np.min(log_total_likelihoods) + 1
    log_shift = max(0, min(ideal_log_shift, max_log_shift))
    logging.info(f"ideal_log_shift={ideal_log_shift}, max_log_shift={max_log_shift}")
    log_total_likelihoods += log_shift

    if __debug__:
        total_likelihoods = np.sum(np.exp(log_total_likelihoods), axis=0)
        good_indices = np.logical_or(
            np.isclose(total_likelihoods, np.exp(log_shift)), np.isclose(total_likelihoods, 0)
        )
        bad_indices = np.logical_not(good_indices)
        if np.any(bad_indices):
            logging.warning(
                f"Some likelihoods don't add up to {np.exp(log_shift)} or 0:{total_likelihoods[bad_indices]}"
            )

    return log_total_likelihoods


def analysis(
    outdir: Path,
    diffs: np.ndarray,
    n_reward_samples: int,
    rng: np.random.Generator,
    temperature: float,
    approximate: bool,
    norm_diffs: bool,
    true_reward: Optional[np.ndarray] = None,
) -> None:
    # TODO: Try normalizing diffs
    if norm_diffs:
        diffs = (diffs.T / np.linalg.norm(diffs, axis=1)).T
    ndims = diffs.shape[1]

    samples = cover_sphere(n_samples=n_reward_samples, ndims=ndims, rng=rng)

    if true_reward is not None:
        samples = np.concatenate((samples, true_reward.reshape(1, -1)))

    likelihoods = cum_reward_likelihoods(
        reward_samples=samples,
        diffs=diffs,
        temperature=temperature,
        approximate=approximate,
    )

    plot_counts(outdir, counts=np.sum(likelihoods > 0.0, axis=0))

    map_rewards = samples[np.argmax(likelihoods, axis=0)]
    plot_rewards(map_rewards, outdir, name="map_reward")

    if true_reward is not None:
        plot_gt_likelihood(outdir, likelihoods)

        true_reward_copies = np.tile(true_reward, (len(diffs), 1))
        dispersions_gt = mean_l2_dispersions(
            reward_samples=samples,
            likelihoods=likelihoods,
            target_rewards=true_reward_copies,
        )
        np.save(outdir / "dispersion_gt.npy", dispersions_gt)
        plot_dispersions(outdir, dispersions_gt, name="dispersion_gt")

    mean_rewards = find_means(rewards=samples, likelihoods=likelihoods)
    proj_mean_rewards = (mean_rewards.T / np.linalg.norm(mean_rewards, axis=1)).T

    plot_rewards(mean_rewards, outdir, name="mean_reward")
    plot_rewards(proj_mean_rewards, outdir, name="proj_mean_reward")

    log_big_shifts(diffs, mean_rewards)

    # TODO: Try not exponetiation and using logits as raw probabilities
    dispersions_mean = mean_l2_dispersions(
        reward_samples=samples,
        likelihoods=likelihoods,
        target_rewards=proj_mean_rewards,
    )

    np.save(outdir / "dispersion_mean.npy", dispersions_mean)
    plot_dispersions(outdir, dispersions_mean, name="dispersion_mean")


def one_modality_analysis(
    in_path: Path,
    outdir: Path,
    n_samples: int = 1000,
    max_preferences: int = 1000,
    temperature: float = 1.0,
    approximate: bool = False,
    reward_path: Optional[Path] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    logging.basicConfig(level=verbosity)
    in_path, outdir = Path(in_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diffs = np.load(in_path)[:max_preferences]
    true_reward = np.load(reward_path) if reward_path is not None else None

    analysis(
        outdir=outdir,
        diffs=diffs,
        n_reward_samples=n_samples,
        rng=np.random.default_rng(),
        temperature=temperature,
        approximate=approximate,
        true_reward=true_reward,
    )


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


def plot_gt_likelihood(outdir: Path, likelihoods: np.ndarray) -> None:
    logging.info("Plotting likelihood")
    true_likelihoods = likelihoods[-1]
    # true_likelihoods = true_likelihoods[true_likelihoods <= 1e10]
    true_likelihoods = true_likelihoods[50:]
    plt.plot(true_likelihoods)
    plt.title("Ground truth posterior likelihood")
    plt.xlabel("Human preferences")
    plt.ylabel("True reward posterior")
    plt.savefig(outdir / "gt_likelihood.png")
    plt.close()


def plot_dispersions(outdir: Path, dispersions: np.ndarray, name: str) -> None:
    plt.plot(dispersions)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / f"{name}.png")
    plt.close()

    log_dispersion = np.log(dispersions)
    log_dispersion[log_dispersion == -np.inf] = None
    plt.plot(log_dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Log-mean dispersion")
    plt.title("Log-concentration of posterior with data")
    plt.savefig(outdir / f"log_{name}.png")
    plt.close()


def plot_counts(outdir: Path, counts: np.ndarray, threshold: int = 200) -> None:
    plt.plot(counts)
    plt.title("Number of rewards with nonzero likelihood")
    plt.xlabel("Number of preferences")
    plt.ylabel("Count")
    plt.savefig(outdir / "counts.png")
    plt.close()

    small_counts = counts[counts < threshold]
    if np.any(small_counts):
        plt.plot(small_counts)
        plt.title("Number of rewards with nonzero likelihood")
        plt.xlabel("Number of preferences")
        plt.ylabel("Count")
        plt.savefig(outdir / "small_counts.png")
        plt.close()


def plot_rewards(rewards: np.ndarray, outdir: Path, name: str) -> None:
    for i, dimension in enumerate(rewards.T):
        plt.plot(dimension)
        plt.ylim(-1, 1)
        plt.xlabel("Preferences")
        plt.ylabel(f"{i}-th dimension of reward")
        plt.title(f"{i}-th dimension of reward")
        plt.savefig(outdir / f"{i}.{name}.png")
        plt.close()


def compare_modalities(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    action_path: Optional[Path] = None,
    n_samples: int = 100_000,
    max_comparisons: int = 1000,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if traj_path is not None:
        paths["traj"] = Path(traj_path)
    if state_path is not None:
        paths["state"] = Path(state_path)
    if action_path is not None:
        paths["action"] = Path(action_path)

    rng = np.random.default_rng()

    diffs = {
        key: gather(in_path.parent, in_path.name, n=max_comparisons)
        for key, in_path in paths.items()
    }
    diffs["joint"] = np.concatenate(
        [diff[: max_comparisons // len(paths)] for diff in diffs.values()]
    )
    rng.shuffle(diffs["joint"])
    ndims = list(diffs.values())[0].shape[1]  # type: ignore
    reward_samples = cover_sphere(n_samples, ndims, rng)

    likelihoods = {key: cum_reward_likelihoods(reward_samples, diff) for key, diff in diffs.items()}
    mean_rewards = {
        key: find_means(rewards=reward_samples, likelihoods=likelihood)
        for key, likelihood in likelihoods.items()
    }

    dispersions_mean = {
        key: mean_l2_dispersions(
            reward_samples,
            likelihoods=cum_reward_likelihoods(reward_samples, diff),
            target_rewards=mean_rewards[key],
        )
        for key, diff in diffs.items()
    }
    pkl.dump(dispersions_mean, (outdir / "dispersions_mean.pkl").open("wb"))

    for name, dispersion in dispersions_mean.items():
        plt.plot(dispersion, label=name)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior for different modalities")
    plt.legend()
    plt.savefig(outdir / "comparison_mean.png")
    plt.close()


def gather(indir: Path, name: str, n: int) -> np.ndarray:
    paths = list(indir.glob(f"{name}.[0-9]*.npy"))
    data = []
    current_size = 0
    while current_size < n and len(paths) > 0:
        path = paths.pop()
        array = np.load(path)
        data.append(array)
        current_size += len(array)

    return np.concatenate(data)[:n]


def plot_joint_data(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    action_path: Optional[Path] = None,
    reward_path: Optional[Path] = None,
    n_reward_samples: int = 100_000,
    data_per_modality: int = 1000,
    temperature: float = 1.0,
    approximate: bool = False,
    verbosity: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
    logging.basicConfig(level=verbosity)

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

    diffs = np.concatenate([gather(path.parent, path.name, n=data_per_modality) for path in paths])
    rng.shuffle(diffs)

    true_reward = np.load(reward_path) if reward_path is not None else None

    analysis(
        outdir,
        diffs,
        n_reward_samples,
        rng,
        temperature=temperature,
        approximate=approximate,
        true_reward=true_reward,
    )


if __name__ == "__main__":
    fire.Fire(
        {"single": one_modality_analysis, "compare": compare_modalities, "joint": plot_joint_data}
    )

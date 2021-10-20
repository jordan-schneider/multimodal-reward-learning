import logging
import pickle as pkl
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from scipy.spatial.distance import cosine  # type: ignore

from mrl.reward_model.boltzmann import boltzmann_likelihood
from mrl.reward_model.hinge import hinge_likelihood
from mrl.reward_model.logspace import cum_likelihoods
from mrl.util import np_gather, setup_logging


def cover_sphere(n_samples: int, ndims: int, rng: np.random.Generator) -> np.ndarray:
    samples = rng.multivariate_normal(mean=np.zeros(ndims), cov=np.eye(ndims), size=(n_samples))
    samples = normalize(samples)
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
    # TODO: Finish


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
    assert np.allclose(np.linalg.norm(reward_samples, axis=1), 1)
    assert np.allclose(np.linalg.norm(target_rewards, axis=1), 1)
    dots = np.clip(reward_samples @ target_rewards.T, -1.0, 1.0)

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


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalizes the vectors in an array on the 1th axis.

    Args:
        x (np.ndarray): 2D array of vectors.

    Returns:
        np.ndarray: 2D array x such that np.linalg.norm(x, axis=1) == 1
    """
    shape = x.shape
    norms = np.linalg.norm(x, axis=1)
    out = (x.T / norms).T
    assert out.shape == shape, f"shape: expected={shape} actual={out.shape}"
    assert np.allclose(
        np.linalg.norm(out, axis=1), 1
    ), f"norm: expected={1} actual={np.linalg.norm(out, axis=1)}"
    return out


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
    likelihoods = cum_likelihoods(per_diff_likelihoods)

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
    proj_mean_rewards = normalize(mean_rewards)

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
    norm_diffs: bool = False,
    use_hinge: bool = False,
    reward_path: Optional[Path] = None,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    in_path, outdir = Path(in_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    diffs = np_gather(in_path.parent, in_path.name, max_preferences)
    true_reward = np.load(reward_path) if reward_path is not None else None

    analysis(
        outdir=outdir,
        diffs=diffs,
        n_reward_samples=n_samples,
        rng=np.random.default_rng(),
        temperature=temperature,
        approximate=approximate,
        norm_diffs=norm_diffs,
        use_hinge=use_hinge,
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
    reward_path: Optional[Path] = None,
    n_samples: int = 100_000,
    max_comparisons: int = 1000,
    norm_diffs: bool = False,
    use_hinge: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    setup_logging(level=verbosity)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    if traj_path is not None:
        paths["traj"] = Path(traj_path)
    if state_path is not None:
        paths["state"] = Path(state_path)
    if action_path is not None:
        paths["action"] = Path(action_path)

    if reward_path is not None:
        true_reward = np.load(reward_path)

    reward_likelihood = boltzmann_likelihood if not use_hinge else hinge_likelihood

    rng = np.random.default_rng()

    diffs = {
        key: np_gather(in_path.parent, in_path.name, n=max_comparisons)
        for key, in_path in paths.items()
    }
    diffs["joint"] = np.concatenate(
        [diff[: max_comparisons // len(paths)] for diff in diffs.values()]
    )
    rng.shuffle(diffs["joint"])

    if norm_diffs:
        diffs = {key: normalize(diff) for key, diff in diffs.items()}

    ndims = list(diffs.values())[0].shape[1]  # type: ignore
    reward_samples = cover_sphere(n_samples, ndims, rng)

    if reward_path is not None:
        reward_samples = np.concatenate((reward_samples, [true_reward]), axis=0)

    likelihoods = {
        key: cum_likelihoods(reward_likelihood(reward=reward_samples, diffs=diff))
        for key, diff in diffs.items()
    }
    mean_rewards = {
        key: normalize(find_means(rewards=reward_samples, likelihoods=likelihood))
        for key, likelihood in likelihoods.items()
    }

    # for dim in range(ndims):
    #     for name, mean in mean_rewards.items():
    #         plt.plot(mean[:, dim], label=name)
    #     plt.ylim(-1, 1)
    #     plt.xlabel("Preferences")
    #     plt.ylabel(f"{dim}-th dimension of reward")
    #     plt.title(f"{dim}-th dimension of reward")
    #     plt.legend()
    #     plt.savefig(outdir / f"{dim}.mean_reward.png")
    #     plt.close()

    dispersions_mean = {
        key: mean_l2_dispersions(
            reward_samples=reward_samples,
            likelihoods=cum_likelihoods(reward_likelihood(reward_samples, diff)),
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
    plt.savefig(outdir / "dispersion_mean.png")
    plt.close()

    # if reward_path is not None:
    #     true_reward_index = np.where(np.all(reward_samples == true_reward, axis=1))[0][0]
    #     assert true_reward_index == len(list(likelihoods.values())[0]) - 1
    #     gt_likelihood = {key: l[-1] for key, l in likelihoods.items()}
    #     for name, likelihood in gt_likelihood.items():
    #         plt.plot(likelihood, label=name)
    #     plt.title("Likelihood of ground truth reward")
    #     plt.xlabel("Human preferences")
    #     plt.ylabel("Posterior likelihood")
    #     plt.legend()
    #     plt.savefig(outdir / "gt_likelihood.png")
    #     plt.close()

    # true_reward_copies = np.tile(true_reward, (max_comparisons, 1))
    # dispersions_gt = {
    #     key: mean_l2_dispersions(
    #         reward_samples=reward_samples,
    #         likelihoods=l,
    #         target_rewards=true_reward_copies,
    #     )
    #     for key, l in likelihoods.items()
    # }

    # for name, dispersion in dispersions_gt.items():
    #     plt.plot(dispersion, label=name)
    # plt.xlabel("Human preferences")
    # plt.ylabel("Mean dispersion")
    # plt.title("Concentration of posterior for different modalities")
    # plt.legend()
    # plt.savefig(outdir / "dispersion_gt.png")
    # plt.close()


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
    norm_diffs: bool = False,
    use_hinge: bool = False,
    verbosity: Literal["DEBUG", "INFO"] = "INFO",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()
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
        [np_gather(path.parent, path.name, n=data_per_modality) for path in paths]
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


if __name__ == "__main__":
    fire.Fire(
        {"single": one_modality_analysis, "compare": compare_modalities, "joint": plot_joint_data}
    )

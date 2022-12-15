import logging
from typing import Dict, Optional, cast

import numpy as np
from tqdm import trange  # type: ignore

from mrl.aligned_rewards.aligned_reward_set import AlignedRewardSet
from mrl.inference.results import Results
from mrl.inference.sphere import find_centroid
from mrl.util import normalize_vecs


def analysis(
    results: Results,
    aligned_reward_set: Optional[AlignedRewardSet] = None,
    compute_centroids: bool = False,
    compute_mean_dispersions: bool = False,
):
    results.start("")
    reward_samples = cast(np.ndarray, results.get("reward_sample"))
    true_reward = (
        cast(np.ndarray, results.get("true_reward"))
        if results.has("true_reward")
        else None
    )

    for trial in results.experiment_names():
        if trial == "":
            continue
        results.start(trial)

        likelihoods = cast(Dict[str, np.ndarray], results.get("likelihood"))
        if likelihoods is None:
            continue
        if aligned_reward_set is not None:
            logging.info("Computing P(aligned)")
            prob_aligned = {
                key: aligned_reward_set.prob_aligned(
                    rewards=reward_samples, densities=l
                )
                for key, l in likelihoods.items()
            }
            results.update("prob_aligned", prob_aligned)

        logging.info("Computing posterior entropies")
        entropies = {key: entropy(l) for key, l in likelihoods.items()}
        results.update("entropy", entropies)

        logging.info("Computing nonzero likelihood counts")
        counts = {key: np.sum(l > 0.0, axis=0) for key, l in likelihoods.items()}
        results.update("count", counts)

        logging.info("Computing raw and normalized mean rewards")
        save_means(reward_samples, likelihoods, results)

        if compute_centroids:
            logging.info("Finding centroid dispersion statistics")
            save_centroid_dispersions(reward_samples, likelihoods, results)

        if compute_mean_dispersions:
            logging.info("Finding mean dispersion")
            proj_mean_rewards = results.get("proj_mean_reward")
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
            logging.info("Finding gt dispersion")
            save_gt_dispersion(reward_samples, true_reward, likelihoods, results)


def entropy(likelihoods: np.ndarray) -> np.ndarray:
    """Compute the entropy of a set of likelihoods.

    Args:
        likelihoods (np.ndarray): A set of likelihoods.

    Returns:
        np.ndarray: The entropy of each likelihood.
    """
    l = np.ma.masked_equal(likelihoods, 0.0)
    return -np.sum(l * np.log(l), axis=0)


def save_means(
    rewards: np.ndarray, likelihoods: Dict[str, np.ndarray], results: Results
):
    mean_rewards = {}
    proj_mean_rewards = {}
    for key, l in likelihoods.items():
        mean_rewards[key] = find_means(rewards=rewards, likelihoods=l)
        proj_mean_rewards[key] = normalize_vecs(mean_rewards[key])
    results.update("mean_reward", mean_rewards)
    results.update("proj_mean_reward", proj_mean_rewards)


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


def save_gt_dispersion(
    reward_samples: np.ndarray,
    true_reward: np.ndarray,
    likelihoods: Dict[str, np.ndarray],
    results: Results,
):
    true_reward_index = np.where(np.all(reward_samples == true_reward, axis=1))[0][0]
    assert true_reward_index == len(list(likelihoods.values())[0]) - 1

    dispersions_gt = {
        key: mean_geodesic_dispersion(
            reward_samples=reward_samples,
            likelihoods=l,
            target_rewards=np.tile(true_reward, (l.shape[1], 1)),
        )
        for key, l in likelihoods.items()
    }
    results.update("dispersion_gt", dispersions_gt)


def save_centroid_dispersions(
    rewards: np.ndarray, likelihoods: Dict[str, np.ndarray], results: Results
) -> None:
    centroid_per_modality = {}
    dispersion_centroid_per_modality = {}
    proj_mean_rewards = results.get("proj_mean_reward")

    for key, l in likelihoods.items():
        centroids = []
        dispersions = []
        for t in trange(l.shape[1]):
            try:
                centroid, dist = find_centroid(
                    points=rewards,
                    weights=l[:, t],
                    max_iter=10,
                    init=proj_mean_rewards[key][t],
                )
                if np.any(np.isnan(centroid)):
                    continue
                assert np.allclose(
                    np.linalg.norm(centroid), 1.0
                ), f"centroid={centroid} has norm={np.linalg.norm(centroid)} far from 1."
                centroids.append(centroid)
                dispersions.append(dist)
            except Exception as e:
                logging.warning(f"Failed to find centroid for time t={t}", exc_info=e)

        centroid_per_modality[key] = np.stack(centroids)
        assert np.allclose(np.linalg.norm(centroid_per_modality[key], axis=1), 1.0)
        dispersion_centroid_per_modality[key] = np.array(dispersions)

    assert len(centroid_per_modality) == len(likelihoods)
    assert len(dispersion_centroid_per_modality) == len(likelihoods)
    results.update("centroid_per_modality", centroid_per_modality)
    results.update("dispersion_centroid", dispersion_centroid_per_modality)


def mean_geodesic_dispersion(
    reward_samples: np.ndarray,
    likelihoods: np.ndarray,
    target_rewards: np.ndarray,
    expect_monotonic: bool = False,
) -> np.ndarray:
    # Arc length is angle times radius, and the radius is 1, so the arc length between the mean
    # reward and each sample is just the angle between them, which you can get using the standard
    # cos(theta) = a @ b / (|a| * |b|) trick, and the norm of all vectors is 1.
    assert np.allclose(
        np.linalg.norm(reward_samples, axis=1), 1
    ), "Reward samples not normalized"
    assert np.allclose(np.linalg.norm(target_rewards, axis=1), 1)
    dots = np.clip(reward_samples @ target_rewards.T, -1.0, 1.0)

    dists = np.arccos(dots)
    assert dists.shape == (
        len(reward_samples),
        len(target_rewards),
    ), f"dists shape={dists.shape}, expected {(len(reward_samples), len(target_rewards))}"
    assert not np.any(np.all(dists == 0.0, axis=0))

    weighted_dists = np.stack(
        [
            np.average(dists[:, i], weights=likelihoods[:, i])
            for i in range(dists.shape[1])
        ]
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

                logging.info(f"{np.sum(l_2 == 1)} likelihoods are 1")

                logging.warning(
                    f"Average dispersion went up from {weighted_dists[t]} to {weighted_dists[t+1]} on timestep={t}.\nThe largest terms are rewards=\n{big_rewards_2}\ntarget={target_rewards[t+1]}\ndists=\n{big_dists_2}\nlikelihoods=\n{big_likelihoods_2}\ndenom={denominator_2}\ncontributions=\n{big_contributions_2}"
                )

                # What rewards's contributions changed the most from the last timestep
                contributions_1 = d_1 * l_1 / denominator_1
                assert np.allclose(np.sum(contributions_1), weighted_dists[t])
                diffs = contributions_2 - contributions_1
                assert np.allclose(
                    np.sum(diffs), np.sum(contributions_2) - np.sum(contributions_1)
                )
                big_diff_indices = np.argsort(diffs)[::-1][:10]
                big_diff_rewards = reward_samples[big_diff_indices]
                big_diff_dists = d_2[big_diff_indices] - d_1[big_diff_indices]
                big_diff_likelihoods = l_2[big_diff_indices] - l_1[big_diff_indices]
                big_diffs = diffs[big_diff_indices]
                logging.warning(
                    f"The largest difference is due to rewards=\n{big_diff_rewards}\n with differences of targets={target_rewards[t+1] - target_rewards[t]}\ndists=\n{big_diff_dists}\nlikelihoods=\n{big_diff_likelihoods}\ndenom={denominator_1 - denominator_2}\ncontributions=\n{big_diffs}"
                )

    return weighted_dists

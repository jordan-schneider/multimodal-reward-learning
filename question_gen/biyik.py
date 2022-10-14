import logging
from typing import Tuple

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances  # type: ignore

from question_gen.logspace import log_normalize_logs


def boltzmann_likelihood(
    reward: np.ndarray,
    diffs: np.ndarray,
    temperature: float = 1.0,
    approximate: bool = False,
) -> np.ndarray:
    """Returns the Boltzmann-rational likelihood of each reward under each feature difference.

    The Boltzmann-rational model for preferences is sometimes called the Luce-Shepard model.

    Args:
        reward (np.ndarray): Reward or batch of rewards to determine likelihood of.
        diffs (np.ndarray): Differences between features of preferred and dispreffered objects.
        temperature (float): Temperature parameter for Boltzmann distribution.
        approximate (bool): Use a large-value approximation for the liklihood.
    Returns:
        np.ndarray: (Batch of) log proportional likelihoods of each reward under each halfplane.
    """
    logging.debug(f"Inference temp={temperature}")
    if len(reward.shape) == 1:
        reward = reward.reshape(1, -1)
    assert len(diffs) > 0

    reward = reward.astype(np.float128)
    diffs = diffs.astype(np.float128)

    assert not np.any(np.isnan(reward))
    assert not np.any(np.isnan(diffs))

    # This function assumes that the reward posterior is defined on the unit sphere by restricting
    # the given likelihood to exactly the sphere, rather than taking a quotient space (by projecting
    # the likelihood for all rewards on every ray to their unit length point. If I ever want to do
    # that instead, the likelihood is |w| * log(1/2 * (1 + exp(w @ diffs))) / (w @ diffs) in general
    # and (log(1/2) + log1p(exp(w @ diffs))) / (w @ diffs) in our case, as |w|=1.
    strengths = (reward @ diffs.T) / temperature
    logging.debug(
        f"reward={reward}, diffs nan={np.any(np.isnan(diffs))}, temp={temperature}"
    )
    assert not np.any(np.isnan(strengths))
    if approximate:
        log_likelihoods = strengths
    else:
        exp_strengths = np.exp(-strengths)
        assert not np.any(np.isnan(exp_strengths))

        infs = np.isinf(exp_strengths)
        not_infs = np.logical_not(infs)

        if np.any(infs):
            log_likelihoods = np.empty((len(reward), len(diffs)))

            # If np.exp(...) is inf, then 1 + np.exp(...) is approximately np.exp(...)
            # so log1p(exp(-reward @ diffs))) \approx rewards @ diffs
            log_likelihoods[infs] = strengths[infs]
            log_likelihoods[not_infs] = -np.log1p(
                exp_strengths[not_infs], dtype=np.float128
            )
            assert not np.any(np.isnan(log_likelihoods))
        else:
            log_likelihoods = -np.log1p(exp_strengths, dtype=np.float128)

    if np.any(np.isneginf(log_likelihoods)):
        logging.warning("Some reward-halfplane pairs have -inf log likelihood")
    if np.any(np.exp(log_likelihoods) == 0):
        logging.warning("Some reward-halfplane pairs have 0 likelihood")

    assert np.all(
        log_likelihoods <= 0
    ), f"Max log likelihood is {np.max(log_likelihoods)} not <=0"

    return log_likelihoods


def infogain(feature_diffs: np.ndarray, reward_posterior: np.ndarray) -> np.ndarray:
    """Computes the infogain of each question.

    Args:
        feature_diffs (np.ndarray): Difference between reward features of trajectories being compared in each question.
        reward_posterior (np.ndarray): Point based monte carlo estimate of the reward posterior.

    Returns:
        np.ndarray: Expected infogain of each question.
    """
    assert len(feature_diffs.shape) == 2
    assert len(reward_posterior.shape) == 2
    n_reward_samples = reward_posterior.shape[0]
    log_likelihoods = (
        boltzmann_likelihood(reward_posterior, feature_diffs),
        boltzmann_likelihood(reward_posterior, -feature_diffs),
    )
    normalized_log_likelihoods = tuple(
        log_normalize_logs(log_likelihoods[i]) for i in range(2)
    )
    infogains: Tuple[np.ndarray, ...] = tuple(
        np.sum(
            np.exp2(log_likelihoods[i])
            * (normalized_log_likelihoods[i] + np.log2(n_reward_samples)),
            axis=0,
        )
        for i in range(2)
    )

    return (infogains[0] + infogains[1]) / n_reward_samples


def successive_elimination(
    question_samples: np.ndarray,
    reward_samples: np.ndarray,
    n_out_questions: int,
    initial_questions=200,
) -> np.ndarray:
    """Selects a batch of questions to ask using the successive elimination algorithm from Bikiy's
    Batch Active Learning from Preferences paper.

    Args:
        question_samples (np.ndarray): Unique feature differences of trajectories being compared in each question.
        reward_samples (np.ndarray): Point based monte carlo estimate of the reward posterior.
        n_out_questions (int): Desired batch size.
        inital_questions (int, optional): Number of questions to greedily select in the first step. Defaults to 200.

    Returns:
        np.ndarray: Index array of selected questions.
    """
    assert n_out_questions >= 1
    assert question_samples.shape[0] >= n_out_questions
    infogains = infogain(question_samples, reward_samples)
    assert (
        infogains.shape[0] == question_samples.shape[0]
    ), f"Infogains and question_samples shapes do not match {infogains.shape} != {question_samples.shape}"
    indices = np.argpartition(infogains, initial_questions - 1)[:initial_questions]

    greedy_questions = question_samples[indices]

    dists = pairwise_distances(greedy_questions, metric="euclidean")

    # We only care about pairs of distinct questions.
    dists[np.where(np.eye(dists.shape[0]))] = np.inf

    while len(indices) > n_out_questions:
        mins = np.where(dists == np.min(dists))
        assert np.array_equal(
            np.sort(mins[0]), np.sort(mins[1])
        ), f"Minimum distances not symmetric. mins={mins}, dists={dists}"
        assert (
            len(mins[0]) >= 1
        ), f"There should be at least two minimums. mins={mins}, dists={dists}"
        delete_dist_index = mins[0][0]

        dists = np.delete(dists, delete_dist_index, axis=0)
        dists = np.delete(dists, delete_dist_index, axis=1)

        indices = np.delete(indices, delete_dist_index)

    return indices

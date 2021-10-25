import logging

import numpy as np


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
    if len(reward.shape) == 1:
        reward = reward.reshape(1, -1)
    assert len(diffs) > 0

    reward = reward.astype(np.float128)
    diffs = diffs.astype(np.float128)

    # This function assumes that the reward posterior is defined on the unit sphere by restricting
    # the given likelihood to exactly the sphere, rather than taking a quotient space (by projecting
    # the likelihood for all rewards on every ray to their unit length point. If I ever want to do
    # that instead, the likelihood is |w| * log(1/2 * (1 + exp(w @ diffs))) / (w @ diffs) in general
    # and (log(1/2) + log1p(exp(w @ diffs))) / (w @ diffs) in our case, as |w|=1.
    strengths = (reward @ diffs.T) / temperature
    if approximate:
        log_likelihoods = strengths
    else:
        exp_strengths = np.exp(-strengths)

        infs = np.isinf(exp_strengths)
        not_infs = np.logical_not(infs)

        if np.any(infs):
            log_likelihoods = np.empty((len(reward), len(diffs)))

            # If np.exp(...) is inf, then 1 + np.exp(...) is approximately np.exp(...)
            # so log1p(exp(-reward @ diffs))) \approx rewards @ diffs
            log_likelihoods[infs] = strengths[infs]
            log_likelihoods[not_infs] = -np.log1p(exp_strengths[not_infs], dtype=np.float128)
        else:
            log_likelihoods = -np.log1p(exp_strengths, dtype=np.float128)

    if np.any(np.isneginf(log_likelihoods)):
        logging.warning("Some reward-halfplane pairs have -inf log likelihood")
    if np.any(np.exp(log_likelihoods) == 0):
        logging.warning("Some reward-halfplane pairs have 0 likelihood")

    assert np.all(log_likelihoods <= 0)

    return log_likelihoods

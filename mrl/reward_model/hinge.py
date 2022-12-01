import numpy as np

from mrl.reward_model.boltzmann import boltzmann_likelihood


def hinge_likelihood(
    reward: np.ndarray, diffs: np.ndarray, temperature: float = 1.0, approximate: bool = False
) -> np.ndarray:
    """Returns a hinge likelihood for a reward under each feature difference.

    We assume that if a reward agrees with the preference encoded by a diff, it's likelihood is 1.
    Otherwise, we use the Boltzmann-rational likelihood.

    Args:
        reward (np.ndarray): Reward or batch of rewards to determine likelihood of.
        diffs (np.ndarray): Differences between features of preferred and dispreffered objects.
        temperature (float): Temperature to use for Boltzmann-rational likelihood.
        approximate (bool): Whether to use approximate likelihoods.
    Returns:
        np.ndarray:  (Batch of) log proportional likelihoods of each reward under each halfplane.
    """

    log_likelihood = boltzmann_likelihood(reward, diffs, temperature, approximate)
    log_likelihood[log_likelihood > np.log(0.5)] = 0.0
    return log_likelihood

# Functions for working in log-space in order to avoid under- and overflows.
import logging

import numpy as np
from scipy.special import logsumexp  # type: ignore


def log_normalize_logs(x: np.ndarray) -> np.ndarray:
    """Uses the logsumexp trick to normalize log-space values across the 0th dimension.

    Args:
        x (np.ndarray): 2D Array of log-d values.

    Returns:
        np.ndarray: Array of log-d values such that np.sum(np.exp(out)) == 1
    """
    denom = logsumexp(x, axis=0)
    out = x - denom
    if np.any(np.isneginf(out)):
        logging.warning("Some normalized items have -inf log likelihood")
    if np.any(np.exp(out) == 0):
        logging.warning("Some normalized items have 0 likelihood")
    assert np.allclose(np.sum(np.exp(out), axis=0), 1)
    return out


def log_shift(logs: np.ndarray) -> np.ndarray:
    """Adds/subtracts to log values so no item over/underflows when exp'd.

    Args:
        logs (np.ndarray): An array of log-d values.

    Returns:
        np.ndarray: An array x such that np.exp(x) does not go out of bounds.
    """
    logs = logs.astype(np.float128)
    smallest_meaningful_log = np.log(np.finfo(np.float128).tiny)
    largest_meainingful_log = np.log(np.finfo(np.float128).max)
    max_log_shift = max(0, largest_meainingful_log - np.max(logs) - 100)
    ideal_log_shift = smallest_meaningful_log - np.min(logs) + 1
    log_shift = max(0, min(ideal_log_shift, max_log_shift))
    logging.info(f"ideal_log_shift={ideal_log_shift}, max_log_shift={max_log_shift}")
    logs += log_shift

    return logs


def cum_likelihoods(log_likelihoods: np.ndarray, shift: bool):
    """Computes the cumulative likelihoods in logspace across the 1-th dimension.

    We compute the cumulative likelihood, normalize using the logsumexp trick, and then try to add
    a value to the whole vector to make sure each item isn't under or overflowing. This means that
    the returned value is only a proportional likelihood, not a probability.

    Args:
        log_likelihoods (np.ndarray): 2D array of log-d values.

    Returns:
        np.ndarray: An array of cumulative likelihoods.
    """
    assert np.all(np.isfinite(log_likelihoods))

    log_total_likelihoods = np.cumsum(log_likelihoods, axis=1)
    assert np.all(np.isfinite(log_total_likelihoods))

    if np.any(np.isneginf(log_total_likelihoods)):
        logging.warning("Some cumulative terms have -inf log total likelihood")
    if np.any(np.exp(log_total_likelihoods) == 0):
        logging.warning("Some cumulative terms have 0 total unnormalized likelihood")

    log_total_likelihoods = log_normalize_logs(log_total_likelihoods)
    assert np.all(np.isfinite(log_total_likelihoods))

    if shift:
        log_total_likelihoods = log_shift(log_total_likelihoods)

    total_likelihoods = np.exp(log_total_likelihoods)

    if np.any(total_likelihoods == 0):
        logging.warning("Some cumulative terms have 0 total shifted likelihood")
    assert np.all(np.isfinite(total_likelihoods))
    return total_likelihoods

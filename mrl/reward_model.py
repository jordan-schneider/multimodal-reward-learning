import logging
import pickle as pkl
from pathlib import Path
from typing import List, Literal, Optional, Tuple, cast

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


# TODO: at some point should probably use something better than scipy, do we have a license for
# ibm's cplex solver?
def is_redundant_constraint(halfspace: np.ndarray, halfspaces: np.ndarray, epsilon=0.0001) -> bool:
    if len(halfspaces) == 0:
        return False

    # Let h be a halfspace constraint in the set of contraints H.
    # We have a constraint c^w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to
    # add c back into H.
    # Thus, we want to minimize c^T w subject to Hw >= 0.
    # First we need to change this into the form min c^T x subject to Ax <= b.
    # Our problem is equivalent to min c^T w subject to  -H w <= 0.

    b = np.zeros((len(halfspace),))
    lp_solution = linprog(
        halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1), method="revised simplex"
    )
    logging.debug(f"LP Solution={lp_solution}")
    if lp_solution["status"] != 0:
        logging.info("Revised simplex method failed. Trying interior point method.")
        lp_solution = linprog(halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1))

    if (
        lp_solution["status"] != 0
    ):  # Not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue.
        raise RuntimeError(f"Linear program failed with status {lp_solution['status']}")

    if lp_solution["fun"] < -epsilon:
        # If less than zero then constraint is needed to keep c^T w >=0
        return False
    else:
        # redundant since without constraint c^T w >=0
        logging.debug("Redundant")
        return True


def remove_redundant_constraints(
    normals: np.ndarray, precision: float = 0.0001
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a new array with all redundant halfspaces removed.

    Parameters
    -----------
    halfspaces : list of halfspace normal vectors such that np.dot(halfspaces[i], w) >= 0 for all i

    epsilon : numerical precision for determining if redundant via LP solution

    Returns
    -----------
    list of non-redundant halfspaces
    """
    # for each row in halfspaces, check if it is redundant

    non_redundant: List[np.ndarray] = []
    indices: List[int] = []

    for i, normal in enumerate(normals):
        logging.debug(f"Checking half space {normal}")

        halfspaces_lp = np.array(non_redundant + normals[i + 1 :])

        if not is_redundant_constraint(normal, halfspaces_lp, precision):
            logging.debug("Not redundant")
            non_redundant.append(normal)
            indices.append(i)
        else:
            logging.debug("Redundant")
    return np.array(non_redundant), np.array(indices)


def log_normalize_logs(x: np.ndarray) -> np.ndarray:
    logging.debug(f"min={np.min(x)}, max={np.max(x)}, ratio={np.max(x) - np.min(x)}")
    denom = logsumexp(x, axis=0)
    logging.debug(f"min denom={np.min(denom)}, max={np.max(denom)}")
    return x - denom


def reward_prop_likelihood_by_diff(
    reward: np.ndarray, diffs: np.ndarray, counts: Optional[np.ndarray] = None
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
    if counts is None:
        counts = np.ones(diffs.shape[0])
    assert len(diffs) > 0

    # This function assumes that the reward posterior is defined on the unit sphere by restricting
    # the given likelihood to exactly the sphere, rather than taking a quotient space (by projecting
    # the likelihood for all rewards on every ray to their unit length point. If I ever want to do
    # that instead, the likelihood is |w| * log(1/2 * (1 + exp(w @ diffs))) / (w @ diffs) in general
    # and (log(1/2) + log1p(exp(w @ diffs))) / (w @ diffs) in our case, as |w|=1.

    strengths = -reward @ diffs.T
    exp_strengths = np.exp(strengths)

    infs = np.isinf(exp_strengths)
    not_infs = np.logical_not(infs)

    log_likelihoods = np.empty((len(diffs), len(reward)))

    # If np.exp(...) is inf, then 1 + np.exp(...) is approximately np.exp(...)
    # so log1p(exp(-reward @ diffs))) \approx rewards @ diffs
    log_likelihoods[infs.T] = -strengths[infs].T
    log_likelihoods[not_infs.T] = -np.log1p(exp_strengths[not_infs])

    # Duplicate halfplanes result in likelihoods^count terms, which is multiplication in log space
    log_likelihoods = counts * log_likelihoods.T
    assert log_likelihoods.shape == (len(reward), len(diffs))

    return log_likelihoods


def mean_l2_dispersions(
    diffs: np.ndarray, counts: Optional[np.ndarray] = None, n_samples: int = 100_000
) -> np.ndarray:
    # Get reward sampled uniformly on the unit sphere
    reward_samples = np.random.standard_normal(size=(n_samples, diffs.shape[1]))
    reward_samples = (reward_samples.T / np.linalg.norm(reward_samples, axis=1)).T

    assert np.allclose(np.linalg.norm(reward_samples, axis=1), 1.0)

    log_likelihoods = reward_prop_likelihood_by_diff(reward_samples, diffs, counts)

    if np.any(np.isneginf(log_likelihoods)):
        logging.warning("Some counted halfplanes have -inf log likelihood")
    if np.any(np.exp(log_likelihoods) == 0):
        logging.warning("Some counted halfplanes have 0 likelihood")

    log_total_likelihoods = np.cumsum(log_likelihoods, axis=1)
    assert log_total_likelihoods.shape == (n_samples, len(diffs))

    if np.any(np.isneginf(log_total_likelihoods)):
        logging.warning("Some rewards have -inf log total likelihood")

    if np.any(np.exp(log_total_likelihoods) == 0):
        logging.warning("Some rewards have 0 total unnormalized likelihood")

    log_total_likelihoods = log_normalize_logs(log_total_likelihoods)

    smallest_meaningful_log = np.log(np.finfo(np.float64).tiny)
    largest_meainingful_log = np.log(np.finfo(np.float64).max)
    max_log_shift = largest_meainingful_log - np.max(log_total_likelihoods) - 100
    assert max_log_shift > 0
    ideal_log_shift = smallest_meaningful_log - np.min(log_total_likelihoods) + 1
    log_shift = max(0, min(ideal_log_shift, max_log_shift))
    logging.info(f"ideal_log_shift={ideal_log_shift}, max_log_shift={max_log_shift}")
    log_total_likelihoods += log_shift

    total_likelihoods = np.exp(log_total_likelihoods)
    assert total_likelihoods.shape == (n_samples, len(diffs))
    np.sum(total_likelihoods, axis=1)

    if np.any(total_likelihoods == 0):
        logging.warning("Some rewards have 0 total likelihood")
    assert np.all(np.isfinite(total_likelihoods))

    bad_timesteps = np.sum(total_likelihoods, axis=0) == 0
    if np.any(bad_timesteps):
        logging.warning("Some timesteps have 0 total likelihood for all rewards")
        last_good_timestep = np.argmax(bad_timesteps).item()
    else:
        last_good_timestep = len(diffs)

    logging.info(f"last_good_timestep={last_good_timestep}")

    mean_rewards = np.stack(
        np.average(reward_samples, weights=total_likelihoods[:, i], axis=0)
        for i in range(last_good_timestep)
    ).T
    assert mean_rewards.shape == (
        reward_samples.shape[1],
        last_good_timestep,
    ), f"mean_rewards shape={mean_rewards.shape}, expected {(reward_samples.shape[1], last_good_timestep)}"
    assert np.all(np.isfinite(mean_rewards))
    mean_rewards = mean_rewards / np.linalg.norm(mean_rewards, axis=0)
    assert np.all(np.isfinite(mean_rewards))

    lengths = np.linalg.norm(mean_rewards, axis=0)
    if not np.allclose(lengths, 1.0):
        error = np.abs(lengths - 1.0)
        worst_error = np.max(error)
        worst_index = np.argmax(error)
        logging.error(
            f"mean reward vector with length={lengths[worst_index]} error={worst_error}, mean reward={mean_rewards[worst_index]}"
        )
        exit()
    assert mean_rewards.shape == (reward_samples.shape[1], last_good_timestep)

    # Arc length is angle times radius, and the radius is 1, so the arc length between the mean
    # reward and each sample is just the angle between them, which you can get using the standard
    # cos(theta) = a @ b / (|a| * |b|) trick, and the norm of all vectors is 1.
    # einsum magic is just taking dot product row-wise
    dots = reward_samples @ mean_rewards
    dots[dots < -1.0] = -1.0
    dots[dots > 1.0] = 1.0

    dists = np.arccos(dots)
    assert dists.shape == (
        n_samples,
        last_good_timestep,
    ), f"dists shape={dists.shape}, expected {(n_samples, last_good_timestep,)}"

    assert not np.any(np.all(dists == 0.0, axis=0))

    weighted_dists = np.zeros((len(diffs),))
    for i in range(last_good_timestep):
        # TODO: There might be a less terrible way to do this, but np.average can't handle it
        weighted_dists[i] = np.average(dists[:, i], weights=total_likelihoods[:, i], axis=0)

    if logging.getLogger().isEnabledFor(logging.DEBUG):
        exit()

    zero_dispersion_timesteps = weighted_dists == 0.0
    nonzero_counts_by_timestep = np.sum(total_likelihoods > 0.0, axis=0)

    if not np.all(nonzero_counts_by_timestep[zero_dispersion_timesteps] == 1):
        logging.warning("Some zero dispersion timesteps have multiple nonzero likelihoods")
        logging.warning(nonzero_counts_by_timestep[zero_dispersion_timesteps])

    return weighted_dists


def plot_dispersion(
    in_path: Path, outdir: Path, n_samples: int = 1000, verbosity: Literal["INFO", "DEBUG"] = "INFO"
) -> None:
    logging.basicConfig(level=verbosity)

    in_path, outdir = Path(in_path), Path(outdir)
    diffs = np.load(in_path)[:1000]
    dispersion = mean_l2_dispersions(diffs, n_samples=n_samples)

    np.save(outdir / "dispersion.npy", dispersion)

    plt.plot(dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / "dispersion.png")
    plt.close()

    log_dispersion = np.log(dispersion)
    log_dispersion[log_dispersion == -np.inf] = None
    plt.plot(log_dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Log-mean dispersion")
    plt.title("Log-concentration of posterior with data")
    plt.savefig(outdir / "log_dispersion.png")
    plt.close()


def compare_modalities(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    action_path: Optional[Path] = None,
    n_samples: int = 1000,
) -> None:
    outdir = Path(outdir)
    paths = {}
    if traj_path is not None:
        paths["traj"] = Path(traj_path)
    if state_path is not None:
        paths["state"] = Path(state_path)
    if action_path is not None:
        paths["action"] = Path(action_path)

    diffs = {key: np.load(in_path) for key, in_path in paths.items()}
    dispersions = {
        key: mean_l2_dispersions(diff, n_samples=n_samples) for key, diff in diffs.items()
    }
    pkl.dump(dispersions, (outdir / "dispersions.pkl").open("wb"))
    for name, dispersion in dispersions.items():
        plt.plot(dispersion, label=name)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior for different modalities")
    plt.legend()
    plt.savefig(outdir / "comparison.png")
    plt.close()


def plot_joint_data(
    outdir: Path,
    traj_path: Optional[Path] = None,
    state_path: Optional[Path] = None,
    action_path: Optional[Path] = None,
    n_samples: int = 1000,
) -> None:
    outdir = Path(outdir)

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

    diffs = np.concatenate(pkl.load(path.open("rb")).numpy() for path in paths)
    dispersion = mean_l2_dispersions(diffs, n_samples=n_samples)
    np.save(outdir / "dispersion.npy", dispersion)
    plt.plot(dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / outname)
    plt.close()


if __name__ == "__main__":
    fire.Fire({"plot_dispersion": plot_dispersion, "compare": compare_modalities})

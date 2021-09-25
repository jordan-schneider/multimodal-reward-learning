import logging
import pickle as pkl
from pathlib import Path
from typing import List, Optional, Tuple, cast

import fire  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
from numpy.lib.function_base import disp
from scipy.optimize import linprog  # type: ignore
from scipy.spatial.distance import cosine  # type: ignore
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


def reward_prop_likelihood(
    reward: np.ndarray, diffs: np.ndarray, counts: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return the proportional likelihood of a reward given a set of reward feature differences.

    Args:
        reward (np.ndarray): Reward or batch of rewards to determine likelihood of.
        diffs (np.ndarray): Differences between features of preferred and dispreffered objects.
        weights (np.ndarray, optional): How many copies of each difference vector are present. Defaults to None, in which case all counts are assumed to be 1.

    Returns:
        np.ndarray: (Batch of) proportional likelihoods of each reward.
    """
    single_reward = len(reward.shape) == 1
    if not single_reward and counts is None:
        counts = np.ones(diffs.shape[0])
    assert len(diffs) > 0

    # Do final product over likelihood in log space for numerical stability
    # (n_diffs, n_rewards)
    likelihoods = -np.log1p(np.exp(-reward @ diffs.T)).T
    if not single_reward:
        assert likelihoods.shape == (len(diffs), len(reward))
        total_likelihoods = np.exp(np.sum((counts * likelihoods.T), axis=1))
        assert total_likelihoods.shape == (len(reward),)
    else:
        assert likelihoods.shape == (len(diffs),)
        total_likelihoods = np.exp(np.sum(likelihoods))
        assert total_likelihoods.shape == ()

    return total_likelihoods


def mean_l2_dispersion(
    diffs: np.ndarray, counts: Optional[np.ndarray] = None, n_samples: int = 100_000
) -> float:
    # Get reward sampled uniformly on the unit sphere
    reward_samples = np.random.standard_normal(size=(n_samples, diffs.shape[1]))
    reward_samples = (reward_samples.T / np.linalg.norm(reward_samples, axis=1)).T

    likelihoods = reward_prop_likelihood(reward_samples, diffs, counts=counts)
    mean_reward = np.average(reward_samples, weights=likelihoods, axis=0)
    mean_reward = mean_reward / np.linalg.norm(mean_reward)

    # Arc length is angle times radius, and the radius is 1, so the arc length between the mean
    # reward and each sample is just the angle between them, which you can get using the standard
    # cos(theta) = a @ b / (|a| * |b|) trick, and the norm of all vectors is 1.
    dists = np.arccos(mean_reward @ reward_samples.T)
    return np.average(dists, weights=likelihoods)


def get_dispersions(diffs: np.ndarray, n_samples: int) -> np.ndarray:
    dispersion = np.empty((len(diffs),))
    for i in tqdm(range(len(diffs))):
        dispersion[i] = mean_l2_dispersion(diffs[: i + 1], n_samples=n_samples)
    return dispersion


def plot_dispersion(in_path: Path, outdir: Path, n_samples: int = 1000) -> None:
    in_path, outdir = Path(in_path), Path(outdir)
    diffs = np.load(in_path)
    dispersion = get_dispersions(diffs, n_samples)

    np.save(outdir / "dispersion.npy", dispersion)
    plt.plot(dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / "dispersion.png")
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
    dispersions = {key: get_dispersions(diff, n_samples) for key, diff in diffs.items()}
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
    dispersion = get_dispersions(diffs, n_samples)
    np.save(outdir / "dispersion.npy", dispersion)
    plt.plot(dispersion)
    plt.xlabel("Human preferences")
    plt.ylabel("Mean dispersion")
    plt.title("Concentration of posterior with data")
    plt.savefig(outdir / outname)
    plt.close()


if __name__ == "__main__":
    fire.Fire({"plot_dispersion": plot_dispersion, "compare": compare_modalities})

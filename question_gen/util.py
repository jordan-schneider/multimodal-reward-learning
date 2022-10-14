import logging
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
from scipy.optimize import linprog  # type: ignore


def setup_logging(
    level: Literal["INFO", "DEBUG"],
    outdir: Optional[Path] = None,
    name: str = "log.txt",
    multiple_files: bool = True,
    force: bool = False,
    append: bool = False,
) -> None:
    FORMAT = "%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s"

    logging.basicConfig(level=level, format=FORMAT, force=force)
    if outdir is not None:
        logger = logging.getLogger()
        files = [
            handler.baseFilename
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        path = str(outdir / name)
        if multiple_files and path not in files:
            fh = logging.FileHandler(filename=path, mode="a" if append else "w")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(FORMAT))
            logging.getLogger().addHandler(fh)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("scipy").setLevel(logging.WARNING)


def is_redundant(
    halfspace: np.ndarray, halfspaces: np.ndarray, epsilon: float = 1e-4
) -> bool:
    # Let h be a halfspace constraint in the set of contraints H.
    # We have a constraint c^T w >= 0 we want to see if we can minimize c^T w and get it to go below 0
    # if not then this constraint is satisfied by the constraints in H, if we can, then we need to
    # add c back into H.
    # Thus, we want to minimize c^T w subject to Hw >= 0.
    # First we need to change this into the form min c^T x subject to Ax <= b.
    # Our problem is equivalent to min c^T w subject to  -H w <= 0.
    if np.any(np.linalg.norm(halfspaces - halfspace) < epsilon):
        return True

    m, _ = halfspaces.shape

    b = np.zeros(m)
    solution = linprog(
        halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1), method="revised simplex"
    )
    if solution["status"] != 0:
        logging.info("Revised simplex method failed. Trying interior point method.")
        solution = linprog(halfspace, A_ub=-halfspaces, b_ub=b, bounds=(-1, 1))

    if solution["status"] != 0:
        # Not sure what to do here. Shouldn't ever be infeasible, so probably a numerical issue.
        raise Exception("LP NOT SOLVABLE")
    elif solution["fun"] < -epsilon:
        # If less than zero then constraint is needed to keep c^T w >=0
        return False
    else:
        # redundant since without constraint c^T w >=0
        return True


def remove_redundant(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    nonredundant = vecs[0].reshape(1, -1)
    indices = [0]
    for i, vec in enumerate(vecs[1:]):
        if not is_redundant(vec, nonredundant):
            nonredundant = np.vstack((nonredundant, vec))
            indices.append(i + 1)
    return nonredundant, np.array(indices)


def remove_zeros(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.where(np.any(vecs != 0.0, axis=1))[0]
    return vecs[indices], indices


def remove_duplicates(
    vecs: np.ndarray, sensitivity: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    unique = vecs[0].reshape(1, -1)
    indices = [0]
    for i, vec in enumerate(vecs[1:]):
        if not np.any(np.linalg.norm(unique - vec, axis=1) <= sensitivity):
            unique = np.vstack((unique, vec))
            indices.append(i + 1)
    return unique, np.array(indices)


def orient(reward: np.ndarray, halfspaces: np.ndarray) -> np.ndarray:
    return (halfspaces.T * ((halfspaces @ reward < 0) * -1)).T

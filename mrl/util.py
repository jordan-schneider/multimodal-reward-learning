import collections
import gc
import logging
import pickle as pkl
import re
from pathlib import Path
from typing import Any, Literal, Optional, OrderedDict, Sequence, Tuple, cast

import numpy as np
import psutil  # type: ignore
import torch
from GPUtil import GPUtil  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from phasic_policy_gradient.train import make_model
from procgen import ProcgenGym3Env
from scipy.optimize import linprog  # type: ignore

from mrl.dataset.random_policy import RandomPolicy


def get_temp_from_pref_path(path: Path) -> float:
    parts = path.parts
    prefs_index = parts.index("prefs")
    temp_index = prefs_index + 2  # prefs/modality/temp/...
    return float(parts[temp_index])


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(
        np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
            a_min=-1.0,
            a_max=1.0,
        )
    )


NORM_DIFF_MODES = Literal[
    "diff-length", "sum-length", "max-length", "log-diff-length", None
]


def get_normalized_diff(
    features: np.ndarray,
    mode: NORM_DIFF_MODES = None,
) -> np.ndarray:
    assert len(features.shape) == 3
    assert features.shape[1] == 2
    diffs = features[:, 0] - features[:, 1]
    if mode == "diff-length":
        diffs /= np.linalg.norm(diffs, axis=1, keepdims=True)
    elif mode == "sum-length":
        feature_lengths = np.linalg.norm(features, axis=2, keepdims=True)
        diffs /= feature_lengths[:, 0] + feature_lengths[:, 1]
    elif mode == "max-length":
        feature_lengths = np.linalg.norm(features, axis=2, keepdims=True)
        diffs /= np.max(feature_lengths, axis=1)
    elif mode == "log-diff-length":
        diffs /= np.log2(np.linalg.norm(diffs, axis=1, keepdims=True) + 1)
    return diffs


def normalize_vecs(x: np.ndarray) -> np.ndarray:
    """Normalizes the vectors in an array on the 1th axis.

    Args:
        x (np.ndarray): 2D array of vectors.

    Returns:
        np.ndarray: 2D array x such that np.linalg.norm(x, axis=1) == 1
    """
    shape = x.shape
    out: np.ndarray = (x.T / np.linalg.norm(x, axis=1)).T
    assert out.shape == shape, f"shape: expected={shape} actual={out.shape}"
    assert np.allclose(
        np.linalg.norm(out, axis=1), 1
    ), f"norm: expected={1} actual={np.linalg.norm(out, axis=1)}"
    return out


def batch(obs: torch.Tensor, obs_dims: int) -> torch.Tensor:
    if len(obs.shape) == obs_dims:
        return obs.reshape((1, *obs.shape))
    elif len(obs.shape) == obs_dims + 1:
        return obs
    else:
        raise ValueError(
            f"Expected obersvation to have {obs_dims} or {obs_dims + 1} dimensions, but has shape {obs.shape}"
        )


def set_seed(seed: int) -> None:
    "Sets numpy and torch random seeds. Use np.random.default_rng() for finer control of numpy randomness."
    np.random.seed(seed)
    torch.manual_seed(seed)


def dump(obj: Any, path: Path) -> None:
    def is_torch(obj):
        torch_classes = (torch.Tensor, torch.nn.Module)
        if isinstance(obj, collections.abc.Mapping):
            v = list(obj.values())[0]
            return is_torch(v)
        if isinstance(obj, collections.Sequence):
            return is_torch(obj[0])
        return isinstance(obj, torch_classes)

    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif is_torch(obj):
        torch.save(obj, path.with_suffix(".pt"))
    else:
        pkl.dump(obj, path.with_suffix(".pkl").open("wb"))


def load(path: Path) -> Any:
    if path.suffix in (".npy", ".npz"):
        return np.load(path)
    elif path.suffix == ".pt":
        return torch.load(path)
    elif path.suffix == ".pkl":
        return pkl.load(path.open("rb"))
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")


def reinit(n: torch.nn.Module) -> None:
    def _init(m):
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    n.apply(_init)


def max_batch_size(
    modality: Literal["state", "traj"],
    prefs: int,
    n_parallel_envs: int,
    step_nbytes: int,
) -> int:
    if modality == "state":
        return max_state_batch_size(
            n_states=prefs,
            n_parallel_envs=n_parallel_envs,
            step_nbytes=step_nbytes,
        )
    elif modality == "traj":
        return max_traj_batch_size(
            n_trajs=prefs,
            n_parallel_envs=n_parallel_envs,
            step_nbytes=step_nbytes,
        )
    else:
        raise ValueError(f"Modality {modality} must be 'state' or 'traj'")


def max_state_batch_size(n_states: int, n_parallel_envs: int, step_nbytes: int) -> int:
    gc.collect()
    free_memory = psutil.virtual_memory().available
    logging.info(f"{free_memory=}")

    batch_timesteps = min(
        n_states // n_parallel_envs, int(free_memory / step_nbytes * 0.8)
    )
    batch_timesteps = max(batch_timesteps, 2)
    logging.info(f"{batch_timesteps=}")

    return batch_timesteps


def max_traj_batch_size(n_trajs: int, n_parallel_envs: int, step_nbytes: int) -> int:
    gc.collect()
    free_memory = psutil.virtual_memory().available
    logging.info(f"{free_memory=}")

    steps_to_done = (n_trajs * 1000 + 1) // n_parallel_envs
    steps_that_will_fit = int(free_memory / step_nbytes * 0.8)

    batch_timesteps = min(steps_to_done, steps_that_will_fit)
    batch_timesteps = max(batch_timesteps, 1001)
    logging.info(f"{batch_timesteps=}")
    return batch_timesteps


def get_policy(
    path: Optional[Path],
    env: ProcgenGym3Env,
    device: torch.device = torch.device("cpu"),
) -> PhasicValueModel:
    if path is not None:
        state_dict = cast(
            OrderedDict[str, torch.Tensor], torch.load(path, map_location=device)
        )
        policy = make_model(env, arch="shared")
        policy.load_state_dict(state_dict)
        policy.to(device=device)
        return policy
    else:
        return RandomPolicy(actype=env.ac_space, num=env.num)


def find_policy_path(
    policydir: Path, overwrite: bool = False
) -> Tuple[Optional[Path], int]:
    models = list(policydir.glob("model*.jd"))
    if len(models) == 0 or overwrite:
        return None, 0

    latest_model = sorted(models)[-1]
    match = re.search("([0-9]+).jd", str(latest_model))
    assert match is not None
    model_iter = int(match.group(1))
    return latest_model, model_iter


def find_best_gpu() -> torch.device:
    device_id = GPUtil.getFirstAvailable(order="load")[0]
    return torch.device(f"cuda:{device_id}")


def np_remove(indir: Path, name: str) -> None:
    logging.debug(f"Removing {indir=}, {name=}")
    paths = [
        path
        for path in indir.iterdir()
        if path.is_file()
        and re.search(
            f"/{name}(\.flip-probs|\.features)(\.[0-9]+)?(\.npy|\.png)", str(path)
        )
    ]
    logging.info(f"Removing {paths}")
    for path in paths:
        path.unlink()


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
    logging.debug(f"LP Solution={solution}")
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
        logging.debug("Redundant")
        return True

def assert_many_allclose(vecs: Sequence[np.ndarray], rtol=1.e-5, atol=1.e-8, equal_nan=False) -> None:
    first = vecs[0]
    for vec in vecs[1:]:
        assert np.allclose(first, vec, rtol, atol, equal_nan)

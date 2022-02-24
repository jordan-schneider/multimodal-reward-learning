import collections
import gc
import logging
import pickle as pkl
import re
from pathlib import Path
from typing import (Any, Callable, Dict, List, Literal, Optional, OrderedDict,
                    Sequence, Tuple, Union, cast)

import numpy as np
import psutil  # type: ignore
import torch
from GPUtil import GPUtil  # type: ignore
from phasic_policy_gradient.ppg import PhasicValueModel
from phasic_policy_gradient.train import make_model
from procgen import ProcgenGym3Env
from scipy.optimize import linprog  # type: ignore
from tqdm import trange  # type: ignore

from mrl.dataset.offline_buffer import RlDataset
from mrl.dataset.random_policy import RandomPolicy
from mrl.envs.feature_envs import FeatureEnv
from mrl.envs.util import get_root_env
from mrl.memprof import get_memory


def get_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.arccos(
        np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
            a_min=-1.0,
            a_max=1.0,
        )
    )


def normalize_diffs(
    features: np.ndarray,
    mode: Literal[
        "diff-length", "sum-length", "max-length", "log-diff-length", None
    ] = None,
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


def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    tqdm: bool = False,
    check_occupancies: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_shape = env.ob_space.shape

    states = np.empty((timesteps, env.num, *state_shape))
    actions = np.empty((timesteps - 1, env.num), dtype=np.int64)
    rewards = np.empty((timesteps, env.num))
    firsts = np.empty((timesteps, env.num), dtype=bool)

    if check_occupancies:
        timesteps_written_to: Dict[int, int] = {}

    def record(t: int, env: ProcgenGym3Env, states, rewards, firsts) -> None:
        if check_occupancies:
            timesteps_written_to[t] = timesteps_written_to.get(t, 0) + 1
        reward, state, first = env.observe()
        state = cast(
            np.ndarray, state
        )  # env.observe typing dones't account for wrapper
        states[t] = state
        rewards[t] = reward
        firsts[t] = first

    times = trange(timesteps - 1) if tqdm else range(timesteps - 1)

    for t in times:
        record(t, env, states, rewards, firsts)
        state = torch.tensor(states[t], device=policy.device)
        first = firsts[t]
        init_state = policy.initial_state(env.num)
        action, _, _ = policy.act(state, first, init_state)
        action = action.cpu().numpy()
        env.act(action)
        actions[t] = action
    record(timesteps - 1, env, states, rewards, firsts)

    if check_occupancies:
        assert timesteps_written_to == {key: 1 for key in range(timesteps)}

    return states, actions, rewards, firsts


class ArrayOrList:
    def __init__(
        self, val: Union[np.ndarray, List[np.ndarray]], max_list_len: int = 1000
    ) -> None:
        self.val = val
        self.list = isinstance(val, list)
        self.max_list_len = max_list_len
        self.make_array_on_set = False
        if self.list:
            self.offset = 0
            if len(self.val) > 0:
                self.array = np.empty((0, *self.val[0].shape))
            else:
                self.make_array_on_set = True

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        if self.list:
            assert isinstance(self.val, list)
            effective_key = key - self.offset
            if effective_key == len(self.val):
                if self.make_array_on_set:
                    self.make_array_on_set = False
                    self.array = np.empty((0, *value.shape))
                if len(self.val) > self.max_list_len:
                    self.coalesce()
                self.val.append(value)
            elif effective_key > len(self.val):
                raise IndexError(f"Index {key} is out of range")
            elif effective_key < 0:
                self.array[key] = value
            else:
                self.val[effective_key] = value
        else:
            self.val[key] = value

    def coalesce(self) -> None:
        if self.list:
            self.array = np.concatenate((self.array, np.array(self.val)))
            self.val = []
            self.offset = self.array.shape[0]

    def __getitem__(self, key: Union[int, slice]) -> Any:
        if self.list:
            self.coalesce()
            self.array[key]
        else:
            return self.val[key]

    def numpy(self) -> np.ndarray:
        if self.list:
            self.coalesce()
            return self.array
        else:
            assert isinstance(self.val, np.ndarray)
            return self.val


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


def procgen_rollout_features(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: Optional[int] = None,
    n_trajs: Optional[int] = None,
    tqdm: bool = False,
) -> np.ndarray:
    root_env = get_root_env(env)
    assert isinstance(root_env, FeatureEnv)

    features = ArrayOrList(
        np.empty((timesteps, env.num, root_env._reward_weights.shape[0]))
        if timesteps is not None
        else []
    )

    def step(t: int):
        _, state, first = env.observe()
        features[t] = root_env.features
        action, _, _ = policy.act(
            torch.tensor(state, device=policy.device),
            torch.tensor(first, device=policy.device),
            policy.initial_state(env.num),
        )
        env.act(action.cpu().numpy())
        return state, action, first

    if n_trajs is not None:
        cur_trajs = 0
        t = 0
        while cur_trajs < n_trajs and (timesteps is None or t < timesteps):
            _, _, first = step(t)
            cur_trajs += np.sum(first)
            t += 1
        features[t] = root_env.features
    elif timesteps is not None:
        times = trange(timesteps - 1) if tqdm else range(timesteps - 1)
        for t in times:
            step(t)
        features[timesteps - 1] = root_env.features
    else:
        raise ValueError("Must specify either timesteps or n_trajs")
    return features.numpy()


class DatasetRoller:
    def __init__(
        self,
        env: ProcgenGym3Env,
        policy: PhasicValueModel,
        timesteps: int = -1,
        n_trajs: Optional[int] = None,
        flags: Sequence[Literal["state", "action", "reward", "first", "feature"]] = [
            "state",
            "action",
            "reward",
            "first",
            "feature",
        ],
        tqdm: bool = False,
    ):
        self.env = env
        self.root_env = cast(FeatureEnv, get_root_env(env))
        assert isinstance(self.root_env, FeatureEnv)

        self.policy = policy

        self.timesteps = timesteps
        if self.timesteps == 0 or self.timesteps < -1:
            raise ValueError("timesteps must be positive or -1")
        self.run_until_done = self.timesteps == -1

        self.n_trajs = n_trajs
        if n_trajs is not None and n_trajs <= 0:
            raise ValueError("n_trajs must be positive")
        if self.run_until_done and self.n_trajs is None:
            raise ValueError("Must specify n_trajs if timesteps is -1")

        self.fixed_size_arrays = timesteps > -1 and n_trajs is None

        self.flags = flags
        self.tqdm = tqdm

        state_shape = env.ob_space.shape

        self.states = self.make_array((timesteps + 1, env.num, *state_shape), "state")
        self.actions = self.make_array((timesteps, env.num), "action", dtype=np.uint8)
        self.rewards = self.make_array((timesteps + 1, env.num), "reward")
        self.firsts = self.make_array((timesteps + 1, env.num), "first", dtype=bool)
        self.features = self.make_array(
            (timesteps + 1, env.num, self.root_env.n_features), "feature"
        )

    def make_array(
        self, shape: Tuple[int, ...], name: str, dtype=np.float32
    ) -> Optional[ArrayOrList]:
        if name in self.flags:
            return ArrayOrList(
                np.empty(shape, dtype=dtype) if self.fixed_size_arrays else []
            )
        else:
            return None

    def record(
        self,
        t: int,
        feature: Optional[np.ndarray],
        state: Optional[np.ndarray],
        reward: Optional[np.ndarray],
        first: Optional[np.ndarray],
    ) -> None:
        if self.states is not None and state is not None:
            self.states[t] = state
        if self.rewards is not None and reward is not None:
            self.rewards[t] = reward
        if self.firsts is not None and first is not None:
            self.firsts[t] = first
        if self.features is not None and feature is not None:
            self.features[t] = feature

    def step(self, t: int):
        with torch.no_grad():
            reward, state, first = self.env.observe()
            self.record(t, self.root_env.features, state, reward, first)
            state_tensor = torch.tensor(state)
            first_tensor = torch.tensor(first)
            del reward, state
            init_state = self.policy.initial_state(self.env.num)
            action, _, _ = self.policy.act(state_tensor, first_tensor, init_state)
            del state_tensor, first_tensor, init_state
            action = action.cpu().numpy()
            self.env.act(action)
            if self.actions is not None:
                self.actions[t] = action
            return first

    def roll(self) -> RlDataset:
        if self.n_trajs is not None:
            self.roll_n_trajs()
        elif self.timesteps > 0:
            self.roll_timesteps()
        else:
            raise ValueError("Must speficy n_trajs if timesteps=-1")

        states_arr, actions_arr, rewards_arr, firsts_arr, features_arr = (
            arr.numpy() if arr is not None else None
            for arr in (
                self.states,
                self.actions,
                self.rewards,
                self.firsts,
                self.features,
            )
        )

        return RlDataset.from_gym3(
            states=states_arr,
            actions=actions_arr,
            rewards=rewards_arr,
            firsts=firsts_arr,
            features=features_arr,
        )

    def roll_timesteps(self):
        times = trange(self.timesteps) if self.tqdm else range(self.timesteps - 1)

        for t in times:
            self.step(t)

        reward, state, first = self.env.observe()
        self.record(self.timesteps, self.root_env.features, state, reward, first)

    def roll_n_trajs(self, gc_period: int = 500):
        assert self.n_trajs is not None
        cur_trajs = -self.env.num  # -1 per env because the first step has first=True
        t = 0
        while cur_trajs < self.n_trajs and (self.run_until_done or t < self.timesteps):
            if t % gc_period == 0:
                gc.collect()
                logging.debug(
                    f"step {t} vm={get_memory()['VmSize']}, peak={get_memory()['VmPeak']}"
                )
            first = self.step(t)

            cur_trajs += np.sum(first)
            del first
            t += 1

        if t == self.timesteps:
            reward, state, first = self.env.observe()
            self.record(self.timesteps, self.root_env.features, state, reward, first)


def procgen_rollout_dataset(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int = -1,
    n_trajs: Optional[int] = None,
    flags: Sequence[Literal["state", "action", "reward", "first", "feature"]] = [
        "state",
        "action",
        "reward",
        "first",
        "feature",
    ],
    tqdm: bool = False,
) -> RlDataset:
    roller = DatasetRoller(
        env=env,
        policy=policy,
        timesteps=timesteps,
        n_trajs=n_trajs,
        flags=flags,
        tqdm=tqdm,
    )
    return roller.roll()


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

    # How many timesteps can we fit into the available memory?
    batch_timesteps = min(
        (n_trajs * 1000 + 1) // n_parallel_envs, int(free_memory / step_nbytes * 0.8)
    )
    batch_timesteps = max(batch_timesteps, 2)
    logging.info(f"{batch_timesteps=}")
    return batch_timesteps


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


def np_gather(
    indir: Path,
    name: str,
    max_nbytes: int = -1,
) -> np.ndarray:
    paths = [
        path
        for path in indir.iterdir()
        if path.is_file() and re.search(f"/{name}(.[0-9]+)?.npy", str(path))
    ]
    logging.info(f"Gathering from {paths}")
    if len(paths) == 0:
        raise FileNotFoundError(f"No {name} files found in {indir}")
    shards = []

    nbytes = 0
    while len(paths) > 0 and (max_nbytes == -1 or nbytes < max_nbytes):
        path = paths.pop()
        array: np.ndarray = np.load(path)
        rows_only = array.reshape((array.shape[0], -1))
        finite_rows = np.all(np.isfinite(rows_only), axis=1)
        nonzero_rows = np.any(rows_only != 0, axis=1)
        if not np.all(finite_rows) or not np.all(nonzero_rows):
            array = array[finite_rows & nonzero_rows]
            np.save(path, array)
        nbytes += array.nbytes
        shards.append(array)
    data = np.concatenate(shards)
    logging.info(f"Loaded array with shape {data.shape} from {indir}")
    return data


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
) -> None:
    FORMAT = "%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s"

    logging.basicConfig(level=level, format=FORMAT)
    if outdir is not None:
        logger = logging.getLogger()
        files = [
            handler.baseFilename
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        ]
        path = str(outdir / name)
        if multiple_files and path not in files:
            fh = logging.FileHandler(filename=path, mode="w")
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

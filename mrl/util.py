from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from GPUtil import GPUtil  # type: ignore
from gym3 import ExtractDictObWrapper
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from tqdm import trange  # type: ignore

from mrl.envs import Miner
from mrl.envs.probe_envs import OneActionNoObsOneTimestepOneReward as Probe
from mrl.offline_buffer import RlDataset


def reinit(n: torch.nn.Module) -> None:
    def _init(m):
        if hasattr(m, "reset_parameters") and callable(m.reset_parameters):
            m.reset_parameters()

    n.apply(_init)


def make_env(name: Literal["miner", "miner_reward", "probe"], num: int, **kwargs) -> ProcgenGym3Env:
    if name == "miner":
        env = ProcgenGym3Env(num=1, env_name="miner")
    elif name == "miner_reward":
        assert "reward_weights" in kwargs.keys(), "Must supply reward_weights to Miner reward env."
        env = Miner(num=num, **kwargs)
    elif name == "probe":
        env = Probe(num=num, **kwargs)

    env = ExtractDictObWrapper(env, "rgb")
    return env


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
        state = cast(np.ndarray, state)  # env.observe typing dones't account for wrapper
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
    def __init__(self, val: Union[np.ndarray, list]) -> None:
        self.val = val
        self.list = isinstance(val, list)

    def __setitem__(self, key: int, value: Any) -> None:
        if self.list:
            assert isinstance(self.val, list)
            if key == len(self.val):
                self.val.append(value)
            elif key > len(self.val):
                raise IndexError(f"Index {key} is out of range")
            else:
                self.val[key] = value
        else:
            self.val[key] = value

    def numpy(self) -> np.ndarray:
        if self.list:
            return np.array(self.val)
        else:
            assert isinstance(self.val, np.ndarray)
            return self.val


def procgen_rollout_features(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: Optional[int] = None,
    n_trajs: Optional[int] = None,
    tqdm: bool = False,
) -> np.ndarray:
    features = ArrayOrList(
        np.empty((timesteps, env.num, Miner.N_FEATURES)) if timesteps is not None else []
    )

    def step():
        _, state, first = env.observe()
        features[t] = env.callmethod("get_last_features")
        action, _, _ = policy.act(
            torch.tensor(state, device=policy.device), first, policy.initial_state(env.num)
        )
        env.act(action.cpu().numpy())
        return state, action, first

    if n_trajs is not None:
        cur_trajs = 0
        t = 0
        while cur_trajs < n_trajs and (timesteps is None or t < timesteps):
            _, _, first = step()
            cur_trajs += np.sum(first)
            t += 1
        features[t] = env.callmethod("get_last_features")
    elif timesteps is not None:
        times = trange(timesteps - 1) if tqdm else range(timesteps - 1)
        for t in times:
            step()
        features[timesteps - 1] = env.callmethod("get_last_features")
    else:
        raise ValueError("Must specify either timesteps or n_trajs")
    return features.numpy()


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
    state_shape = env.ob_space.shape

    def make_array(shape: Tuple[int, ...], name: str, dtype=np.float32) -> Optional[ArrayOrList]:
        if name in flags:
            return ArrayOrList(np.empty(shape, dtype=dtype) if timesteps > 0 else [])
        else:
            return None

    states = make_array((timesteps, env.num, *state_shape), "state")
    actions = make_array((timesteps - 1, env.num), "action", dtype=np.uint8)
    rewards = make_array((timesteps, env.num), "reward")
    firsts = make_array((timesteps, env.num), "first", dtype=bool)
    features = make_array((timesteps, env.num, Miner.N_FEATURES), "feature")

    def record(t: int, env: ProcgenGym3Env, state, reward, first) -> None:
        if states is not None:
            states[t] = state
        if rewards is not None:
            rewards[t] = reward
        if firsts is not None:
            firsts[t] = first
        if features is not None:
            features[t] = env.callmethod("get_last_features")

    def step():
        reward, state, first = env.observe()
        record(t, env, state, reward, first)
        action, _, _ = policy.act(
            torch.tensor(state, device=policy.device), first, policy.initial_state(env.num)
        )
        action = action.cpu().numpy()
        env.act(action)
        if actions is not None:
            actions[t] = action
        return state, action, reward, first

    if n_trajs is not None:
        cur_trajs = 0
        t = 0
        while cur_trajs < n_trajs and (timesteps < 0 or t < timesteps):
            state, _, reward, first = step()

            cur_trajs += np.sum(first)
            t += 1
        record(timesteps - 1, env, state, reward, first)
    elif timesteps > 0:
        times = trange(timesteps - 1) if tqdm else range(timesteps - 1)

        for t in times:
            step()

        reward, state, first = env.observe()
        record(timesteps - 1, env, state, reward, first)

    states_arr, actions_arr, rewards_arr, firsts_arr, features_arr = (
        arr.numpy() if arr is not None else None
        for arr in (states, actions, rewards, firsts, features)
    )

    return RlDataset.from_gym3(
        states=states_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        firsts=firsts_arr,
        features=features_arr,
    )


def find_policy_path(policydir: Path, overwrite: bool = False) -> Tuple[Optional[Path], int]:
    models = list(policydir.glob("model[0-9][0-9][0-9].jd"))
    if len(models) == 0 or overwrite:
        return None, 0

    latest_model = sorted(models)[-1]
    model_iter = int(str(latest_model)[-6:-3])
    return latest_model, model_iter


def find_best_gpu() -> torch.device:
    device_id = GPUtil.getFirstAvailable(order="load")[0]
    return torch.device(f"cuda:{device_id}")

from pathlib import Path
from typing import Optional, Tuple, cast, overload

import numpy as np
import torch
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from tqdm import trange  # type: ignore

from mrl.envs import Miner


@overload
def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    tqdm: bool,
    return_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    return_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    tqdm: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


@overload
def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ...


def procgen_rollout(
    env: ProcgenGym3Env,
    policy: PhasicValueModel,
    timesteps: int,
    *,
    tqdm: bool = False,
    return_features: bool = False,
):
    state_shape = env.ob_space.shape

    states = np.empty((timesteps, env.num, *state_shape))
    actions = np.empty((timesteps - 1, env.num), dtype=np.int64)
    rewards = np.empty((timesteps, env.num))
    firsts = np.empty((timesteps, env.num), dtype=bool)
    if return_features:
        features = np.empty((timesteps, env.num, Miner.N_FEATURES))

    def record(t: int, env: ProcgenGym3Env, states, rewards, firsts, features) -> None:
        reward, state, first = env.observe()
        state = cast(np.ndarray, state)  # env.observe typing dones't account for wrapper
        states[t] = state
        rewards[t] = reward
        firsts[t] = first
        if return_features:
            features[t] = env.callmethod("get_last_features")

    times = trange(timesteps - 1) if tqdm else range(timesteps - 1)

    for t in times:
        record(t, env, states, rewards, firsts, features)
        action, _, _ = policy.act(
            torch.tensor(states[t], device=policy.device), firsts[t], policy.initial_state(env.num)
        )
        action = action.cpu().numpy()
        env.act(action)
        actions[t] = action
    record(timesteps - 1, env, states, rewards, firsts, features)

    if return_features:
        return states, actions, rewards, firsts, features
    else:
        return states, actions, rewards, firsts


def find_policy_path(policydir: Path, overwrite: bool = False) -> Tuple[Optional[Path], int]:
    models = list(policydir.glob("model[0-9][0-9][0-9].jd"))
    if len(models) == 0 or overwrite:
        return None, 0

    latest_model = sorted(models)[-1]
    model_iter = int(str(latest_model)[-6:-3])
    return latest_model, model_iter

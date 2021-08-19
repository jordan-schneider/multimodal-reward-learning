from pathlib import Path
from typing import Optional, Tuple, cast

import numpy as np
import torch
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from tqdm import trange

from mrl.offline_buffer import RLDataset


def procgen_rollout(
    env: ProcgenGym3Env, policy: PhasicValueModel, timesteps: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state_shape = env.ob_space.shape

    states = np.empty((timesteps, env.num, *state_shape))
    actions = np.empty((timesteps - 1, env.num), dtype=np.int64)
    rewards = np.empty((timesteps, env.num))
    firsts = np.empty((timesteps, env.num), dtype=bool)

    def record(t: int, env: ProcgenGym3Env, states, rewards, firsts) -> None:
        reward, state, first = env.observe()
        state = cast(np.ndarray, state)  # env.observe typing dones't account for wrapper
        states[t] = state
        rewards[t] = reward
        firsts[t] = first

    for t in trange(timesteps - 1):
        record(t, env, states, rewards, firsts)
        action, _, _ = policy.act(
            torch.tensor(states[t], device=policy.device), firsts[t], policy.initial_state(env.num)
        )
        action = action.cpu().numpy()
        env.act(action)
        actions[t] = action
    record(timesteps - 1, env, states, rewards, firsts)

    return states, actions, rewards, firsts


def find_policy_path(policydir: Path) -> Tuple[Optional[Path], int]:
    models = policydir.glob("model[0-9][0-9][0-9].jd")
    if not models:
        return None, 0

    latest_model = sorted(models)[-1]
    model_iter = int(str(latest_model)[-6:-3])
    return latest_model, model_iter

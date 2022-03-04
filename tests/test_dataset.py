import logging
from functools import partial
from typing import Tuple

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans, floats, integers, just, shared, tuples
from mrl.dataset.offline_buffer import RlDataset, SarsDataset
from torch.testing import assert_allclose

assert_equal = partial(assert_allclose, atol=0, rtol=0)


def np2t(*args: np.ndarray) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(arg) for arg in args)


normal_floats = floats(allow_nan=False, allow_infinity=False)
traj_length = shared(integers(1, 100))

states_strategy = arrays(
    shape=tuples(traj_length, just(64), just(64), just(3)),
    elements=normal_floats,
    dtype=float,
)
actions_strategy = arrays(
    shape=traj_length, elements=integers(min_value=0, max_value=15), dtype=int
)
rewards_strategy = arrays(shape=traj_length, elements=normal_floats, dtype=float)
dones_strategy = arrays(shape=traj_length, elements=booleans(), dtype=bool).filter(
    lambda arr: not arr.all()
)


@given(
    states=states_strategy,
    actions=actions_strategy,
    rewards=rewards_strategy,
    firsts=dones_strategy,
)
def test_sars_dataset_index(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    firsts: np.ndarray,
) -> None:
    logging.basicConfig(level="DEBUG")
    firsts[0] = True
    dataset = SarsDataset(
        states=states, actions=actions, rewards=rewards, firsts=firsts
    )
    l = len(dataset)
    raw_index = 0
    for dataset_index in range(l):
        while firsts[raw_index + 1]:
            raw_index += 1
        actual_states, actual_actions, actual_rewards, actual_nextstates = dataset[
            dataset_index
        ]
        assert_equal(
            actual_states,
            states[raw_index],
            msg=f"states, raw_index={raw_index}, dataset_index={dataset_index}",
        )
        assert_equal(
            actual_actions,
            actions[raw_index],
            msg=f"actions, raw_index={raw_index}, dataset_index={dataset_index}",
        )
        assert_equal(
            actual_rewards,
            rewards[raw_index],
            msg=f"rewards, raw_index={raw_index}, dataset_index={dataset_index}",
        )
        assert_equal(
            actual_nextstates,
            states[raw_index + 1],
            msg=f"next_states, raw_index={raw_index}, dataset_index={dataset_index}",
        )

        raw_index += 1


@given(
    timesteps=integers(min_value=1, max_value=1000),
    n_envs=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_trajs_done_every_step(timesteps: int, n_envs: int) -> None:
    states = np.empty((timesteps, n_envs, 64, 64, 3))
    actions = np.empty((timesteps, n_envs), dtype=np.int8)
    rewards = np.empty((timesteps, n_envs))
    firsts = np.ones((timesteps, n_envs), dtype=bool)

    data = RlDataset.from_gym3(
        states=states,
        actions=actions,
        rewards=rewards,
        firsts=firsts,
        keep_incomplete=True,
    )
    n_trajs = 0
    for traj in data.trajs(include_last=True):
        n_trajs += 1
        assert traj.states is not None
        assert traj.actions is not None
        assert traj.rewards is not None
        assert traj.states.shape == (1, 64, 64, 3)
        assert traj.actions.shape == (1,)
        assert traj.rewards.shape == (1,)

    assert n_trajs == timesteps * n_envs


@given(
    trajs_per_env=integers(min_value=2, max_value=1000),
    n_envs=integers(min_value=1, max_value=10),
    length=integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_trajs(trajs_per_env: int, n_envs: int, length: int) -> None:
    timesteps = trajs_per_env * length
    states = np.empty((timesteps, n_envs, 64, 64, 3))
    actions = np.empty((timesteps, n_envs), dtype=np.int8)
    rewards = np.empty((timesteps, n_envs))
    firsts = np.zeros((timesteps, n_envs), dtype=bool)
    firsts[::length] = True

    data = RlDataset.from_gym3(
        states=states,
        actions=actions,
        rewards=rewards,
        firsts=firsts,
        keep_incomplete=False,
    )
    n_trajs = 0
    for traj in data.trajs(include_last=True):
        n_trajs += 1
        assert traj.states is not None
        assert traj.actions is not None
        assert traj.rewards is not None
        assert traj.states.shape == (length, 64, 64, 3)
        assert traj.actions.shape == (length,)
        assert traj.rewards.shape == (length,)

    assert n_trajs == (trajs_per_env - 1) * n_envs

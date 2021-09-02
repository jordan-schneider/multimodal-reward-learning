import logging
from typing import Tuple

import numpy as np
import torch
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import booleans, floats, integers, just, shared, tuples
from mrl.offline_buffer import SarsDataset
from torch.testing import assert_equal


def np2t(*args: np.ndarray) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(arg) for arg in args)


normal_floats = floats(allow_nan=False, allow_infinity=False)
traj_length = shared(integers(1, 100))

states_strategy = arrays(
    shape=tuples(traj_length, just(64), just(64), just(3)), elements=normal_floats, dtype=float
)
actions_strategy = arrays(
    shape=traj_length, elements=integers(min_value=0, max_value=15), dtype=int
)
rewards_strategy = arrays(shape=traj_length, elements=normal_floats, dtype=float)
strategy = arrays(shape=traj_length, elements=booleans(), dtype=bool).filter(
    lambda arr: not arr.all()
)


@given(
    states_np=states_strategy,
    actions_np=actions_strategy,
    rewards_np=rewards_strategy,
    dones_np=strategy,
)
def test_sars_dataset_index(
    states_np: np.ndarray, actions_np: np.ndarray, rewards_np: np.ndarray, dones_np: np.ndarray
) -> None:
    logging.basicConfig(level="DEBUG")
    states, actions, rewards, dones = np2t(states_np, actions_np, rewards_np, dones_np)
    dataset = SarsDataset(states, actions, rewards, dones)
    l = len(dataset)
    raw_index = 0
    for dataset_index in range(l):
        while dones[raw_index]:
            raw_index += 1
        actual_states, actual_actions, actual_rewards, actual_nextstate = dataset[dataset_index]
        assert_equal(
            actual_states,
            states[raw_index],
            msg=f"raw_index={raw_index}, dataset_index={dataset_index}",
        )
        assert_equal(
            actual_actions,
            actions[raw_index],
            msg=f"raw_index={raw_index}, dataset_index={dataset_index}",
        )
        assert_equal(
            actual_rewards,
            rewards[raw_index],
            msg=f"raw_index={raw_index}, dataset_index={dataset_index}",
        )

        raw_index += 1

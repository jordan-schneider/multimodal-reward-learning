from typing import Final

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from linear_procgen import Miner
from mrl.dataset.random_policy import RandomPolicy
from mrl.dataset.roller import DatasetRoller, procgen_rollout

N_FEATURES = 5


def test_rollout_fills_arrays():
    NUM: Final[int] = 2
    env = Miner(np.zeros(N_FEATURES), num=NUM)
    env = ExtractDictObWrapper(env, "rgb")

    policy = RandomPolicy(env.ac_space, num=NUM)

    states, actions, rewards, firsts = procgen_rollout(
        env, policy, timesteps=20, check_occupancies=True
    )
    # Assertion is inside procgen_rollout unless -O is passed


def test_rollout_always_firsts():
    N_ENVS: Final[int] = 100
    N_TRAJS = 100
    T = 1001
    env = Miner(np.zeros(N_FEATURES), num=N_ENVS)
    env = ExtractDictObWrapper(env, "rgb")

    policy = RandomPolicy(env.ac_space, num=N_ENVS)

    roller = DatasetRoller(
        env=env,
        policy=policy,
        n_actions=T,
        n_trajs=N_TRAJS,
        flags=["first", "feature"],
        remove_incomplete=True,
    )

    dataset = roller.roll()

    assert roller.firsts is not None
    firsts = roller.firsts.array
    assert len(firsts.shape) == 2
    assert firsts.shape[1] == N_ENVS
    assert np.all(np.any(firsts, axis=0))

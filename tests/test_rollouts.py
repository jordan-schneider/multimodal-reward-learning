from typing import Final

import numpy as np
from gym3 import ExtractDictObWrapper  # type: ignore
from mrl.dataset.random_policy import RandomPolicy
from mrl.dataset.roller import procgen_rollout
from mrl.envs import Miner


def test_rollout_fills_arrays():
    NUM: Final[int] = 2
    env = Miner(np.zeros(4), num=NUM)
    env = ExtractDictObWrapper(env, "rgb")

    policy = RandomPolicy(env.ac_space, num=NUM)

    states, actions, rewards, firsts = procgen_rollout(
        env, policy, timesteps=20, check_occupancies=True
    )
    # Assertion is inside procgen_rollout unless -O is passed

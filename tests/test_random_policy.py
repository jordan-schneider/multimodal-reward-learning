import numpy as np
import pytest
from hypothesis import given, settings
from linear_procgen import make_env
from procgen import ProcgenGym3Env
from question_gen.random_policy import RandomGridPolicy

from .strategies import seeds as seed_strategy


def test_throws_on_nongrid_env():
    env = ProcgenGym3Env(1, "miner")
    with pytest.raises(ValueError):
        RandomGridPolicy(env, np.random.default_rng())


@settings(deadline=None)
@given(seed=seed_strategy)
def test_policy_on_miner(seed: int):
    env = make_env("miner", num=1, reward=1)
    policy = RandomGridPolicy(env, np.random.default_rng(seed))
    for _ in range(100):
        obs, _, _ = env.observe()
        env.act(policy(obs))


@settings(deadline=None)
@given(seed=seed_strategy)
def test_policy_on_maze(seed: int):
    env = make_env("maze", num=1, reward=1)
    policy = RandomGridPolicy(env, np.random.default_rng(seed))
    for _ in range(100):
        obs, _, _ = env.observe()
        env.act(policy(obs))

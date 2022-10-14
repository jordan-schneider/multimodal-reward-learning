import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, one_of
from linear_procgen.util import make_env
from mrl.aligned_rewards.make_ars import make_aligned_reward_set
from mrl.dataset.random_policy import RandomPolicy
from mrl.inference.posterior import cover_sphere


@given(
    reward=arrays(
        dtype=np.float32,
        shape=(6,),
        elements=one_of(
            floats(-1.0, -0.10000000149011612, width=32),
            floats(0.10000000149011612, 1.0, width=32),
        ),
    ),
    seed=integers(0, 2**31 - 1),
)
@pytest.mark.skip()
@settings(deadline=None, max_examples=100)
def test_aligned_reward_set_nonredundant(reward: np.ndarray, seed: int) -> None:
    torch.manual_seed(seed)

    reward = reward / np.linalg.norm(reward)
    env = make_env(name="miner", num=1, reward=reward)

    policy = RandomPolicy(env.ac_space, 1)
    assert policy.device == torch.device("cpu")

    diffs = make_aligned_reward_set(
        reward=reward,
        n_states=2,
        n_trajs=2,
        env=env,
        policy=policy,
        tqdm=False,
    )

    reward_samples = cover_sphere(
        n_samples=100_000, ndims=6, rng=np.random.default_rng(seed)
    )
    agreement = (diffs @ reward_samples.T > 0).T
    reward_aligned = np.all(agreement, axis=1)

    for i in range(diffs.shape[0]):
        agreement_leave_out = np.concatenate((agreement[:i], agreement[i + 1 :]))
        reward_aligned_leave_out = np.all(agreement_leave_out, axis=1)
        assert not np.array_equal(reward_aligned, reward_aligned_leave_out)

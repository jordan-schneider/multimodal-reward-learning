import numpy as np
import torch
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from mrl.aligned_rewards.make_ars import make_aligned_reward_set
from mrl.dataset.random_policy import RandomPolicy
from mrl.envs.util import make_env
from mrl.inference.posterior import cover_sphere


@given(
    reward=arrays(
        dtype=np.float32,
        shape=(5,),
        elements=floats(-1.0, 1.0, width=32).filter(lambda x: np.all(np.abs(x) > 0.1)),
    ).filter(lambda r: np.any(r != 0)),
    seed=integers(0, 2 ** 31 - 1),
)
@settings(deadline=None, max_examples=100)
def test_aligned_reward_set_nonredundant(reward: np.ndarray, seed: int) -> None:
    torch.manual_seed(seed)

    reward = reward / np.linalg.norm(reward)
    env = make_env(kind="miner", num=1, reward=reward)

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
        n_samples=100_000, ndims=5, rng=np.random.default_rng(seed)
    )
    agreement = (diffs @ reward_samples.T > 0).T
    reward_aligned = np.all(agreement, axis=1)

    for i in range(diffs.shape[0]):
        agreement_leave_out = np.concatenate((agreement[:i], agreement[i + 1 :]))
        reward_aligned_leave_out = np.all(agreement_leave_out, axis=1)
        assert not np.array_equal(reward_aligned, reward_aligned_leave_out)

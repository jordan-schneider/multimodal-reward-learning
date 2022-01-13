import numpy as np
import torch
from gym3 import ExtractDictObWrapper  # type: ignore
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from mrl.aligned_reward_set import make_aligned_reward_set
from mrl.envs.miner import Miner
from mrl.random_policy import RandomPolicy


@given(
    reward=arrays(
        dtype=np.float32, shape=(5,), elements=floats(-1.0, 1.0, width=32)
    ).filter(lambda r: np.any(r != 0)),
    n_states=integers(min_value=2, max_value=10),
    n_trajs=integers(min_value=2, max_value=10),
    seed=integers(0, 2 ** 31 - 1),
)
@settings(deadline=None)
def test_aligned_reward_set_consistent(
    reward: np.ndarray, n_states: int, n_trajs: int, seed: int
) -> None:
    torch.manual_seed(seed)

    reward = reward / np.linalg.norm(reward)
    env = ExtractDictObWrapper(Miner(reward_weights=reward, num=1), "rgb")

    policy = RandomPolicy(env.ac_space, 1)
    assert policy.device == torch.device("cpu")

    diffs = make_aligned_reward_set(
        reward, n_states, n_trajs, env, policy, use_done_feature=True, tqdm=False
    )

    assert np.all(diffs @ reward > 0)

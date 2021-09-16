import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import floats, integers, just, tuples
from mrl.reward_model import reward_prop_likelihood


@given(
    rewards=arrays(
        dtype=np.float32,
        elements=floats(allow_infinity=False, allow_nan=False, width=32),
        shape=tuples(integers(min_value=1, max_value=1_000), just(5)),
    ),
    diffs=arrays(
        dtype=np.float32,
        elements=floats(allow_infinity=False, allow_nan=False, width=32),
        shape=tuples(integers(min_value=1, max_value=1_000), just(5)),
    ),
)
def test_likelihood_multiple_rewards(rewards: np.ndarray, diffs: np.ndarray):
    likelihoods = reward_prop_likelihood(rewards, diffs)
    assert likelihoods.shape == (rewards.shape[0],)


@given(
    rewards=arrays(
        dtype=np.float32,
        elements=floats(allow_infinity=False, allow_nan=False, width=32),
        shape=(5,),
    ),
    diffs=arrays(
        dtype=np.float32,
        elements=floats(allow_infinity=False, allow_nan=False, width=32),
        shape=tuples(integers(min_value=1, max_value=1_000), just(5)),
    ),
)
def test_likelihood_single_reward(rewards: np.ndarray, diffs: np.ndarray):
    likelihoods = reward_prop_likelihood(rewards, diffs)
    assert likelihoods.shape == ()

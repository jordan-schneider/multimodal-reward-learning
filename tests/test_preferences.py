import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, tuples, just
from mrl.dataset.preferences import orient_features

feature_array = arrays(
    dtype=int,
    shape=(1, 2, 4),
    elements=integers(min_value=-100, max_value=100),
)
features_array = arrays(
    dtype=int,
    shape=tuples(integers(min_value=1, max_value=100), just(2), just(4)),
    elements=integers(min_value=-100, max_value=100),
)

reward_array = arrays(
    dtype=float,
    shape=(4,),
    elements=floats(allow_nan=False, allow_infinity=False, width=32),
)

seed_strategy = integers(0, 2 ** 31 - 1)


@given(
    feature=feature_array,
    temp=floats(min_value=0.0, max_value=10.0),
    reward=reward_array,
    seed=seed_strategy,
)
def test_orient_feature(
    feature: np.ndarray,
    temp: float,
    reward: np.ndarray,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    diff = feature[:, 0] - feature[:, 1]
    if np.all(diff == 0):
        return
    oriented_feature, _ = orient_features(
        features=feature,
        diffs=diff,
        temperature=temp,
        reward=reward,
        rng=rng,
    )
    if oriented_feature.shape[0] == 0:
        return
    assert len(oriented_feature.shape) == 3
    assert np.all(oriented_feature == feature) or np.all(
        oriented_feature == feature[:, [1, 0]]
    )


@given(
    features=features_array,
    temp=floats(min_value=0.0, max_value=10.0),
    reward=reward_array,
    seed=seed_strategy,
)
def test_orient_features(
    features: np.ndarray,
    temp: float,
    reward: np.ndarray,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    diffs = features[:, 0] - features[:, 1]

    nonzero_strengths = diffs @ reward > 1e-8
    features = features[nonzero_strengths]
    diffs = diffs[nonzero_strengths]
    if features.shape[0] == 0:
        return

    oriented_features, _ = orient_features(
        features=features,
        diffs=diffs,
        temperature=temp,
        reward=reward,
        rng=rng,
    )
    if oriented_features.shape[0] == 0:
        return

    assert len(oriented_features.shape) == 3
    first_order = np.all(
        np.all(np.abs(oriented_features - features) < 1e-5, axis=1), axis=1
    )
    second_order = np.all(
        np.all(np.abs(oriented_features - features[:, [1, 0]]) < 1e-5, axis=1), axis=1
    )
    assert np.all(np.bitwise_or(first_order, second_order))

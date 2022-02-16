import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from mrl.util import get_angle, soft_dedup

epsilon_strat = floats(min_value=1e-3, max_value=1)


@given(
    array=arrays(
        dtype=np.float32,
        shape=tuples(integers(1, 100), integers(1, 20)),
        elements=floats(allow_infinity=False, allow_nan=False, width=32),
    ),
    epsilon=epsilon_strat,
)
def test_soft_dedup(array: np.ndarray, epsilon: float):
    out = soft_dedup(array, epsilon)
    # There must not be more output than input
    assert out.shape[0] <= array.shape[0]

    # Every output row must be in the input
    for row in out:
        assert np.any(np.all(row == array, axis=1))

    # Every output row must be unique
    assert (
        out.shape[0] == np.unique(out, axis=0).shape[0]
    ), f"Not all outputs are unique: {out}"


@given(
    row=arrays(
        dtype=np.float32,
        shape=integers(1, 20),
        elements=floats(allow_nan=False, allow_infinity=False, width=32),
    ).filter(lambda x: not np.all(x == 0) and np.isfinite(get_angle(x, -x))),
    epsilon=epsilon_strat,
)
def test_soft_dedup_keeps_opposites(row: np.ndarray, epsilon: float):
    arr = np.array([row, -row])
    out = soft_dedup(arr, epsilon)
    assert out.shape[0] == 2


def find_ortho(vec: np.ndarray) -> np.ndarray:
    assert len(vec.shape) == 1
    out = np.zeros_like(vec)
    if vec[0] == 0:
        out[0] = 1
    elif vec[1] == 0:
        out[1] = 1
    else:
        out[0] = -vec[1]
        out[1] = vec[0]
    return out


@given(
    row=arrays(
        dtype=np.float32,
        shape=integers(2, 20),
        elements=floats(allow_nan=False, allow_infinity=False, width=32),
    ).filter(lambda x: not np.all(x == 0) and np.isfinite(get_angle(x, find_ortho(x)))),
    epsilon=epsilon_strat,
)
def test_soft_dedup_keeps_ortho(row: np.ndarray, epsilon: float):
    arr = np.array([row, find_ortho(row)])
    out = soft_dedup(arr, epsilon)
    assert out.shape[0] == 2

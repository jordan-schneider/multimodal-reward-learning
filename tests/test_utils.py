from typing import Tuple

import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, integers
from question_gen.util import remove_duplicates, remove_zeros

from .strategies import floats_1000

unique_arrays = arrays(
    dtype=np.float32, shape=(5, 4), elements=floats_1000, unique=True
)
nonzero_arrays = arrays(dtype=np.float32, shape=(5, 4), elements=floats_1000).filter(
    lambda x: not np.any(np.all(x == 0.0, axis=1))
)


def min_magnitude(vecs: np.ndarray) -> float:
    tmp_vecs = np.copy(vecs)
    tmp_vecs[tmp_vecs == 0.0] = np.max(np.abs(tmp_vecs))
    sensitivity = np.min(np.abs(tmp_vecs)) / 2
    return sensitivity


@composite
def arrays_insert(draw, arrays):
    vecs = draw(arrays)
    idx = draw(integers(min_value=0, max_value=vecs.shape[0] - 1))
    return (vecs, idx)


@given(vecs=unique_arrays)
def test_remove_duplicates_noop(vecs: np.ndarray):
    sensitivity = min_magnitude(vecs)
    assert np.array_equal(vecs, remove_duplicates(vecs, sensitivity)[0])


@given(vecs_and_insert=arrays_insert(arrays=unique_arrays), n_duplicates=integers(1, 5))
def test_remove_duplicates(vecs_and_insert: Tuple[np.ndarray, int], n_duplicates: int):
    vecs, start_index = vecs_and_insert
    dup_vecs = np.insert(vecs, start_index, np.tile(vecs[0], (n_duplicates, 1)), axis=0)
    out_vecs, indices = remove_duplicates(dup_vecs, min_magnitude(vecs))
    assert np.array_equal(dup_vecs[np.sort(indices)], vecs)


@given(vecs=arrays(dtype=np.float32, shape=(5, 4), elements=floats_1000))
def test_remove_duplicates_indices(vecs: np.ndarray):
    out_vecs, indices = remove_duplicates(vecs)
    assert np.array_equal(vecs[indices], out_vecs)


@given(vecs=nonzero_arrays)
def test_remove_zeros_noop(vecs: np.ndarray):
    assert np.array_equal(vecs, remove_zeros(vecs)[0])


@given(vecs=nonzero_arrays)
def test_remove_zeros_indices(vecs: np.ndarray):
    out_vecs, indices = remove_zeros(vecs)
    assert np.array_equal(vecs[indices], out_vecs)


@given(vecs_and_insert=arrays_insert(arrays=nonzero_arrays), n_zeros=integers(1, 5))
def test_remove_zeros(vecs_and_insert: Tuple[np.ndarray, int], n_zeros: int):
    vecs, start_index = vecs_and_insert
    zeros_vecs = np.insert(
        vecs, start_index, np.zeros((n_zeros, vecs.shape[1])), axis=0
    )
    out_vecs, _ = remove_zeros(zeros_vecs)
    assert np.array_equal(vecs, out_vecs)

from typing import Tuple

import numpy as np


def mean_geodesic_distance(center, points, weights) -> float:
    dists = np.arccos(np.clip(points @ center, -1, 1))
    assert np.all(dists >= 0)
    return np.average(dists, weights=weights)


def find_centroid(
    points: np.ndarray, weights: np.ndarray, max_iter: int, init: np.ndarray
) -> Tuple[np.ndarray, float]:
    i = 0
    centroid = init.copy()
    centroid /= np.linalg.norm(centroid)
    assert np.allclose(
        np.linalg.norm(centroid), 1.0
    ), f"centroid={centroid} has norm={np.linalg.norm(centroid)} far from 1."

    numerator = points.T * weights

    best_centroid = centroid.copy()
    best_dist = np.inf
    while i < max_iter:
        grad = np.sum(numerator / np.sqrt(1 - (points @ centroid) ** 2), axis=1)
        centroid += grad

        if np.any(np.logical_not(np.isfinite(centroid))):
            # If any dimensions are inf, then set those dimensions to 1, set all the finite dims to
            # 0, and normalize.
            centroid = np.logical_not(np.isfinite(centroid)).astype(centroid.dtype)

        # If you can't normalize because the norm is inf, then scale everything down by 100
        while not np.isfinite(np.linalg.norm(centroid)):
            centroid /= 100

        centroid /= np.linalg.norm(centroid)

        assert np.allclose(
            np.linalg.norm(centroid), 1.0
        ), f"centroid={centroid} has norm={np.linalg.norm(centroid)} far from 1."

        dist = mean_geodesic_distance(centroid, points, weights)

        if dist < best_dist:
            best_dist = dist
            best_centroid = centroid.copy()

        i += 1
    return best_centroid, best_dist

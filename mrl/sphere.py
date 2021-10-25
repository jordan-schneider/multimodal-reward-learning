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


# rng = np.random.default_rng()
# points = rng.uniform(low=-1, high=1, size=(1000, 5))
# points = normalize(points)
# points = np.abs(points)
# weights = rng.uniform(size=1000)

# weighted_mean = np.average(points, weights=weights, axis=0)

# results = [find_centroid(points, weights, tolerance=1e-10, max_iter=10000, init=p) for p in points]
# results.append(find_centroid(points, weights, tolerance=1e-10, max_iter=10000, init=weighted_mean))

# centroid, dist = min(results, key=lambda x: x[1])
# print(centroid)
# print(dist)

# print(np.where([np.all(r[0] == centroid) for r in results]))

# print(min([mean_geodesic_distance(p, points, weights) for p in points]))

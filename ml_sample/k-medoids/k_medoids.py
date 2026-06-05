from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def _pairwise_distances(points_array: np.ndarray) -> np.ndarray:
    diff = points_array[:, None, :] - points_array[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _assign_points(pairwise_distances: np.ndarray, medoid_indices: Sequence[int]) -> list[int]:
    distances = pairwise_distances[:, list(medoid_indices)]
    nearest_medoid_pos = np.argmin(distances, axis=1)
    medoid_array = np.asarray(medoid_indices)
    return medoid_array[nearest_medoid_pos].astype(int).tolist()


def _total_cost(pairwise_distances: np.ndarray, medoid_indices: Sequence[int]) -> float:
    distances = pairwise_distances[:, list(medoid_indices)]
    return float(np.min(distances, axis=1).sum())


def k_medoids(
    points_array: np.ndarray,
    k: int,
    seed: int = 42,
    max_iter: int = 100,
) -> tuple[list[int], list[int], float]:
    if points_array.ndim != 2:
        raise ValueError("points_array must be 2D")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if k > len(points_array):
        raise ValueError("k must be less than or equal to number of points")

    n_samples = len(points_array)
    pairwise = _pairwise_distances(points_array)

    rng = random.Random(seed)
    medoid_indices = rng.sample(range(n_samples), k)

    current_cost = _total_cost(pairwise, medoid_indices)

    for _ in range(max_iter):
        medoid_set = set(medoid_indices)
        non_medoids = np.array([i for i in range(n_samples) if i not in medoid_set], dtype=int)
        if non_medoids.size == 0:
            break

        current_cols = pairwise[:, medoid_indices]
        best_swap_cost = current_cost
        best_swap: tuple[int, int] | None = None

        for medoid_pos in range(k):
            other_positions = [pos for pos in range(k) if pos != medoid_pos]
            if other_positions:
                other_min = np.min(current_cols[:, other_positions], axis=1)
            else:
                other_min = np.full(n_samples, np.inf)

            candidate_cols = pairwise[:, non_medoids]
            candidate_costs = np.minimum(other_min[:, None], candidate_cols).sum(axis=0)
            pos_best_idx = int(np.argmin(candidate_costs))
            pos_best_cost = float(candidate_costs[pos_best_idx])

            if pos_best_cost < best_swap_cost:
                best_swap_cost = pos_best_cost
                best_swap = (medoid_pos, int(non_medoids[pos_best_idx]))

        if best_swap is None:
            break

        medoid_pos, candidate = best_swap
        medoid_indices[medoid_pos] = candidate
        current_cost = best_swap_cost

    assignments = _assign_points(pairwise, medoid_indices)
    return medoid_indices, assignments, current_cost

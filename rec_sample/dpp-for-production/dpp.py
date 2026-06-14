"""DPP-based diversified recommendation.

Reference:
Practical Diversified Recommendations on YouTube with Determinantal Point Processes
(CIKM 2018)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array1D = np.ndarray
Array2D = np.ndarray


@dataclass(frozen=True)
class DPPParams:
    alpha: float
    sigma: float


def jaccard_distance_matrix(binary_features: Array2D) -> Array2D:
    n_items = binary_features.shape[0]
    distances = np.zeros((n_items, n_items), dtype=np.float64)

    for i in range(n_items):
        a = binary_features[i].astype(bool)
        for j in range(i + 1, n_items):
            b = binary_features[j].astype(bool)
            union = np.logical_or(a, b).sum()
            if union == 0:
                value = 0.0
            else:
                inter = np.logical_and(a, b).sum()
                value = 1.0 - (inter / union)
            distances[i, j] = value
            distances[j, i] = value

    return distances


def project_to_psd(matrix: Array2D, eps: float = 1e-12) -> Array2D:
    symmetric = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    if float(np.min(eigvals)) >= -eps:
        return symmetric
    clipped = np.maximum(eigvals, eps)
    return (eigvecs * clipped) @ eigvecs.T


def build_kernel(qualities: Array1D, distances: Array2D, params: DPPParams) -> Array2D:
    if qualities.ndim != 1:
        raise ValueError("qualities must be a 1D array")
    if params.sigma <= 0:
        raise ValueError("sigma must be > 0")
    n_items = qualities.size
    if distances.shape != (n_items, n_items):
        raise ValueError("distances must be (N, N) matching qualities")

    kernel = np.zeros((n_items, n_items), dtype=np.float64)
    np.fill_diagonal(kernel, qualities**2)

    scale = -1.0 / (2.0 * params.sigma * params.sigma)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            value = params.alpha * qualities[i] * qualities[j] * np.exp(distances[i, j] * scale)
            kernel[i, j] = value
            kernel[j, i] = value

    return project_to_psd(kernel)


def greedy_approx_max(kernel: Array2D, m: int) -> list[int]:
    n_items = kernel.shape[0]
    if m < 0 or m > n_items:
        raise ValueError("m must satisfy 0 <= m <= N")

    selected: list[int] = []
    remaining = set(range(n_items))

    for _ in range(m):
        best = -1
        best_det = -np.inf
        for c in remaining:
            idx = selected + [c]
            sub = kernel[np.ix_(idx, idx)]
            det = float(np.linalg.det(sub))
            if det > best_det:
                best_det = det
                best = c

        if best < 0:
            break

        selected.append(best)
        remaining.remove(best)

    return selected


def rank_feed_via_dpp(
    qualities: Array1D,
    features: Array2D,
    params: DPPParams,
    window_size: int,
) -> list[int]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if qualities.shape[0] != features.shape[0]:
        raise ValueError("qualities and features must have same first dimension")

    distances = jaccard_distance_matrix(features)
    full_kernel = build_kernel(qualities, distances, params)

    remaining = list(range(len(qualities)))
    ranked: list[int] = []

    while remaining:
        current = min(window_size, len(remaining))
        local = full_kernel[np.ix_(remaining, remaining)]
        local_selected = greedy_approx_max(local, current)
        global_selected = [remaining[i] for i in local_selected]

        ranked.extend(global_selected)
        selected_set = set(global_selected)
        remaining = [i for i in remaining if i not in selected_set]

    return ranked

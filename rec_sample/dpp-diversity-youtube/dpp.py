"""DPP-based feed ranking following Algorithm 1 from the paper.

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
    """Parameters for the simple DPP kernel from Eq. (9) and Eq. (10)."""

    alpha: float
    sigma: float


def jaccard_distance_matrix(binary_features: Array2D) -> Array2D:
    """Compute pairwise Jaccard distance matrix for binary feature vectors."""

    n_items = binary_features.shape[0]
    distances = np.zeros((n_items, n_items), dtype=np.float64)

    for i in range(n_items):
        a = binary_features[i].astype(bool)
        for j in range(i + 1, n_items):
            b = binary_features[j].astype(bool)
            union = np.logical_or(a, b).sum()
            if union == 0:
                d_ij = 0.0
            else:
                intersection = np.logical_and(a, b).sum()
                d_ij = 1.0 - (intersection / union)
            distances[i, j] = d_ij
            distances[j, i] = d_ij

    return distances


def project_to_psd(matrix: Array2D, eps: float = 1e-12) -> Array2D:
    """Project a symmetric matrix to the PSD cone by clipping eigenvalues."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eps)
    return (eigenvectors * clipped) @ eigenvectors.T


def build_kernel(qualities: Array1D, distances: Array2D, params: DPPParams) -> Array2D:
    """Build the DPP kernel using Eq. (9) and Eq. (10) in the paper."""

    if params.sigma <= 0:
        raise ValueError("sigma must be > 0")
    if qualities.ndim != 1:
        raise ValueError("qualities must be a 1D array")
    if distances.shape != (qualities.size, qualities.size):
        raise ValueError("distances shape must be (N, N) and match qualities length")

    n_items = qualities.size
    kernel = np.zeros((n_items, n_items), dtype=np.float64)

    # Eq. (9): L_ii = q_i^2
    np.fill_diagonal(kernel, qualities**2)

    # Eq. (10): L_ij = alpha * q_i * q_j * exp(-D_ij / (2*sigma^2)) for i != j
    scale = -1.0 / (2.0 * params.sigma * params.sigma)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            value = params.alpha * qualities[i] * qualities[j] * np.exp(distances[i, j] * scale)
            kernel[i, j] = value
            kernel[j, i] = value

    return project_to_psd(kernel)


def greedy_approx_max(kernel: Array2D, m: int) -> list[int]:
    """Greedy approximation for argmax_{|Y|=m} det(L_Y)."""

    n_items = kernel.shape[0]
    if m < 0 or m > n_items:
        raise ValueError("m must satisfy 0 <= m <= N")

    selected: list[int] = []
    remaining: set[int] = set(range(n_items))

    for _ in range(m):
        best_idx = -1
        best_det = -np.inf

        for candidate in remaining:
            indices = selected + [candidate]
            submatrix = kernel[np.ix_(indices, indices)]
            det_value = float(np.linalg.det(submatrix))

            if det_value > best_det:
                best_det = det_value
                best_idx = candidate

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def rank_feed_via_dpp(
    qualities: Array1D,
    features: Array2D,
    params: DPPParams,
    window_size: int,
) -> list[int]:
    """Rank a feed via DPP windows as described in Algorithm 1."""

    if window_size <= 0:
        raise ValueError("window_size must be > 0")

    distances = jaccard_distance_matrix(features)
    full_kernel = build_kernel(qualities=qualities, distances=distances, params=params)

    remaining_indices = list(range(len(qualities)))
    ranked_indices: list[int] = []

    while remaining_indices:
        current_size = min(window_size, len(remaining_indices))

        local_kernel = full_kernel[np.ix_(remaining_indices, remaining_indices)]
        local_selected = greedy_approx_max(local_kernel, current_size)

        selected_global = [remaining_indices[i] for i in local_selected]
        ranked_indices.extend(selected_global)

        selected_set = set(selected_global)
        remaining_indices = [idx for idx in remaining_indices if idx not in selected_set]

    return ranked_indices

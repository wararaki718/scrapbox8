"""Core DPP MAP inference algorithms.

This module contains math-focused functions for building an L-ensemble
kernel and running greedy MAP inference with trace outputs.
"""

from __future__ import annotations

import numpy as np


def rbf_similarity(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """Compute RBF similarity between two vectors.

    Args:
        x: First vector.
        y: Second vector.
        sigma: Gaussian kernel bandwidth.

    Returns:
        RBF similarity value in [0, 1].
    """
    dist2 = float(np.sum((x - y) ** 2))
    return float(np.exp(-dist2 / (2.0 * sigma**2)))


def build_l_kernel(
    points: np.ndarray,
    sigma: float = 0.16,
    quality: np.ndarray | None = None,
) -> np.ndarray:
    """Build an L-ensemble kernel matrix from points and optional quality terms.

    L-ensemble parameterization used here:
        L_ij = q_i * s_ij * q_j
    where q_i is an item quality term and s_ij is pairwise similarity.
    In this implementation, s_ij is computed by an RBF kernel.

    Args:
        points: Point array of shape (N, D).
        sigma: Bandwidth for pairwise RBF similarities.
        quality: Optional non-negative quality vector of shape (N,).

    Returns:
        Positive semi-definite L kernel with tiny diagonal jitter for stability.
    """
    n = len(points)
    s = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            s[i, j] = rbf_similarity(points[i], points[j], sigma)

    if quality is None:
        quality = np.ones(n, dtype=float)

    # L-ensemble construction: L = diag(q) * S * diag(q)
    # (element-wise form: L_ij = q_i * s_ij * q_j)
    l_kernel = (quality[:, None] * s) * quality[None, :]
    l_kernel += np.eye(n) * 1e-9
    return l_kernel


def logdet_subset(l_kernel: np.ndarray, subset: list[int]) -> float:
    """Compute log-determinant objective for a selected subset.

    Args:
        l_kernel: Full L-ensemble kernel matrix.
        subset: Selected item indices.

    Returns:
        log(det(L_subset)). Returns 0.0 for empty subset and -inf for invalid sign.
    """
    if len(subset) == 0:
        return 0.0

    submatrix = l_kernel[np.ix_(subset, subset)]
    sign, logdet = np.linalg.slogdet(submatrix)
    if sign <= 0:
        return -np.inf
    return float(logdet)


def greedy_map_with_trace(
    l_kernel: np.ndarray,
    max_k: int,
    stop_when_nonpositive: bool = False,
) -> tuple[list[int], list[dict]]:
    """Run greedy MAP inference and record per-step diagnostics.

    Args:
        l_kernel: Full L-ensemble kernel matrix.
        max_k: Maximum number of selected items.
        stop_when_nonpositive: If True, stop when best marginal gain <= 0.

    Returns:
        A tuple of selected indices and a trace list for visualization.
        Each trace entry stores gains, objective, Cholesky factor of L_S,
                and residual matrix after a rank-1 Schur update.

        Marginal gain definition:
            For each candidate i not in S,
            gain(i) = logdet(L_{S U {i}}) - logdet(L_S)
            This is the incremental improvement in the DPP objective when i
            is added to the current selected set S. Greedy MAP chooses the
            candidate with the maximum marginal gain.

                Note:
                        - `chol_sub` is recomputed from the original `l_kernel` restricted to
                            the selected indices `S`.
                        - `rank1_matrix` tracks only the remaining (not-yet-selected) items,
                            after removing the chosen pivot via rank-1 Schur updates.
    """
    n = l_kernel.shape[0]
    selected: list[int] = []
    trace: list[dict] = []

    remaining_global = list(range(n))
    schur_matrix = l_kernel.copy()

    for step in range(max_k):
        base = logdet_subset(l_kernel, selected)
        gains = np.full(n, np.nan, dtype=float)

        best_gain = -np.inf
        best_i = None

        for i in range(n):
            if i in selected:
                continue

            # Marginal gain for candidate i:
            #   Delta_i = logdet(L_{S U {i}}) - logdet(L_S)
            # This is the one-step objective improvement used by greedy MAP.
            score = logdet_subset(l_kernel, selected + [i])
            gain = score - base
            gains[i] = gain
            if gain > best_gain:
                best_gain = gain
                best_i = i

        if best_i is None:
            break

        if stop_when_nonpositive and best_gain <= 1e-10:
            break

        selected.append(best_i)

        # Selected-side view: build L_S directly from the original L kernel,
        # then compute its Cholesky factor for diagnostics/visualization.
        l_sub = l_kernel[np.ix_(selected, selected)]
        chol_sub = np.linalg.cholesky(l_sub)

        # Residual-side view: update Schur complement by rank-1 update around
        # the chosen pivot, then remove that pivot from remaining candidates.
        pivot_pos = remaining_global.index(best_i)
        pivot_val = float(schur_matrix[pivot_pos, pivot_pos])

        if pivot_val > 1e-12:
            kvec = schur_matrix[:, pivot_pos]
            schur_updated = schur_matrix - np.outer(kvec, kvec) / pivot_val
        else:
            schur_updated = schur_matrix.copy()

        schur_updated = 0.5 * (schur_updated + schur_updated.T)

        schur_next = np.delete(np.delete(schur_updated, pivot_pos, axis=0), pivot_pos, axis=1)
        remaining_next = [idx for idx in remaining_global if idx != best_i]

        trace.append(
            {
                "step": step + 1,
                "selected": selected.copy(),
                "gains": gains.copy(),
                "best_i": best_i,
                "best_gain": float(best_gain),
                "objective": logdet_subset(l_kernel, selected),
                "chol_sub": chol_sub.copy(),
                "rank1_matrix": schur_next.copy(),
                "remaining_after": remaining_next.copy(),
            }
        )

        schur_matrix = schur_next
        remaining_global = remaining_next

        if len(remaining_global) == 0:
            break

    return selected, trace

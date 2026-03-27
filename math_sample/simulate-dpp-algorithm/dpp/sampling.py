"""Sampling utilities for Determinantal Point Processes (DPP)."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def _orthonormalize_rows(matrix: Array, eps: float = 1e-10) -> Array:
    """Return an orthonormal row basis using modified Gram-Schmidt."""
    if matrix.size == 0:
        return matrix

    basis: list[Array] = []
    for row in matrix:
        vec = row.astype(float, copy=True)
        for b in basis:
            vec -= np.dot(vec, b) * b
        norm = np.linalg.norm(vec)
        if norm > eps:
            basis.append(vec / norm)

    if not basis:
        return np.zeros((0, matrix.shape[1]), dtype=float)
    return np.vstack(basis)


def sample_dpp(l_kernel: Array, seed: int | None = None) -> list[int]:
    """Sample one subset from a DPP defined by an L kernel.

    This follows the classic eigen-decomposition sampler:
    1) Select eigenvectors with probability $\lambda / (1 + \lambda)$.
    2) Iteratively sample an item with probability proportional to row norms.
    3) Condition and re-orthonormalize the selected eigenvectors.

    Args:
        l_kernel: Symmetric positive semi-definite matrix $L$.
        seed: Optional random seed.

    Returns:
        Sorted list of sampled indices.
    """
    if l_kernel.shape[0] != l_kernel.shape[1]:
        raise ValueError("l_kernel must be square")

    rng = np.random.default_rng(seed)

    evals, evecs = np.linalg.eigh(l_kernel)
    evals = np.clip(evals, a_min=0.0, a_max=None)
    keep_prob = evals / (1.0 + evals)
    selected = rng.random(len(evals)) < keep_prob

    v = evecs[:, selected].T
    if v.size == 0:
        return []

    chosen: list[int] = []
    n_items = l_kernel.shape[0]

    while v.shape[0] > 0:
        row_norm_sq = np.sum(v * v, axis=0)
        total = float(np.sum(row_norm_sq))
        if total <= 1e-12:
            break

        probs = row_norm_sq / total
        item = int(rng.choice(n_items, p=probs))
        chosen.append(item)

        column = v[:, item]
        pivot_idx = int(np.argmax(np.abs(column)))
        pivot_val = column[pivot_idx]
        if abs(pivot_val) <= 1e-12:
            break

        pivot_row = v[pivot_idx].copy()
        v = np.delete(v, pivot_idx, axis=0)

        if v.shape[0] > 0:
            v = v - np.outer(v[:, item] / pivot_val, pivot_row)
            v = _orthonormalize_rows(v)

    return sorted(set(chosen))


def run_sampling_experiment(
    l_kernel: Array,
    n_trials: int,
    seed: int = 0,
) -> tuple[list[list[int]], Array]:
    """Run repeated DPP sampling and estimate item marginals.

    Args:
        l_kernel: DPP kernel matrix $L$.
        n_trials: Number of independent samples.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of sampled subsets and estimated marginal selection frequencies.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    rng = np.random.default_rng(seed)
    all_samples: list[list[int]] = []
    counts = np.zeros(l_kernel.shape[0], dtype=float)

    for _ in range(n_trials):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        subset = sample_dpp(l_kernel, seed=sample_seed)
        all_samples.append(subset)
        counts[subset] += 1.0

    return all_samples, counts / n_trials


def _conditional_kernel(
    l_kernel: Array,
    selected: list[int],
    inv_selected_kernel: Array,
) -> Array:
    """Compute conditional kernel via Schur complement.

    Formula:
    $C = L - L_{:,Y} (L_{Y,Y})^{-1} L_{Y,:}$.
    """
    if not selected:
        return l_kernel.copy()

    l_all_y = l_kernel[:, selected]
    return l_kernel - l_all_y @ inv_selected_kernel @ l_all_y.T


def map_inference_rank1_updates(
    l_kernel: Array,
    max_length: int | None = None,
    eps: float = 1e-10,
) -> tuple[list[int], list[dict[str, object]]]:
    """Run greedy MAP inference while tracking rank-1 update states.

    This function follows a Schur-complement based greedy procedure and updates
    $(L_{Y,Y})^{-1}$ with block/rank-1 updates after selecting each item.

    At step $t$, candidate gain is approximated by the conditional diagonal:
    $g_i = C_{ii}$ where
    $C = L - L_{:,Y}(L_{Y,Y})^{-1}L_{Y,:}$.

    When adding item $j$, with $A = L_{Y,Y}$, $b = L_{Y,j}$, $c = L_{j,j}$,
    and $s = c - b^\top A^{-1} b$, the inverse update is:

    $A'^{-1} =
    \begin{bmatrix}
    A^{-1} + \frac{A^{-1}bb^\top A^{-1}}{s} & -\frac{A^{-1}b}{s} \\
    -\frac{b^\top A^{-1}}{s} & \frac{1}{s}
    \end{bmatrix}$.

    Args:
        l_kernel: Symmetric PSD kernel matrix $L$.
        max_length: Maximum subset size. If None, uses matrix size.
        eps: Numerical threshold to stop selection.

    Returns:
        A tuple of:
        - selected subset indices by greedy MAP.
        - per-step history dictionaries for visualization/tracing.
    """
    if l_kernel.shape[0] != l_kernel.shape[1]:
        raise ValueError("l_kernel must be square")

    n_items = l_kernel.shape[0]
    target_length = n_items if max_length is None else min(max_length, n_items)
    if target_length <= 0:
        raise ValueError("max_length must be positive when provided")

    selected: list[int] = []
    inv_selected = np.zeros((0, 0), dtype=float)
    history: list[dict[str, object]] = []

    for step in range(target_length):
        conditional_before = _conditional_kernel(l_kernel, selected, inv_selected)
        gains = np.diag(conditional_before).copy()
        if selected:
            gains[selected] = -np.inf

        chosen = int(np.argmax(gains))
        best_gain = float(gains[chosen])
        if not np.isfinite(best_gain) or best_gain <= eps:
            break

        if not selected:
            schur = float(l_kernel[chosen, chosen])
            if schur <= eps:
                break
            inv_selected = np.array([[1.0 / schur]], dtype=float)
        else:
            b = l_kernel[np.ix_(selected, [chosen])]
            u = inv_selected @ b
            schur = float(l_kernel[chosen, chosen] - (b.T @ u)[0, 0])
            if schur <= eps:
                break

            top_left = inv_selected + (u @ u.T) / schur
            top_right = -u / schur
            bottom_left = -u.T / schur
            bottom_right = np.array([[1.0 / schur]], dtype=float)
            inv_selected = np.block(
                [[top_left, top_right], [bottom_left, bottom_right]]
            )

        selected.append(chosen)
        conditional_after = _conditional_kernel(l_kernel, selected, inv_selected)
        selected_kernel = l_kernel[np.ix_(selected, selected)].copy()
        inv_selected_kernel = inv_selected.copy()

        history.append(
            {
                "step": step + 1,
                "selected": selected.copy(),
                "chosen": chosen,
                "schur": schur,
                "gains": gains,
                "conditional_kernel": conditional_after,
                "selected_kernel": selected_kernel,
                "inv_selected_kernel": inv_selected_kernel,
            }
        )

    return selected, history

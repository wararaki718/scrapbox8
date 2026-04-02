from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GreedyMapResult:
    selected: list[int]
    gains: list[float]


def fast_greedy_map_inference(
    kernel: np.ndarray,
    max_length: int | None = None,
    epsilon: float = 1e-12,
    stop_at_one: bool = False,
) -> GreedyMapResult:
    """Algorithm 1 from the paper: fast exact greedy MAP inference for DPP.

    Args:
        kernel: Symmetric PSD kernel matrix L of shape (M, M).
        max_length: Optional cardinality constraint N.
        epsilon: Numerical threshold to avoid division by tiny values.
        stop_at_one: If True, stop when best d_j^2 < 1 (unconstrained rule in paper).

    Returns:
        Selected indices and corresponding log-gains log(d_j^2).
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")

    m = kernel.shape[0]
    if max_length is None:
        max_length = m
    if max_length <= 0:
        return GreedyMapResult(selected=[], gains=[])

    # c[i] stores the row vector c_i (Eq. 3); d2[i] stores d_i^2 (Eq. 4).
    c: list[np.ndarray] = [np.empty((0,), dtype=float) for _ in range(m)]
    d2 = np.diag(kernel).astype(float).copy()

    selected: list[int] = []
    gains: list[float] = []
    selected_mask = np.zeros(m, dtype=bool)

    while len(selected) < max_length:
        candidates = np.where(~selected_mask)[0]
        if candidates.size == 0:
            break

        best_pos = int(np.argmax(d2[candidates]))
        j = int(candidates[best_pos])
        dj2 = float(d2[j])

        if dj2 <= epsilon:
            break
        if stop_at_one and dj2 < 1.0:
            break

        selected.append(j)
        gains.append(float(np.log(max(dj2, epsilon))))
        selected_mask[j] = True

        dj = float(np.sqrt(max(dj2, epsilon)))

        for i in np.where(~selected_mask)[0]:
            # e_i = (L_ji - <c_j, c_i>) / d_j (Eq. 9 derivation)
            dot_ci = float(np.dot(c[j], c[i])) if c[j].size > 0 else 0.0
            ei = (float(kernel[j, i]) - dot_ci) / dj
            c[i] = np.append(c[i], ei)
            d2[i] = d2[i] - ei * ei

    return GreedyMapResult(selected=selected, gains=gains)

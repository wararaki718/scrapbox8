from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SlidingWindowGreedyResult:
    selected: list[int]
    window_determinants: list[float]


def _safe_logdet(submatrix: np.ndarray, jitter: float = 1e-10) -> float:
    sign, logdet = np.linalg.slogdet(submatrix + np.eye(submatrix.shape[0]) * jitter)
    if sign <= 0:
        return -1e18
    return float(logdet)


def fast_greedy_map_sliding_window(
    kernel: np.ndarray,
    window_size: int,
    max_length: int,
    epsilon: float = 1e-12,
) -> SlidingWindowGreedyResult:
    """Algorithm 2 in the paper: greedy MAP inference with sliding-window diversity.

    This implementation follows the same objective as Eq. (10): at each step,
    maximize log det on the active window (latest w-1 selected items + candidate).
    """
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if max_length <= 0:
        return SlidingWindowGreedyResult(selected=[], window_determinants=[])

    m = kernel.shape[0]
    selected: list[int] = []
    selected_mask = np.zeros(m, dtype=bool)
    window_determinants: list[float] = []

    while len(selected) < min(max_length, m):
        active_window = selected[-(window_size - 1) :] if window_size > 1 else []
        base_logdet = 0.0
        if active_window:
            base_sub = kernel[np.ix_(active_window, active_window)]
            base_logdet = _safe_logdet(base_sub)

        best_i = -1
        best_gain = -1e18

        for i in np.where(~selected_mask)[0]:
            augmented = active_window + [int(i)]
            sub = kernel[np.ix_(augmented, augmented)]
            gain = _safe_logdet(sub) - base_logdet
            if gain > best_gain:
                best_gain = gain
                best_i = int(i)

        if best_i < 0 or best_gain < np.log(epsilon):
            break

        selected.append(best_i)
        selected_mask[best_i] = True

        active_after = selected[-window_size:]
        sub_after = kernel[np.ix_(active_after, active_after)]
        window_determinants.append(_safe_logdet(sub_after))

    return SlidingWindowGreedyResult(
        selected=selected,
        window_determinants=window_determinants,
    )

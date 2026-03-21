from __future__ import annotations

import numpy as np


def make_dpp_kernel(
    relevance: np.ndarray,
    sim: np.ndarray,
    quality_scale: float = 3.0,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Build DPP kernel as L = diag(q) S diag(q)."""
    q = np.exp(quality_scale * relevance)
    l = np.outer(q, q) * sim
    l = (l + l.T) / 2.0
    l += jitter * np.eye(len(relevance))
    return l


def dpp_greedy_map(l: np.ndarray, k: int) -> list[int]:
    """Approximate DPP MAP inference with greedy selection."""
    n = l.shape[0]
    cis = np.zeros((k, n))
    di2s = np.clip(np.diag(l).copy(), a_min=0.0, a_max=None)
    selected: list[int] = []

    for it in range(min(k, n)):
        j = int(np.argmax(di2s))
        if di2s[j] <= 1e-12:
            break

        selected.append(j)

        if it == k - 1:
            break

        ci_opt = cis[:it, j]
        di_opt = np.sqrt(di2s[j])
        eis = (l[j, :] - ci_opt @ cis[:it, :]) / (di_opt + 1e-12)
        cis[it, :] = eis
        di2s = di2s - eis**2
        di2s[j] = -np.inf

    return selected

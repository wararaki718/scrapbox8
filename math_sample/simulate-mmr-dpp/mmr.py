from __future__ import annotations

import numpy as np


def mmr_select(
    relevance: np.ndarray,
    sim: np.ndarray,
    k: int,
    lambda_rel: float = 0.7,
) -> list[int]:
    """Select top-k items with MMR greedy selection."""
    n = len(relevance)
    selected: list[int] = []
    candidates = set(range(n))

    for _ in range(min(k, n)):
        best_idx = None
        best_score = -np.inf

        for i in candidates:
            if not selected:
                novelty_penalty = 0.0
            else:
                novelty_penalty = np.max(sim[i, selected])

            score = lambda_rel * relevance[i] - (1.0 - lambda_rel) * novelty_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        assert best_idx is not None
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected

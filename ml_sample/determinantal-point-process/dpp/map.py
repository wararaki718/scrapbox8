from __future__ import annotations

import numpy as np

from .common import validate_l_kernel


def map_inference_dpp(l_kernel: np.ndarray, max_length: int | None = None) -> list[int]:
    """Approximate DPP MAP inference by greedy log-det maximization."""
    validate_l_kernel(l_kernel)

    n_items = l_kernel.shape[0]
    if max_length is None:
        max_length = n_items
    if max_length <= 0:
        return []
    max_length = min(max_length, n_items)

    # c[k, i]: coefficient used to project item i onto previously selected bases.
    c = np.zeros((max_length, n_items), dtype=float)
    # d2[i] corresponds to the current Schur complement diagonal term.
    d2 = np.clip(np.diag(l_kernel).astype(float), a_min=0.0, a_max=None)

    selected: list[int] = []
    selected_mask = np.zeros(n_items, dtype=bool)

    # Greedy step: adding item j increases log det by log(d2[j]).
    # Therefore we stop once the best remaining d2[j] is <= 1.
    for k in range(max_length):
        j = int(np.argmax(d2))
        if d2[j] <= 1.0:
            break

        selected.append(j)
        selected_mask[j] = True

        # Cholesky-like rank-1 update for all remaining items.
        if k == max_length - 1:
            break

        dj = np.sqrt(d2[j])
        for i in range(n_items):
            if selected_mask[i]:
                d2[i] = -np.inf
                continue

            if k == 0:
                proj = 0.0
            else:
                proj = float(np.dot(c[:k, j], c[:k, i]))

            c[k, i] = (l_kernel[j, i] - proj) / dj
            d2[i] = max(d2[i] - c[k, i] * c[k, i], 0.0)

        d2[j] = -np.inf

    selected.sort()
    return selected

from __future__ import annotations

import numpy as np


def facility_location_value(sim: np.ndarray, selected: list[int]) -> float:
    """Compute the facility-location submodular objective for a selected set."""
    if not selected:
        return 0.0
    return float(np.max(sim[:, selected], axis=1).sum())


def greedy_submodular_maximization(
    sim: np.ndarray, k: int
) -> tuple[list[int], list[float], list[float]]:
    """Select k items by greedy maximization and track gains and objective values."""
    n = sim.shape[0]
    selected: list[int] = []
    values: list[float] = [0.0]
    gains: list[float] = []
    current = 0.0

    for _ in range(k):
        best_j = -1
        best_gain = -1.0
        for j in range(n):
            if j in selected:
                continue
            candidate = selected + [j]
            val = facility_location_value(sim, candidate)
            gain = val - current
            if gain > best_gain:
                best_gain = gain
                best_j = j

        selected.append(best_j)
        current += best_gain
        gains.append(best_gain)
        values.append(current)

    return selected, gains, values

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_sample_points(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_clusters = 5
    points_per_cluster = 8
    n_features = 5

    centers = rng.uniform(-10.0, 10.0, size=(n_clusters, n_features))
    noise = rng.normal(0.0, 0.8, size=(n_clusters, points_per_cluster, n_features))

    points = centers[:, None, :] + noise
    return points.reshape(n_clusters * points_per_cluster, n_features).astype(float)


def format_clusters(
    points_array: np.ndarray, assignments: Sequence[int], medoid_indices: Sequence[int]
) -> str:
    def _point_to_tuple(p: np.ndarray) -> tuple[float, ...]:
        return tuple(float(x) for x in p)

    clusters = {m: [] for m in medoid_indices}
    for idx, m in enumerate(assignments):
        clusters[m].append((idx, points_array[idx]))

    lines: list[str] = []
    for m in medoid_indices:
        lines.append(f"- medoid #{m} {_point_to_tuple(points_array[m])}")
        for idx, p in clusters[m]:
            lines.append(f"  - point #{idx} {_point_to_tuple(p)}")
    return "\n".join(lines)

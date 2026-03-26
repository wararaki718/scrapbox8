from __future__ import annotations

import numpy as np


def make_points(n: int = 80, seed: int = 42) -> np.ndarray:
    """Generate 2D sample points from a simple three-cluster mixture."""
    rng = np.random.default_rng(seed)
    centers = np.array(
        [
            [2.2, 1.8],
            [-2.2, 1.5],
            [0.1, -2.3],
        ]
    )
    probs = np.array([0.40, 0.35, 0.25])
    ids = rng.choice(len(centers), size=n, p=probs)
    x = centers[ids] + 0.70 * rng.standard_normal((n, 2))
    return x


def rbf_similarity_matrix(x: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    """Build an RBF kernel (Gram) similarity matrix for input points."""
    diff = x[:, None, :] - x[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    sim = np.exp(-dist2 / (2.0 * sigma * sigma))
    np.fill_diagonal(sim, 1.0)
    return sim

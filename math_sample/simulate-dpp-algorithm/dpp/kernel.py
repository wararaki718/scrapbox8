"""Kernel construction utilities for DPP simulation."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def generate_item_features(n_items: int, dim: int, seed: int = 0) -> Array:
    """Generate item feature vectors.

    Args:
        n_items: Number of candidate items $N$.
        dim: Feature dimension $d$.
        seed: Random seed.

    Returns:
        Feature matrix $X \in \mathbb{R}^{N \times d}$.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n_items, dim))


def quality_scores(features: Array) -> Array:
    """Convert feature vectors into positive quality scores.

    The score follows the common decomposition:
    $L_{ij} = q_i S_{ij} q_j$, where $q_i > 0$ is the quality term.

    Args:
        features: Feature matrix $X$.

    Returns:
        Positive vector $q \in \mathbb{R}_{>0}^N$.
    """
    norms = np.linalg.norm(features, axis=1)
    centered = norms - np.mean(norms)
    return np.exp(0.6 * centered)


def rbf_similarity(features: Array, sigma: float = 1.0) -> Array:
    """Compute pairwise similarity matrix with the RBF kernel.

    Formula:
    $S_{ij} = \exp(-\|x_i - x_j\|^2 / (2\sigma^2))$.

    Args:
        features: Feature matrix $X$.
        sigma: Length-scale parameter $\sigma$.

    Returns:
        Similarity matrix $S \in [0, 1]^{N \times N}$.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    diff = features[:, None, :] - features[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)
    return np.exp(-sq_dist / (2.0 * sigma * sigma))


def build_l_kernel(qualities: Array, similarity: Array) -> Array:
    """Build the L-ensemble kernel matrix.

    Formula:
    $L = \mathrm{diag}(q)\,S\,\mathrm{diag}(q)$.

    Args:
        qualities: Quality vector $q$.
        similarity: Similarity matrix $S$.

    Returns:
        Positive semi-definite kernel matrix $L$.
    """
    if qualities.ndim != 1:
        raise ValueError("qualities must be a 1D vector")
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be a square matrix")
    if similarity.shape[0] != qualities.shape[0]:
        raise ValueError("dimensions of qualities and similarity do not match")

    return (qualities[:, None] * similarity) * qualities[None, :]

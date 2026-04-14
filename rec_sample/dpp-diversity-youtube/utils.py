"""Utility functions for generating sample inputs for DPP ranking."""

from __future__ import annotations

import numpy as np


def make_sample_input(seed: int = 7) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create deterministic sample data for running DPP ranking experiments.

    Returns:
        qualities: shape (N,) quality scores in [0, 1]
        features: shape (N, T) binary token matrix
        titles: list of video-like item names
    """

    rng = np.random.default_rng(seed)

    titles = [
        "NBA Highlights",
        "Basketball Training Drills",
        "Premier League Goals",
        "Soccer Tactics Explained",
        "Italian Pasta Recipe",
        "Street Food Tour",
        "Machine Learning Basics",
        "Neural Networks Intuition",
        "Stand-up Comedy Set",
        "Sketch Comedy Compilation",
        "Travel Vlog Tokyo",
        "Travel Vlog Kyoto",
    ]

    # Token dimensions: [basketball, soccer, food, ml, comedy, travel]
    features = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    # Make quality scores with mild noise so similar items can still compete.
    base = np.array([0.92, 0.86, 0.88, 0.81, 0.78, 0.74, 0.83, 0.8, 0.76, 0.73, 0.79, 0.75])
    noise = rng.normal(loc=0.0, scale=0.02, size=base.shape[0])
    qualities = np.clip(base + noise, 0.05, 0.99)

    return qualities.astype(np.float64), features, titles

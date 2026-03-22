"""Simulation orchestration for DPP MAP inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .algorithm import build_l_kernel, greedy_map_with_trace
from .visualization import animate_dpp_trace


def run_simulation(
    seed: int = 42,
    n_points: int = 24,
    sigma: float = 0.17,
    max_k: int = 9,
    stop_when_nonpositive: bool = False,
    output_dir: str = "artifacts",
    output_stem: str = "dpp_map_greedy",
) -> tuple[list[int], list[dict], Path]:
    """Execute full DPP MAP simulation and save a GIF animation.

    Args:
        seed: Random seed.
        n_points: Number of sampled 2D points.
        sigma: RBF kernel bandwidth.
        max_k: Maximum number of greedy selections.
        stop_when_nonpositive: Whether to apply theoretical stop condition.
        output_dir: Directory where GIF is saved.
        output_stem: Filename stem of the saved GIF.

    Returns:
        A tuple of selected indices, trace objects, and GIF path.
    """
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.05, 0.95, size=(n_points, 2))
    quality = rng.uniform(0.8, 1.25, size=n_points)

    l_kernel = build_l_kernel(points, sigma=sigma, quality=quality)
    selected, trace = greedy_map_with_trace(
        l_kernel,
        max_k=max_k,
        stop_when_nonpositive=stop_when_nonpositive,
    )

    output_path = animate_dpp_trace(
        points=points,
        trace=trace,
        out_dir=Path(output_dir),
        stem=output_stem,
    )
    return selected, trace, output_path

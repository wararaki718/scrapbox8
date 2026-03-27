"""Backward-compatible public API for DPP simulation utilities.

This module now re-exports functions from smaller modules grouped by
responsibility:
- kernel.py: feature generation and kernel construction
- sampling.py: DPP subset sampling
- visualization.py: GIF visualization
"""

from .kernel import build_l_kernel, generate_item_features, quality_scores, rbf_similarity
from .sampling import map_inference_rank1_updates, run_sampling_experiment, sample_dpp
from .visualization import (
    save_assumption_gif,
    save_map_inference_trace_gif,
    save_matrix_heatmap,
)

__all__ = [
    "build_l_kernel",
    "generate_item_features",
    "quality_scores",
    "rbf_similarity",
    "map_inference_rank1_updates",
    "run_sampling_experiment",
    "sample_dpp",
    "save_assumption_gif",
    "save_map_inference_trace_gif",
    "save_matrix_heatmap",
]

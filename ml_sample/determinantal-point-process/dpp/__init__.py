from __future__ import annotations

import numpy as np

from .common import DPPResult, build_psd_kernel, generate_demo_kernel, load_matrix, validate_l_kernel
from .map import map_inference_dpp
from .sample import sample_dpp


def run_demo(
    n_items: int,
    n_features: int,
    temperature: float,
    seed: int,
    mode: str = "sample",
    max_length: int | None = None,
) -> DPPResult:
    rng, l_kernel = generate_demo_kernel(
        n_items=n_items,
        n_features=n_features,
        temperature=temperature,
        seed=seed,
    )

    return run_dpp(
        l_kernel=l_kernel,
        mode=mode,
        rng=rng,
        max_length=max_length,
    )


def run_dpp(
    l_kernel: np.ndarray,
    mode: str = "sample",
    rng: np.random.Generator | None = None,
    max_length: int | None = None,
) -> DPPResult:
    validate_l_kernel(l_kernel)

    if mode == "sample":
        if rng is None:
            rng = np.random.default_rng()
        selected = sample_dpp(l_kernel=l_kernel, rng=rng)
    elif mode == "map":
        selected = map_inference_dpp(l_kernel=l_kernel, max_length=max_length)
    else:
        raise ValueError("mode must be either 'sample' or 'map'")

    return DPPResult(selected_indices=selected, kernel=l_kernel)


def run_from_features(
    features: np.ndarray,
    mode: str = "sample",
    rng: np.random.Generator | None = None,
    max_length: int | None = None,
    temperature: float = 1.0,
) -> DPPResult:
    l_kernel = build_psd_kernel(features=features, temperature=temperature)
    return run_dpp(l_kernel=l_kernel, mode=mode, rng=rng, max_length=max_length)


__all__ = [
    "DPPResult",
    "build_psd_kernel",
    "load_matrix",
    "map_inference_dpp",
    "run_demo",
    "run_dpp",
    "run_from_features",
    "sample_dpp",
    "validate_l_kernel",
]

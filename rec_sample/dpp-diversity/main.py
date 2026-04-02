from __future__ import annotations

import numpy as np

from fast_greedy_map import fast_greedy_map_inference
from fast_greedy_map_sliding_window import fast_greedy_map_sliding_window


def build_demo_kernel(num_items: int = 50, embed_dim: int = 8, seed: int = 7) -> np.ndarray:
    """Create a PSD kernel L = B B^T, with normalized feature vectors and relevance."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_items, embed_dim))
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    # Positive relevance scores.
    relevance = np.exp(rng.normal(loc=0.0, scale=0.3, size=(num_items,)))
    b = relevance[:, None] * features
    kernel = b @ b.T

    # Ensure exact symmetry numerically.
    return 0.5 * (kernel + kernel.T)


def main() -> None:
    kernel = build_demo_kernel(num_items=40, embed_dim=10, seed=42)

    print("=== Algorithm 1: Fast Greedy MAP Inference ===")
    result1 = fast_greedy_map_inference(kernel=kernel, max_length=10, epsilon=1e-12)
    print("selected:", result1.selected)
    print("num selected:", len(result1.selected))
    print("sum log-gain:", float(np.sum(result1.gains)))
    print()

    print("=== Algorithm 2: Sliding-Window Fast Greedy MAP ===")
    result2 = fast_greedy_map_sliding_window(
        kernel=kernel,
        window_size=5,
        max_length=10,
        epsilon=1e-12,
    )
    print("selected:", result2.selected)
    print("num selected:", len(result2.selected))


if __name__ == "__main__":
    main()

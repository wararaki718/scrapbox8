"""Run a small DPP ranking experiment with sample inputs."""

from __future__ import annotations

import numpy as np

from dpp import DPPParams, rank_feed_via_dpp
from utils import make_sample_input


def main() -> None:
    qualities, features, titles = make_sample_input(seed=7)

    params = DPPParams(alpha=1.2, sigma=0.45)
    window_size = 4

    ranked = rank_feed_via_dpp(
        qualities=qualities,
        features=features,
        params=params,
        window_size=window_size,
    )

    baseline = np.argsort(-qualities).tolist()

    print("=== Input (quality descending baseline) ===")
    for order, idx in enumerate(baseline, start=1):
        print(f"{order:2d}. [{qualities[idx]:.3f}] {titles[idx]}")

    print("\n=== Output (DPP diversified ranking) ===")
    for order, idx in enumerate(ranked, start=1):
        print(f"{order:2d}. [{qualities[idx]:.3f}] {titles[idx]}")


if __name__ == "__main__":
    main()

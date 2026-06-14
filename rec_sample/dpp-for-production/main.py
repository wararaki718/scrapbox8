from __future__ import annotations

from dpp import rank_feed_via_dpp
from utils import build_demo_inputs


def choose_window_size(n_items: int) -> int:
    return max(1, min(5, n_items))


def main() -> list[int]:
    """Run a small demo of DPP-based diversified ranking."""

    qualities, features, params = build_demo_inputs()

    ranked = rank_feed_via_dpp(
        qualities=qualities,
        features=features,
        params=params,
        window_size=choose_window_size(len(qualities)),
    )
    print(ranked)
    return ranked


if __name__ == "__main__":
    main()
